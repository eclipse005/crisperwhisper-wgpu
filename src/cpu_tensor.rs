//! Minimal CPU tensors + gemm helpers (fp32). Layout follows row-major `[m, k]`
//! activations and `[out, in]` linear weights, matching HuggingFace `nn.Linear`.

use std::sync::atomic::{AtomicU8, Ordering};

use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use crate::int8_gemm::Int8Mat;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeType {
    /// Exact official-alignment path (product default).
    Fp32,
    /// Optional CT2-compatible full-model int8 (`quantization="int8"`):
    /// linears + conv + tied embeddings/logits, int8_float32 leftovers.
    /// Not the product default.
    Int8,
}

impl ComputeType {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fp32" | "float32" => Some(Self::Fp32),
            "int8" => Some(Self::Int8),
            _ => None,
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Self::Fp32 => 0,
            Self::Int8 => 1,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Int8,
            _ => Self::Fp32,
        }
    }

    pub fn packs_int8(self) -> bool {
        matches!(self, Self::Int8)
    }

    pub fn uses_int8(self) -> bool {
        self.packs_int8()
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32 (exact alignment path)",
            Self::Int8 => "int8 (CT2-compatible full model; int8_float32 leftovers; per-row absmax)",
        }
    }
}

static COMPUTE_TYPE: AtomicU8 = AtomicU8::new(0);

pub fn set_compute_type(v: ComputeType) {
    COMPUTE_TYPE.store(v.as_u8(), Ordering::Relaxed);
}

pub fn compute_type() -> ComputeType {
    ComputeType::from_u8(COMPUTE_TYPE.load(Ordering::Relaxed))
}

pub fn set_int8_enabled(v: bool) {
    set_compute_type(if v { ComputeType::Int8 } else { ComputeType::Fp32 });
}

pub fn int8_enabled() -> bool {
    compute_type() != ComputeType::Fp32
}

/// Dispatch: libmvec on the default fp32 path, CT2 `exp512_ps` / `vec_erf` on int8.
/// Switching INT8 leftovers to libmvec kept JFK gold-close but dropped
/// audio1 30s from 25/28 to 23/28 vs gold (reintroduced `Panthor!`) and
/// below CT2's 24/28. Do not leave that dispatch in place.
pub fn gelu_erf_inplace(xs: &mut [f32]) {
    if compute_type().uses_int8() {
        crate::ct2_math::gelu_erf_ct2_inplace(xs);
    } else {
        crate::simd_math::gelu_erf_inplace(xs);
    }
}

/// Dispatch: libmvec on the default fp32 path, CT2 softmax on int8.
/// See `gelu_erf_inplace` for why INT8 keeps the CT2 leftover softmax.
pub fn softmax_inplace_rows(mat: &mut [f32], rows: usize, cols: usize) {
    if compute_type().uses_int8() {
        crate::ct2_math::softmax_ct2_inplace_rows(mat, rows, cols);
    } else {
        crate::simd_math::softmax_inplace_rows(mat, rows, cols);
    }
}

pub struct CpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "CpuTensor len mismatch (shape {:?})", shape);
        Self { data, shape }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

pub struct CpuWeight {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl CpuWeight {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "CpuWeight len mismatch {rows}x{cols}");
        Self { data, rows, cols }
    }
}

pub struct Linear {
    pub weight: CpuWeight,
    pub bias: Option<Vec<f32>>,
    pub int8: Option<Int8Mat>,
    /// Encoder-sized fp32 weights packed into 16-column panels. Present when
    /// the default path packs at load; `weight.data` is then empty.
    pub fp32_packed: Option<crate::fp32_gemm::Fp32Packed>,
    pub is_encoder: bool,
}

impl Linear {
    pub fn pack_int8(&mut self) {
        let mut q = Int8Mat::quantize(
            &self.weight.data,
            self.weight.rows,
            self.weight.cols,
        );
        q.discard_unpacked_if_vnni();
        self.int8 = Some(q);
        // Drop the fp32 copy; the int8 path never reads it, and keeping both
        // would nearly double encoder/decoder RAM on a 15 GiB host.
        self.weight.data = Vec::new();
    }

    pub fn pack_fp32(&mut self) {
        let n = self.weight.rows;
        let k = self.weight.cols;
        if !crate::fp32_gemm::Fp32Packed::can_pack(n, k) {
            return;
        }
        self.fp32_packed = Some(crate::fp32_gemm::Fp32Packed::pack(&self.weight.data, n, k));
        // Packing is size-preserving when n % 16 == 0; drop the row-major
        // copy so encoder weights do not double in RAM.
        self.weight.data = Vec::new();
    }

    pub fn forward(&self, x: &[f32], seq: usize) -> Vec<f32> {
        if compute_type().uses_int8() {
            if let Some(q) = self.int8.as_ref() {
                return crate::int8_gemm::linear_int8(x, q, self.bias.as_deref(), seq);
            }
        }
        if let Some(p) = self.fp32_packed.as_ref() {
            let mut out = crate::fp32_gemm::linear_packed(x, p, seq);
            if let Some(bias) = &self.bias {
                apply_bias(&mut out, bias, seq);
            }
            return out;
        }
        let mut out = linear(x, &self.weight, seq);
        if let Some(bias) = &self.bias {
            apply_bias(&mut out, bias, seq);
        }
        out
    }
}

pub fn linear(x: &[f32], w: &CpuWeight, seq: usize) -> Vec<f32> {
    let k = w.cols;
    let n = w.rows;
    assert_eq!(x.len(), seq * k, "linear input len {} != seq {seq} * k {k}", x.len());
    if seq == 1 {
        return linear_gemv(x, w);
    }
    let mut out = vec![0.0f32; seq * n];
    gemm_row_major(&mut out, x, w, seq, 0.0);
    out
}

fn linear_gemv(x: &[f32], w: &CpuWeight) -> Vec<f32> {
    // m=1 packed gemm is slower (pack / thread-pool overhead each step).
    // Explicit AVX-512 FMA, 4 output rows at a time so `x` stays hot in L1.
    let n = w.rows;
    let k = w.cols;
    let mut out = vec![0.0f32; n];
    gemv_into(x, &w.data, n, k, &mut out);
    out
}

pub fn gemv_into(x: &[f32], w: &[f32], n: usize, k: usize, out: &mut [f32]) {
    assert_eq!(x.len(), k);
    assert_eq!(w.len(), n * k);
    assert_eq!(out.len(), n);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && k >= 16 {
            unsafe { gemv_avx512(x, w, n, k, out) };
            return;
        }
    }
    gemv_scalar(x, w, n, k, out);
}

fn gemv_scalar(x: &[f32], w: &[f32], n: usize, k: usize, out: &mut [f32]) {
    let chunk = (n / (rayon::current_num_threads() * 4)).max(64).min(2048);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
        let row0 = ci * chunk;
        for (offset, o) in slab.iter_mut().enumerate() {
            let w_row = &w[(row0 + offset) * k..(row0 + offset + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += x[j] * w_row[j];
            }
            *o = acc;
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemv_avx512(x: &[f32], w: &[f32], _n: usize, k: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    const NR: usize = 4;
    #[derive(Copy, Clone)]
    struct Ptr(*const f32);
    unsafe impl Send for Ptr {}
    unsafe impl Sync for Ptr {}
    impl Ptr {
        unsafe fn add(self, off: usize) -> *const f32 {
            self.0.add(off)
        }
    }
    let xp = Ptr(x.as_ptr());
    let wp = Ptr(w.as_ptr());
    let k16 = k & !15;
    // One chunk per rayon worker. Each output row's FMA sequence is
    // independent, so larger chunks are bit-identical to the old
    // 4-row tasks and drop the task-spawn cost on the vocab GEMV
    // (n ≈ 52k) that dominates every decoder step.
    let n = out.len();
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = {
        let raw = ((n + nthreads - 1) / nthreads).max(NR);
        raw.div_ceil(NR) * NR
    };
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
        let row0 = ci * chunk;
        let mut off = 0;
        while off < slab.len() {
            let r0 = row0 + off;
            let nr = (slab.len() - off).min(NR);
            // Prefetch the next output-row group; does not change the FMA order.
            let next = r0 + nr;
            if next < row0 + slab.len() {
                for r in 0..nr.min(slab.len() - (off + nr)) {
                    _mm_prefetch(wp.add((next + r) * k) as *const i8, _MM_HINT_T0);
                }
            }
            let mut accv = [_mm512_setzero_ps(); NR];
            let mut p = 0;
            while p + 64 <= k16 {
                for t in 0..4 {
                    let xv = _mm512_loadu_ps(xp.add(p + t * 16));
                    for r in 0..nr {
                        let wv = _mm512_loadu_ps(wp.add((r0 + r) * k + p + t * 16));
                        accv[r] = _mm512_fmadd_ps(xv, wv, accv[r]);
                    }
                }
                p += 64;
            }
            while p < k16 {
                let xv = _mm512_loadu_ps(xp.add(p));
                for r in 0..nr {
                    let wv = _mm512_loadu_ps(wp.add((r0 + r) * k + p));
                    accv[r] = _mm512_fmadd_ps(xv, wv, accv[r]);
                }
                p += 16;
            }
            for r in 0..nr {
                let mut acc = _mm512_reduce_add_ps(accv[r]);
                let wr = (r0 + r) * k;
                for j in k16..k {
                    acc += *xp.add(j) * *wp.add(wr + j);
                }
                slab[off + r] = acc;
            }
            off += nr;
        }
    });
}

pub fn gemm_row_major(out: &mut [f32], x: &[f32], w: &CpuWeight, m: usize, beta: f32) {
    let n = w.rows;
    let k = w.cols;
    unsafe {
        gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            beta != 0.0,
            x.as_ptr(),
            1,
            k as isize,
            w.data.as_ptr(),
            k as isize,
            1,
            beta,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
}

/// C[m, n] = A[m, k] @ B[k, n]  (both row-major).
pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut out = vec![0.0f32; m * n];
    unsafe {
        gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            1,
            n as isize,
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
    out
}

/// C[m, n] = A[m, k] @ B[n, k]^T  where B is stored row-major [n, k].
pub fn matmul_b_t(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    if m == 1 {
        let mut out = vec![0.0f32; n];
        gemv_into(a, b, n, k, &mut out);
        return out;
    }
    let mut out = vec![0.0f32; m * n];
    unsafe {
        gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
    out
}


/// C[m, n] = A[m, k] @ B[k, n] with explicit pointer strides (row-major: cs=1, rs=ncols).
pub unsafe fn gemm_strided(
    m: usize,
    n: usize,
    k: usize,
    c: *mut f32,
    c_cs: isize,
    c_rs: isize,
    a: *const f32,
    a_cs: isize,
    a_rs: isize,
    b: *const f32,
    b_cs: isize,
    b_rs: isize,
    parallelism: Parallelism,
) {
    gemm(
        m,
        n,
        k,
        c,
        c_cs,
        c_rs,
        false,
        a,
        a_cs,
        a_rs,
        b,
        b_cs,
        b_rs,
        0.0,
        1.0,
        false,
        false,
        false,
        parallelism,
    );
}

/// C[m, n] = A[m, k] @ B[n, k]^T with explicit pointer strides.
pub unsafe fn gemm_bt_strided(
    m: usize,
    n: usize,
    k: usize,
    c: *mut f32,
    c_cs: isize,
    c_rs: isize,
    a: *const f32,
    a_cs: isize,
    a_rs: isize,
    b: *const f32,
    b_cs: isize,
    b_rs: isize,
    parallelism: Parallelism,
) {
    gemm(
        m,
        n,
        k,
        c,
        c_cs,
        c_rs,
        false,
        a,
        a_cs,
        a_rs,
        b,
        b_cs,
        b_rs,
        0.0,
        1.0,
        false,
        false,
        false,
        parallelism,
    );
}

pub fn apply_bias(out: &mut [f32], bias: &[f32], seq: usize) {
    let n = bias.len();
    assert_eq!(out.len(), seq * n);
    if seq * n >= 2048 {
        out.par_chunks_mut(n).for_each(|row| {
            for (o, b) in row.iter_mut().zip(bias.iter()) {
                *o += *b;
            }
        });
    } else {
        for t in 0..seq {
            let row = &mut out[t * n..(t + 1) * n];
            for (o, b) in row.iter_mut().zip(bias.iter()) {
                *o += *b;
            }
        }
    }
}

/// Official-alignment LayerNorm (two-pass mean then variance). Used by the
/// default fp32 path and must not change.
pub fn layer_norm(x: &[f32], seq: usize, dim: usize, weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), seq * dim);
    assert_eq!(weight.len(), dim);
    assert_eq!(bias.len(), dim);
    let mut out = vec![0.0f32; x.len()];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && dim % 16 == 0 && dim >= 16 {
            unsafe { layer_norm_avx512(x, &mut out, dim, weight, bias, eps) };
            return out;
        }
    }
    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .for_each(|(dst, src)| layer_norm_row(dst, src, weight, bias, eps));
    out
}

fn layer_norm_row(dst: &mut [f32], src: &[f32], weight: &[f32], bias: &[f32], eps: f32) {
    let dim = src.len();
    let mut mean = 0.0f32;
    for &v in src {
        mean += v;
    }
    mean /= dim as f32;
    let mut var = 0.0f32;
    for &v in src {
        let d = v - mean;
        var += d * d;
    }
    var /= dim as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..dim {
        dst[i] = (src[i] - mean) * inv * weight[i] + bias[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn layer_norm_avx512(
    x: &[f32],
    out: &mut [f32],
    dim: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) {
    use std::arch::x86_64::*;
    #[derive(Copy, Clone)]
    struct CPtr(*const f32);
    unsafe impl Send for CPtr {}
    unsafe impl Sync for CPtr {}
    impl CPtr {
        unsafe fn add(self, off: usize) -> *const f32 {
            self.0.add(off)
        }
    }
    let wp = CPtr(weight.as_ptr());
    let bp = CPtr(bias.as_ptr());
    let dim_f = dim as f32;
    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .for_each(|(dst, src)| {
            let mut sum = _mm512_setzero_ps();
            let mut p = 0;
            while p < dim {
                sum = _mm512_add_ps(sum, _mm512_loadu_ps(src.as_ptr().add(p)));
                p += 16;
            }
            let mean = _mm512_reduce_add_ps(sum) / dim_f;
            let meanv = _mm512_set1_ps(mean);
            let mut sq = _mm512_setzero_ps();
            p = 0;
            while p < dim {
                let d = _mm512_sub_ps(_mm512_loadu_ps(src.as_ptr().add(p)), meanv);
                sq = _mm512_fmadd_ps(d, d, sq);
                p += 16;
            }
            let inv = 1.0 / (_mm512_reduce_add_ps(sq) / dim_f + eps).sqrt();
            let invv = _mm512_set1_ps(inv);
            p = 0;
            while p < dim {
                let v = _mm512_mul_ps(
                    _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(src.as_ptr().add(p)), meanv), invv),
                    _mm512_loadu_ps(wp.add(p)),
                );
                let v = _mm512_add_ps(v, _mm512_loadu_ps(bp.add(p)));
                _mm512_storeu_ps(dst.as_mut_ptr().add(p), v);
                p += 16;
            }
        });
}

/// CT2 CPU LayerNorm (`src/cpu/kernels.cc`, float kernel).
///
/// On CPU, CT2 `compute_type=int8` falls back to `int8_float32`: leftover
/// tensors (LN, bias, residuals) are float32, and BF16 LayerNorm is
/// GPU-only. This is the one-pass formula CT2 actually runs:
/// `mean = sum/n`, `var = max(sumsq/n - mean^2, 0)`,
/// `y = (x-mean) * rsqrt(var+eps) * gamma + beta`.
/// Used only on the optional int8 path.
pub fn layer_norm_ct2(
    x: &[f32],
    seq: usize,
    dim: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), seq * dim);
    assert_eq!(weight.len(), dim);
    assert_eq!(bias.len(), dim);
    let mut out = vec![0.0f32; x.len()];
    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 8-wide one-pass LN for the optional INT8 path only.
        // We do *not* use the CT2 4.7.1 AVX2 .so association (mul+add sumsq,
        // mean=sum*(1/n), fmsub variance, y=(x-mean)*(rstd*gamma)+beta):
        // that kernel is bit-exact vs this machine's CT2, but it moved INT8
        // further from official transformers fp32 gold and below CT2's own
        // closeness to gold (audio1 full 396/408 -> 388/408 vs gold).
        // Keep FMA sumsq, mean=sum/n, and ((x-mean)*rstd)*gamma+bias, which
        // stays closer to gold and no worse than CT2.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && dim % 8 == 0 && dim >= 8 {
            unsafe { layer_norm_ct2_avx2(x, &mut out, dim, weight, bias, eps) };
            return out;
        }
    }
    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .for_each(|(dst, src)| layer_norm_ct2_row(dst, src, weight, bias, eps));
    out
}

fn layer_norm_ct2_row(dst: &mut [f32], src: &[f32], weight: &[f32], bias: &[f32], eps: f32) {
    let dim = src.len() as f32;
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    for &v in src {
        sum += v;
        sumsq += v * v;
    }
    let mean = sum / dim;
    let variance = (sumsq / dim - mean * mean).max(0.0);
    let inv = 1.0 / (variance + eps).sqrt();
    for i in 0..src.len() {
        dst[i] = (src[i] - mean) * inv * weight[i] + bias[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn layer_norm_ct2_avx2(
    x: &[f32],
    out: &mut [f32],
    dim: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) {
    use std::arch::x86_64::*;
    #[derive(Copy, Clone)]
    struct CPtr(*const f32);
    unsafe impl Send for CPtr {}
    unsafe impl Sync for CPtr {}
    impl CPtr {
        unsafe fn add(self, off: usize) -> *const f32 {
            self.0.add(off)
        }
    }
    // 8-wide horizontal add: permute2f128, shuffle 0x4E, shuffle 0xB1.
    #[inline]
    unsafe fn reduce_add(v: __m256) -> f32 {
        let v1 = _mm256_permute2f128_ps(v, v, 0x1);
        let v = _mm256_add_ps(v, v1);
        let v1 = _mm256_shuffle_ps(v, v, 0x4E);
        let v = _mm256_add_ps(v, v1);
        let v1 = _mm256_shuffle_ps(v, v, 0xB1);
        let v = _mm256_add_ps(v, v1);
        _mm256_cvtss_f32(v)
    }
    let wp = CPtr(weight.as_ptr());
    let bp = CPtr(bias.as_ptr());
    let dim_f = dim as f32;
    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .for_each(|(dst, src)| {
            let mut sumv = _mm256_setzero_ps();
            let mut sqv = _mm256_setzero_ps();
            let mut p = 0;
            while p < dim {
                let v = _mm256_loadu_ps(src.as_ptr().add(p));
                sumv = _mm256_add_ps(sumv, v);
                sqv = _mm256_fmadd_ps(v, v, sqv);
                p += 8;
            }
            let mean = reduce_add(sumv) / dim_f;
            let variance = (reduce_add(sqv) / dim_f - mean * mean).max(0.0);
            let inv = 1.0 / (variance + eps).sqrt();
            let meanv = _mm256_set1_ps(mean);
            let invv = _mm256_set1_ps(inv);
            p = 0;
            while p < dim {
                let v = _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(src.as_ptr().add(p)), meanv), invv),
                    _mm256_loadu_ps(wp.add(p)),
                );
                let v = _mm256_add_ps(v, _mm256_loadu_ps(bp.add(p)));
                _mm256_storeu_ps(dst.as_mut_ptr().add(p), v);
                p += 8;
            }
        });
}


pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// 1D convolution in PyTorch layout: `weight` is `[out, in, k]`, `x` is `[in, t]`.
pub fn conv1d(
    x: &[f32],
    in_c: usize,
    time: usize,
    weight: &[f32],
    bias: &[f32],
    out_c: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), in_c * time);
    assert_eq!(weight.len(), out_c * in_c * k);
    assert_eq!(bias.len(), out_c);
    let out_t = (time + 2 * padding - k) / stride + 1;
    let k_col = in_c * k;
    // Unfold to [out_t, in_c * k], then gemm with W [out_c, in_c * k].
    let mut cols = vec![0.0f32; out_t * k_col];
    cols.par_chunks_mut(k_col).enumerate().for_each(|(t, row)| {
        let center = t * stride;
        for c in 0..in_c {
            for ki in 0..k {
                let src_t = center + ki;
                let val = if src_t < padding {
                    0.0
                } else {
                    let u = src_t - padding;
                    if u < time {
                        x[c * time + u]
                    } else {
                        0.0
                    }
                };
                row[c * k + ki] = val;
            }
        }
    });
    // Borrow weights — do not copy the conv kernels on every window.
    let mut out_tn = vec![0.0f32; out_t * out_c]; // [out_t, out_c]
    unsafe {
        gemm(
            out_t,
            out_c,
            k_col,
            out_tn.as_mut_ptr(),
            1,
            out_c as isize,
            false,
            cols.as_ptr(),
            1,
            k_col as isize,
            weight.as_ptr(),
            k_col as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
    apply_bias(&mut out_tn, bias, out_t);
    // Return [out_c, out_t] to stay in conv layout.
    let mut out = vec![0.0f32; out_c * out_t];
    out.par_chunks_mut(out_t).enumerate().for_each(|(c, col)| {
        for t in 0..out_t {
            col[t] = out_tn[t * out_c + c];
        }
    });
    out
}


/// CT2-style int8 conv: unfold to `[out_t, in*k]`, then the same per-row
/// int8 GEMM used for linears. `weight_q` is packed from `[out, in, k]`
/// (row-major, already the `[out, in*k]` view).
pub fn conv1d_int8(
    x: &[f32],
    in_c: usize,
    time: usize,
    weight_q: &crate::int8_gemm::Int8Mat,
    bias: &[f32],
    out_c: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), in_c * time);
    assert_eq!(weight_q.rows, out_c);
    assert_eq!(weight_q.cols, in_c * k);
    assert_eq!(bias.len(), out_c);
    let out_t = (time + 2 * padding - k) / stride + 1;
    let k_col = in_c * k;
    let mut cols = vec![0.0f32; out_t * k_col];
    cols.par_chunks_mut(k_col).enumerate().for_each(|(t, row)| {
        let center = t * stride;
        for c in 0..in_c {
            for ki in 0..k {
                let src_t = center + ki;
                let val = if src_t < padding {
                    0.0
                } else {
                    let u = src_t - padding;
                    if u < time {
                        x[c * time + u]
                    } else {
                        0.0
                    }
                };
                row[c * k + ki] = val;
            }
        }
    });
    let out_tn = crate::int8_gemm::linear_int8(&cols, weight_q, Some(bias), out_t);
    let mut out = vec![0.0f32; out_c * out_t];
    out.par_chunks_mut(out_t).enumerate().for_each(|(c, col)| {
        for t in 0..out_t {
            col[t] = out_tn[t * out_c + c];
        }
    });
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_matches_naive_closely() {
        let n = 128;
        let k = 1280;
        let mut w = vec![0.0f32; n * k];
        let mut x = vec![0.0f32; k];
        for i in 0..w.len() {
            w[i] = ((i as i32 % 17) - 8) as f32 * 0.03;
        }
        for i in 0..k {
            x[i] = ((i as i32 % 11) - 5) as f32 * 0.02;
        }
        let mut got = vec![0.0f32; n];
        gemv_into(&x, &w, n, k, &mut got);
        let mut max = 0.0f32;
        for r in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += x[j] * w[r * k + j];
            }
            max = max.max((got[r] - acc).abs());
        }
        assert!(max < 2e-4, "gemv max abs err {max}");
    }

    #[test]
    fn layer_norm_ct2_matches_one_pass_formula() {
        // dim=16 hits the AVX2 kernel; a few ulps vs a serial one-pass sum is ok.
        let dim = 16;
        let src: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.37 - 2.1).collect();
        let w: Vec<f32> = (0..dim).map(|i| 0.8 + i as f32 * 0.02).collect();
        let b: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01 - 0.05).collect();
        let eps = 1e-5f32;
        let got = layer_norm_ct2(&src, 1, dim, &w, &b, eps);
        let mut sum = 0.0f32;
        let mut sumsq = 0.0f32;
        for &v in &src {
            sum += v;
            sumsq += v * v;
        }
        let mean = sum / dim as f32;
        let variance = (sumsq / dim as f32 - mean * mean).max(0.0);
        let inv = 1.0 / (variance + eps).sqrt();
        for i in 0..dim {
            let exp = (src[i] - mean) * inv * w[i] + b[i];
            let d = (got[i] - exp).abs();
            assert!(d < 2e-6, "idx {i}: got {} exp {exp} d={d}", got[i]);
        }
    }
}
