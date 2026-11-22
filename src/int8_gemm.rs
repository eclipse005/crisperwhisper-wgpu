//! CT2-compatible weight-int8 / dynamic-activation-int8 GEMM.
//!
//! Weight quant matches CTranslate2 `quantization="int8"`:
//! per-output-row symmetric absmax, `scale = 127 / amax`, `q = rint(w * scale)`
//! (ties-to-even). Activations are quantized the same way at each forward
//! (per row over the full K), then shifted to u8 for AVX-512 VNNI `vpdpbusd`
//! with the standard 128 * row_sum compensation — the same integer contract
//! as Intel MKL `cblas_gemm_s8u8s32` (the backend this machine's CT2 4.7.1
//! actually selects: unpacked, `CblasRowOffset`, ao=bo=0, alpha=1, beta=0).
//! Empirically the int32 result is bit-identical to that MKL entry, including
//! K=1280/5120 and accumulators above 2^24.
//!
//! The compute graph matches CT2: a full int32 GEMM, then a separate
//! `dequantize_gemm_output` pass. On this host CT2's Selected ISA is AVX2,
//! so dequant and activation quant are 8-wide (`_mm256_div_ps`,
//! `_mm256_round_ps` + `_mm256_cvttps_epi32`, `reduce_m256` amax). They are
//! not 16-wide AVX-512 float kernels.
//!
//! Dequant: `y = (acc_i32 * (1 / scale_a)) / scale_b + bias`
//! (reciprocal of the activation scale, then divide by the weight scale).
//! This is not the same rounding as `acc / (scale_a * scale_b)`.
//!
//! CT2 on CPU resolves `compute_type=int8` to `int8_float32`: leftover
//! tensors stay float32 (BF16 LayerNorm / GELU / Dequantize are GPU-only).
//! This kernel therefore does **not** snap outputs to bf16.
//!
//! This path is optional and must not become the product default. The fp32
//! implementation remains the exact official-transformers gold route.

use rayon::prelude::*;

const VNNI_N: usize = 16; // 16 i32 accumulators per zmm
const VNNI_K: usize = 4;  // 4 u8*i8 products per lane
const MR: usize = 8;      // register-blocked rows of A
const NR_TILES: usize = 2; // 32 output columns

pub const DEFAULT_GROUP: usize = 64;

pub struct Int8Mat {
    pub rows: usize,
    pub cols: usize,
    pub cols_padded: usize,
    pub rows_padded: usize,
    pub group: usize,
    pub n_groups: usize,
    /// Row-major i8 `[rows_padded, cols_padded]` (scalar path).
    pub data: Vec<i8>,
    /// VNNI packed i32: tiles of `[n_tiles, k_tiles, 16]`.
    pub packed: Vec<i32>,
    /// Per-output-row, per-K-group scale: `[rows_padded, n_groups]`.
    pub scale: Vec<f32>,
    /// `sum_{k in g} q[r, k]` for the u8-offset VNNI correction: `[rows_padded, n_groups]`.
    pub row_sum: Vec<i32>,
}

impl Int8Mat {
    pub fn quantize(weight: &[f32], rows: usize, cols: usize) -> Self {
        // CT2 int8: one scale per output row over the full K (pad to VNNI_K).
        let cols_padded = cols.div_ceil(VNNI_K) * VNNI_K;
        Self::quantize_grouped(weight, rows, cols, cols_padded)
    }

    /// Dequantize one packed weight row into `dst` (`dst.len() == cols`).
    /// `scale` is the CT2 quantize scale (127/amax).
    /// Works from the unpacked i8 matrix or, if that was dropped on a VNNI
    /// host to save RAM, from the VNNI packed tiles (same bytes).
    pub fn dequant_row(&self, row: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.cols);
        assert!(row < self.rows);
        // scale layout: [g, rows_padded], n_groups is 1 on the CT2 path.
        let mut invs = vec![0.0f32; self.n_groups];
        for g in 0..self.n_groups {
            let s = self.scale[g * self.rows_padded + row];
            invs[g] = if s == 0.0 { 0.0 } else { 1.0 / s };
        }
        if !self.data.is_empty() {
            let src = &self.data[row * self.cols_padded..row * self.cols_padded + self.cols];
            for (j, &q) in src.iter().enumerate() {
                let g = (j / self.group).min(self.n_groups - 1);
                dst[j] = q as f32 * invs[g];
            }
            return;
        }
        let k_tiles = self.cols_padded / VNNI_K;
        let nt = row / VNNI_N;
        let lane = row % VNNI_N;
        for j in 0..self.cols {
            let kt = j / VNNI_K;
            let slot = ((nt * k_tiles) + kt) * VNNI_N + lane;
            let q = self.packed[slot].to_le_bytes()[j % VNNI_K] as i8;
            let g = (j / self.group).min(self.n_groups - 1);
            dst[j] = q as f32 * invs[g];
        }
    }

    /// On AVX-512 VNNI hosts the live GEMM only reads `packed`. Drop the
    /// duplicate row-major i8 copy (~same size as `packed`) after load.
    pub fn discard_unpacked_if_vnni(&mut self) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
                self.data = Vec::new();
            }
        }
    }

    pub fn quantize_grouped(weight: &[f32], rows: usize, cols: usize, group: usize) -> Self {
        assert_eq!(weight.len(), rows * cols);
        let cols_padded = cols.div_ceil(VNNI_K) * VNNI_K;
        let rows_padded = rows.div_ceil(VNNI_N) * VNNI_N;
        let group = group.clamp(VNNI_K, cols_padded) / VNNI_K * VNNI_K;
        assert!(cols_padded % group == 0, "cols_padded {cols_padded} not divisible by group {group}");
        let n_groups = cols_padded / group;
        let mut data = vec![0i8; rows_padded * cols_padded];
        let mut scale = vec![0.0f32; rows_padded * n_groups];
        let mut row_sum = vec![0i32; rows_padded * n_groups];
        for r in 0..rows {
            let src = &weight[r * cols..(r + 1) * cols];
            for g in 0..n_groups {
                let start = g * group;
                let end = (start + group).min(cols);
                if start >= cols {
                    scale[g * rows_padded + r] = 1e-12;
                    continue;
                }
                let mut max = 0.0f32;
                for &v in &src[start..end] {
                    let a = v.abs();
                    if a > max {
                        max = a;
                    }
                }
                // CT2: scale = 127/amax (amax==0 -> 127 so scale=1), q = rint(w*scale).
                let s = if max == 0.0 { 1.0 } else { 127.0 / max };
                scale[g * rows_padded + r] = s;
                let mut sum = 0i32;
                let dst_end = (start + group).min(cols_padded);
                // only the real (unpadded) range of this group in the source
                for p in start..end {
                    let q = (src[p] * s).round_ties_even().clamp(-127.0, 127.0) as i8;
                    data[r * cols_padded + p] = q;
                    sum += q as i32;
                }
                let _ = dst_end;
                row_sum[g * rows_padded + r] = sum;
            }
        }
        let n_tiles = rows_padded / VNNI_N;
        let k_tiles = cols_padded / VNNI_K;
        let mut packed = vec![0i32; n_tiles * k_tiles * VNNI_N];
        for nt in 0..n_tiles {
            for kt in 0..k_tiles {
                let slot = ((nt * k_tiles) + kt) * VNNI_N;
                for t in 0..VNNI_N {
                    let r = nt * VNNI_N + t;
                    let p = kt * VNNI_K;
                    let base = r * cols_padded + p;
                    packed[slot + t] = i32::from_le_bytes([
                        data[base] as u8,
                        data[base + 1] as u8,
                        data[base + 2] as u8,
                        data[base + 3] as u8,
                    ]);
                }
            }
        }
        Self {
            rows,
            cols,
            cols_padded,
            rows_padded,
            group,
            n_groups,
            data,
            packed,
            scale,
            row_sum,
        }
    }
}

pub fn linear_int8(x: &[f32], w: &Int8Mat, bias: Option<&[f32]>, seq: usize) -> Vec<f32> {
    let k = w.cols;
    let kp = w.cols_padded;
    let n = w.rows;
    assert_eq!(x.len(), seq * k);
    if let Some(b) = bias {
        assert_eq!(b.len(), n);
    }
    let ng = w.n_groups;
    let mut a = vec![0u8; seq * kp];
    let mut a_scale = vec![0.0f32; seq * ng];
    // CT2 `quantize_s8` with shift_to_uint8 (one scale per row over full K).
    a.par_chunks_mut(kp)
        .zip(a_scale.par_chunks_mut(ng))
        .zip(x.par_chunks(k))
        .for_each(|((dst, scale), src)| {
            scale[0] = quantize_u8_row(src, &mut dst[..k]);
            for g in 1..ng {
                scale[g] = 1e-12;
            }
        });

    let mut out = vec![0.0f32; seq * n];
    gemm_u8_i8(&a, &a_scale, w, bias, seq, &mut out);
    out
}

/// CT2 `quantize_s8_row` with `shift_to_uint8=true`, `round_before_cast=true`.
/// AVX2 when available: `reduce_amax` + `round_ps(v*scale - int8_min)` + `cvttps`.
fn quantize_u8_row(src: &[f32], dst: &mut [u8]) -> f32 {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && src.len() >= 1 {
            unsafe {
                return quantize_u8_row_avx2(src.as_ptr(), dst.as_mut_ptr(), src.len());
            }
        }
    }
    quantize_u8_row_scalar(src, dst)
}

fn quantize_u8_row_scalar(src: &[f32], dst: &mut [u8]) -> f32 {
    let mut max = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > max {
            max = a;
        }
    }
    let s = if max == 0.0 { 1.0 } else { 127.0 / max };
    for (j, &v) in src.iter().enumerate() {
        // CT2: round(v * scale - int8_min) with nearby-even, then trunc to int.
        dst[j] = (v * s + 128.0).round_ties_even() as u8;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_max_m256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let v1 = _mm256_permute2f128_ps(v, v, 0x1);
    let v = _mm256_max_ps(v, v1);
    let v1 = _mm256_shuffle_ps(v, v, 0x4E);
    let v = _mm256_max_ps(v, v1);
    let v1 = _mm256_shuffle_ps(v, v, 0xB1);
    let v = _mm256_max_ps(v, v1);
    _mm256_cvtss_f32(v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_u8_row_avx2(x: *const f32, dst: *mut u8, depth: usize) -> f32 {
    use std::arch::x86_64::*;
    const ROUND_NEAR: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let mut maxv = _mm256_setzero_ps();
    let mut j = 0;
    while j + 8 <= depth {
        let v = _mm256_loadu_ps(x.add(j));
        let a = _mm256_andnot_ps(_mm256_set1_ps(-0.0), v);
        maxv = _mm256_max_ps(maxv, a);
        j += 8;
    }
    let mut amax = reduce_max_m256(maxv);
    while j < depth {
        amax = amax.max((*x.add(j)).abs());
        j += 1;
    }
    let s = if amax == 0.0 { 1.0 } else { 127.0 / amax };
    let vec_s = _mm256_set1_ps(s);
    let vec_min = _mm256_set1_ps(-128.0); // int8_min; sub means +128
    j = 0;
    while j + 8 <= depth {
        let v = _mm256_loadu_ps(x.add(j));
        let v = _mm256_sub_ps(_mm256_mul_ps(v, vec_s), vec_min);
        let v = _mm256_round_ps(v, ROUND_NEAR);
        let i = _mm256_cvttps_epi32(v);
        store_u8_from_i32(i, dst.add(j), 8);
        j += 8;
    }
    if j < depth {
        let rem = depth - j;
        let mut tmp = [0.0f32; 8];
        for t in 0..rem {
            tmp[t] = *x.add(j + t);
        }
        let v = _mm256_loadu_ps(tmp.as_ptr());
        let v = _mm256_sub_ps(_mm256_mul_ps(v, vec_s), vec_min);
        let v = _mm256_round_ps(v, ROUND_NEAR);
        let i = _mm256_cvttps_epi32(v);
        store_u8_from_i32(i, dst.add(j), rem);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn store_u8_from_i32(v: std::arch::x86_64::__m256i, dst: *mut u8, count: usize) {
    use std::arch::x86_64::*;
    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr().cast(), v);
    for t in 0..count {
        *dst.add(t) = tmp[t] as u8;
    }
}

fn gemm_u8_i8(
    a: &[u8],
    a_scale: &[f32],
    w: &Int8Mat,
    bias: Option<&[f32]>,
    m: usize,
    out: &mut [f32],
) {
    let n = w.rows;
    // CT2 path is one scale per output row (full-K). Keep a scalar fallback
    // for the unused grouped helper so it still has a defined result.
    if w.n_groups == 1 {
        let mut acc = vec![0i32; m * n];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
                unsafe { gemm_u8_i8_vnni(a, w, m, &mut acc) };
                dequantize_gemm_output_ct2(&acc, a_scale, w, bias, m, out);
                return;
            }
        }
        gemm_u8_i8_scalar_i32(a, w, m, &mut acc);
        dequantize_gemm_output_ct2(&acc, a_scale, w, bias, m, out);
        return;
    }
    gemm_u8_i8_scalar_grouped(a, a_scale, w, bias, out);
}

fn gemm_u8_i8_scalar_i32(a: &[u8], w: &Int8Mat, _m: usize, acc_out: &mut [i32]) {
    const OFFSET: i32 = 128;
    let n = w.rows;
    let k = w.cols_padded;
    acc_out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let ai = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let bj = &w.data[j * k..(j + 1) * k];
            let mut acc = 0i32;
            for p in 0..k {
                acc += ai[p] as i32 * bj[p] as i32;
            }
            acc -= OFFSET * w.row_sum[j];
            row[j] = acc;
        }
    });
}

fn gemm_u8_i8_scalar_grouped(
    a: &[u8],
    a_scale: &[f32],
    w: &Int8Mat,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    const OFFSET: i32 = 128;
    let n = w.rows;
    let k = w.cols_padded;
    let ng = w.n_groups;
    let group = w.group;
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let ai = &a[i * k..(i + 1) * k];
        let sa = &a_scale[i * ng..(i + 1) * ng];
        for j in 0..n {
            let bj = &w.data[j * k..(j + 1) * k];
            let mut acc_f = 0.0f32;
            for g in 0..ng {
                let start = g * group;
                let end = start + group;
                let mut acc = 0i32;
                for p in start..end {
                    acc += ai[p] as i32 * bj[p] as i32;
                }
                acc -= OFFSET * w.row_sum[g * w.rows_padded + j];
                acc_f += (acc as f32) * (1.0 / sa[g]) / w.scale[g * w.rows_padded + j];
            }
            if let Some(bias) = bias {
                acc_f += bias[j];
            }
            row[j] = acc_f;
        }
    });
}

/// CT2 `dequantize_gemm_output` (Selected ISA AVX2): 8-wide
/// `cvtepi32_ps`, mul by `1/a_scale`, div by `b_scale`, optional add bias.
fn dequantize_gemm_output_ct2(
    acc: &[i32],
    a_scale: &[f32],
    w: &Int8Mat,
    bias: Option<&[f32]>,
    m: usize,
    out: &mut [f32],
) {
    let n = w.rows;
    debug_assert_eq!(w.n_groups, 1);
    debug_assert_eq!(acc.len(), m * n);
    debug_assert_eq!(out.len(), m * n);
    debug_assert_eq!(a_scale.len(), m);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                dequantize_gemm_output_avx2(acc, a_scale, &w.scale[..n], bias, n, out);
            }
            return;
        }
    }
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let sa = a_scale[i];
        let inv_sa = 1.0 / sa;
        let c = &acc[i * n..(i + 1) * n];
        for j in 0..n {
            let mut v = (c[j] as f32) * inv_sa / w.scale[j];
            if let Some(b) = bias {
                v += b[j];
            }
            row[j] = v;
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_gemm_output_avx2(
    acc: &[i32],
    a_scale: &[f32],
    b_scale: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    out.par_chunks_mut(n)
        .zip(acc.par_chunks(n))
        .zip(a_scale.par_iter())
        .for_each(|((y, c), &sa)| {
            let sa = _mm256_set1_ps(1.0 / sa);
            let mut j = 0;
            while j + 8 <= n {
                let mut v = _mm256_cvtepi32_ps(_mm256_loadu_si256(c.as_ptr().add(j).cast()));
                v = _mm256_mul_ps(v, sa);
                v = _mm256_div_ps(v, _mm256_loadu_ps(b_scale.as_ptr().add(j)));
                if let Some(b) = bias {
                    v = _mm256_add_ps(v, _mm256_loadu_ps(b.as_ptr().add(j)));
                }
                _mm256_storeu_ps(y.as_mut_ptr().add(j), v);
                j += 8;
            }
            if j < n {
                let rem = n - j;
                // CT2 leftover: pad c with 0, pad b_scale with 0, store only rem.
                let mut ctmp = [0i32; 8];
                let mut stmp = [0.0f32; 8];
                let mut btmp = [0.0f32; 8];
                for t in 0..rem {
                    ctmp[t] = c[j + t];
                    stmp[t] = b_scale[j + t];
                    if let Some(b) = bias {
                        btmp[t] = b[j + t];
                    }
                }
                let mut v = _mm256_cvtepi32_ps(_mm256_loadu_si256(ctmp.as_ptr().cast()));
                v = _mm256_mul_ps(v, sa);
                v = _mm256_div_ps(v, _mm256_loadu_ps(stmp.as_ptr()));
                if bias.is_some() {
                    v = _mm256_add_ps(v, _mm256_loadu_ps(btmp.as_ptr()));
                }
                let mut ytmp = [0.0f32; 8];
                _mm256_storeu_ps(ytmp.as_mut_ptr(), v);
                y[j..n].copy_from_slice(&ytmp[..rem]);
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn compensate_i32(
    acc: std::arch::x86_64::__m512i,
    row_sum: *const i32,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    let corr = _mm512_mullo_epi32(_mm512_loadu_si512(row_sum.cast()), _mm512_set1_epi32(128));
    _mm512_sub_epi32(acc, corr)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn store16i(dst: *mut i32, v: std::arch::x86_64::__m512i, j0: usize, n: usize) {
    use std::arch::x86_64::*;
    let remain = n.saturating_sub(j0);
    if remain == 0 {
        return;
    }
    if remain >= 16 {
        _mm512_storeu_si512(dst.cast(), v);
    } else {
        let mask = (1u16 << remain) - 1;
        _mm512_mask_storeu_epi32(dst.cast(), mask, v);
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
struct Pack(*const i32, *const i32, *const u8);
#[cfg(target_arch = "x86_64")]
unsafe impl Send for Pack {}
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for Pack {}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn gemm_u8_i8_vnni(a: &[u8], w: &Int8Mat, _m: usize, out: &mut [i32]) {
    let n = w.rows;
    let k = w.cols_padded;
    let k_tiles = k / VNNI_K;
    let n_tiles = w.rows_padded / VNNI_N;
    let pack = Pack(w.packed.as_ptr(), w.row_sum.as_ptr(), a.as_ptr());
    let job_rows = MR * 4;
    out.par_chunks_mut(n * job_rows)
        .enumerate()
        .for_each(|(job, slab)| unsafe {
            vnni_job(pack, job, slab, n, k, k_tiles, n_tiles, job_rows);
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn vnni_job(
    pack: Pack,
    job: usize,
    slab: &mut [i32],
    n: usize,
    k: usize,
    k_tiles: usize,
    n_tiles: usize,
    job_rows: usize,
) {
    use std::arch::x86_64::*;
    let i0 = job * job_rows;
    let rows = slab.len() / n;
    let mut i = 0;
    while i + MR <= rows {
        let base = i0 + i;
        let mut nt = 0;
        while nt + NR_TILES <= n_tiles {
            let mut acc0 = [_mm512_setzero_si512(); MR];
            let mut acc1 = [_mm512_setzero_si512(); MR];
            let p0 = pack.0.add((nt * k_tiles) * VNNI_N);
            let p1 = pack.0.add(((nt + 1) * k_tiles) * VNNI_N);
            let j0 = nt * VNNI_N;
            let j1 = j0 + VNNI_N;
            for kt in 0..k_tiles {
                let b0 = _mm512_loadu_si512(p0.add(kt * VNNI_N).cast());
                let b1 = _mm512_loadu_si512(p1.add(kt * VNNI_N).cast());
                for r in 0..MR {
                    let ap = pack.2.add((base + r) * k + kt * VNNI_K);
                    let a4 = u32::from_le_bytes([*ap, *ap.add(1), *ap.add(2), *ap.add(3)]);
                    let av = _mm512_set1_epi32(a4 as i32);
                    acc0[r] = _mm512_dpbusd_epi32(acc0[r], av, b0);
                    acc1[r] = _mm512_dpbusd_epi32(acc1[r], av, b1);
                }
            }
            if j0 < n {
                for r in 0..MR {
                    let dst0 = slab.as_mut_ptr().add(i * n + r * n + j0);
                    store16i(dst0, compensate_i32(acc0[r], pack.1.add(j0)), j0, n);
                    if j1 < n {
                        let dst1 = slab.as_mut_ptr().add(i * n + r * n + j1);
                        store16i(dst1, compensate_i32(acc1[r], pack.1.add(j1)), j1, n);
                    }
                }
            }
            nt += NR_TILES;
        }
        while nt < n_tiles {
            let mut acc = [_mm512_setzero_si512(); MR];
            let p0 = pack.0.add((nt * k_tiles) * VNNI_N);
            let j0 = nt * VNNI_N;
            for kt in 0..k_tiles {
                let b0 = _mm512_loadu_si512(p0.add(kt * VNNI_N).cast());
                for r in 0..MR {
                    let ap = pack.2.add((base + r) * k + kt * VNNI_K);
                    let a4 = u32::from_le_bytes([*ap, *ap.add(1), *ap.add(2), *ap.add(3)]);
                    let av = _mm512_set1_epi32(a4 as i32);
                    acc[r] = _mm512_dpbusd_epi32(acc[r], av, b0);
                }
            }
            if j0 < n {
                for r in 0..MR {
                    let dst = slab.as_mut_ptr().add(i * n + r * n + j0);
                    store16i(dst, compensate_i32(acc[r], pack.1.add(j0)), j0, n);
                }
            }
            nt += 1;
        }
        i += MR;
    }
    while i < rows {
        let base = i0 + i;
        let ai = pack.2.add(base * k);
        for nt in 0..n_tiles {
            let j0 = nt * VNNI_N;
            let p0 = pack.0.add((nt * k_tiles) * VNNI_N);
            let mut acc = _mm512_setzero_si512();
            for kt in 0..k_tiles {
                let ap = ai.add(kt * VNNI_K);
                let a4 = u32::from_le_bytes([*ap, *ap.add(1), *ap.add(2), *ap.add(3)]);
                let av = _mm512_set1_epi32(a4 as i32);
                let b0 = _mm512_loadu_si512(p0.add(kt * VNNI_N).cast());
                acc = _mm512_dpbusd_epi32(acc, av, b0);
            }
            if j0 < n {
                let dst = slab.as_mut_ptr().add(i * n + j0);
                store16i(dst, compensate_i32(acc, pack.1.add(j0)), j0, n);
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_err(m: usize, n: usize, k: usize) -> f32 {
        let mut w = vec![0.0f32; n * k];
        let mut x = vec![0.0f32; m * k];
        for i in 0..w.len() {
            w[i] = ((i as i32 % 17) - 8) as f32 * 0.11;
        }
        for i in 0..x.len() {
            x[i] = ((i as i32 % 11) - 5) as f32 * 0.07;
        }
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let qm = Int8Mat::quantize(&w, n, k);
        let got = linear_int8(&x, &qm, Some(&bias), m);
        let mut max = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += x[i * k + p] * w[j * k + p];
                }
                acc += bias[j];
                let d = (got[i * n + j] - acc).abs();
                if d > max {
                    max = d;
                }
            }
        }
        max
    }

    #[test]
    fn ct2_weight_quant_matches_rint_per_row() {
        // scale = 127/amax, q = rint(w*scale) ties-to-even, same as numpy / CT2.
        let rows = 3;
        let cols = 8;
        let mut w = vec![0.0f32; rows * cols];
        w[0] = 1.0;
        w[1] = -0.5;
        w[2] = 0.25;
        // Construct a halfway case: amax=1 so scale=127, 0.5/127 is not halfway;
        // use values whose product with scale is exactly k+0.5.
        // 1.5/127 * 127 = 1.5 -> rint ties to 2; 2.5 -> 2.
        w[3] = 1.5 / 127.0;
        w[4] = 2.5 / 127.0;
        w[5] = -1.5 / 127.0;
        w[8] = 2.0; // row 1 amax=2, scale=63.5
        w[9] = -2.0;
        w[16] = 0.0; // row 2 all zero -> amax forced to 127, scale=1, q=0
        let qm = Int8Mat::quantize(&w, rows, cols);
        assert_eq!(qm.n_groups, 1);
        // row 0
        let s0 = qm.scale[0];
        assert!((s0 - 127.0).abs() < 1e-6, "s0={s0}");
        let r0 = &qm.data[0 * qm.cols_padded..0 * qm.cols_padded + cols];
        assert_eq!(r0[0], 127);
        assert_eq!(r0[1], -64); // rint(-0.5*127)=rint(-63.5)=-64 (ties away? -63.5 ties to even -64)
        // -0.5*127 = -63.5, ties-to-even -> -64
        assert_eq!(r0[3], 2); // rint(1.5)=2
        assert_eq!(r0[4], 2); // rint(2.5)=2
        assert_eq!(r0[5], -2); // rint(-1.5)=-2
        let s1 = qm.scale[1];
        assert!((s1 - 63.5).abs() < 1e-5, "s1={s1}");
        let s2 = qm.scale[2];
        assert!((s2 - 1.0).abs() < 1e-6, "s2={s2}");
        let r2 = &qm.data[2 * qm.cols_padded..2 * qm.cols_padded + cols];
        assert!(r2.iter().all(|&q| q == 0), "zero row {r2:?}");
    }

    /// oneDNN s8u8s32 integer contract + CT2 dequant, compared to the live kernel.
    #[test]
    fn linear_int8_matches_ct2_dequant_and_s8u8s32_acc() {
        let m = 5;
        let n = 32;
        let k = 64;
        let mut w = vec![0.0f32; n * k];
        let mut x = vec![0.0f32; m * k];
        for i in 0..w.len() {
            w[i] = ((i * 13) % 25) as f32 * 0.05 - 0.6;
        }
        for i in 0..x.len() {
            x[i] = ((i * 7) % 19) as f32 * 0.04 - 0.3;
        }
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 0.04).collect();
        let qm = Int8Mat::quantize(&w, n, k);
        assert_eq!(qm.n_groups, 1);
        let got = linear_int8(&x, &qm, Some(&bias), m);
        let kp = qm.cols_padded;
        let mut max_d = 0.0f32;
        for i in 0..m {
            let src = &x[i * k..(i + 1) * k];
            let mut amax = 0.0f32;
            for &v in src {
                amax = amax.max(v.abs());
            }
            let sa = if amax == 0.0 { 1.0 } else { 127.0 / amax };
            let mut au = vec![0u8; kp];
            for p in 0..k {
                au[p] = (src[p] * sa + 128.0).round_ties_even().clamp(0.0, 255.0) as u8;
            }
            for j in 0..n {
                let bj = &qm.data[j * kp..j * kp + kp];
                let mut acc_u = 0i32;
                let mut acc_s = 0i32;
                let mut wsum = 0i32;
                for p in 0..kp {
                    let u = au[p] as i32;
                    let s = bj[p] as i32;
                    acc_u += u * s;
                    acc_s += (u - 128) * s;
                    wsum += s;
                }
                let acc_comp = acc_u - 128 * wsum;
                assert_eq!(acc_comp, acc_s, "s8u8s32 compensation mismatch at {i},{j}");
                assert_eq!(wsum, qm.row_sum[j], "row_sum mismatch at col {j}");
                let sb = qm.scale[j];
                let exp = (acc_comp as f32) * (1.0 / sa) / sb + bias[j];
                let d = (got[i * n + j] - exp).abs();
                if d > max_d {
                    max_d = d;
                }
            }
        }
        assert!(max_d < 1e-5, "CT2 dequant max abs err {max_d}");
    }


    #[test]
    fn dequant_row_from_packed_matches_unpacked() {
        let n = 33;
        let k = 20;
        let mut w = vec![0.0f32; n * k];
        for i in 0..w.len() {
            w[i] = ((i * 13) % 25) as f32 * 0.05 - 0.6;
        }
        let mut qm = Int8Mat::quantize(&w, n, k);
        let mut from_data = vec![0.0f32; k];
        qm.dequant_row(17, &mut from_data);
        qm.discard_unpacked_if_vnni();
        if qm.data.is_empty() {
            let mut from_packed = vec![0.0f32; k];
            qm.dequant_row(17, &mut from_packed);
            assert_eq!(from_data, from_packed);
            // first and last rows too (tile edges)
            for row in [0, n - 1] {
                let mut a = vec![0.0f32; k];
                let mut b = vec![0.0f32; k];
                // restore from packed only
                qm.dequant_row(row, &mut b);
                // rebuild reference from a fresh matrix that still has data
                let qm2 = Int8Mat::quantize(&w, n, k);
                qm2.dequant_row(row, &mut a);
                assert_eq!(a, b, "row {row}");
            }
        }
    }

    #[test]
    fn linear_int8_dequant_is_bit_exact_vs_scalar_formula() {
        // Odd n exercises the 8-wide leftover path (n % 8 == 1, also % 16 != 0).
        for &(m, n, k) in &[(3, 17, 64), (1, 25, 1280), (8, 41, 32)] {
            let mut w = vec![0.0f32; n * k];
            let mut x = vec![0.0f32; m * k];
            for i in 0..w.len() {
                w[i] = ((i * 13) % 25) as f32 * 0.05 - 0.6;
            }
            for i in 0..x.len() {
                x[i] = ((i * 7) % 19) as f32 * 0.04 - 0.3;
            }
            let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 0.04).collect();
            let qm = Int8Mat::quantize(&w, n, k);
            assert_eq!(qm.n_groups, 1);
            let got = linear_int8(&x, &qm, Some(&bias), m);
            let kp = qm.cols_padded;
            for i in 0..m {
                let src = &x[i * k..(i + 1) * k];
                let mut amax = 0.0f32;
                for &v in src {
                    amax = amax.max(v.abs());
                }
                let sa = if amax == 0.0 { 1.0 } else { 127.0 / amax };
                let mut au = vec![0u8; kp];
                for p in 0..k {
                    au[p] = (src[p] * sa + 128.0).round_ties_even() as u8;
                }
                for j in 0..n {
                    let bj = &qm.data[j * kp..(j + 1) * kp];
                    let mut acc = 0i32;
                    for p in 0..kp {
                        acc += au[p] as i32 * bj[p] as i32;
                    }
                    acc -= 128 * qm.row_sum[j];
                    let exp = (acc as f32) * (1.0 / sa) / qm.scale[j] + bias[j];
                    assert_eq!(
                        got[i * n + j].to_bits(),
                        exp.to_bits(),
                        "bit mismatch at {i},{j} got={} exp={} (m={m} n={n} k={k})",
                        got[i * n + j],
                        exp
                    );
                }
            }
        }
    }

    #[test]
    fn quantize_u8_row_avx2_matches_scalar_and_round_ps() {
        // Include leftover length and a zero row.
        let lens = [1, 7, 8, 9, 17, 128, 1280];
        for &k in &lens {
            let mut x = vec![0.0f32; k];
            for i in 0..k {
                x[i] = ((i * 11) % 23) as f32 * 0.13 - 0.8;
            }
            if k > 3 {
                x[3] = 0.0;
            }
            let mut a = vec![0u8; k];
            let mut b = vec![0u8; k];
            let sa = quantize_u8_row_scalar(&x, &mut a);
            let sb = quantize_u8_row(&x, &mut b);
            assert_eq!(sa.to_bits(), sb.to_bits(), "scale mismatch k={k}");
            assert_eq!(a, b, "u8 mismatch k={k}");
        }
        let x = vec![0.0f32; 16];
        let mut a = vec![0u8; 16];
        let s = quantize_u8_row(&x, &mut a);
        assert_eq!(s, 1.0);
        assert!(a.iter().all(|&v| v == 128), "zero row {a:?}");
    }

    #[test]
    fn int8_gemm_close_to_fp32() {
        let max = max_err(7, 20, 13);
        assert!(max < 0.15, "max abs err {max}");
    }

    #[test]
    fn vnni_blocked_close_to_fp32() {
        // Hits the 8x32 kernel plus m/n tails (m=19, n=48).
        let max = max_err(19, 48, 64);
        assert!(max < 0.25, "max abs err {max}");
    }

    #[test]
    fn vnni_matches_scalar_on_padded_shape() {
        let m = 5;
        let n = 32;
        let k = 64;
        let mut w = vec![0.0f32; n * k];
        let mut x = vec![0.0f32; m * k];
        for i in 0..w.len() {
            w[i] = ((i * 13) % 25) as f32 * 0.05 - 0.6;
        }
        for i in 0..x.len() {
            x[i] = ((i * 7) % 19) as f32 * 0.04 - 0.3;
        }
        let qm = Int8Mat::quantize(&w, n, k);
        let got = linear_int8(&x, &qm, None, m);
        assert_eq!(got.len(), m * n);
        assert!(got.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore]
    fn bench_encoder_like_linear() {
        use crate::cpu_tensor::{linear, CpuWeight};
        let m = 1500;
        let n = 1280;
        let k = 1280;
        let mut w = vec![0.0f32; n * k];
        let mut x = vec![0.0f32; m * k];
        for i in 0..w.len() {
            w[i] = ((i as i32 % 17) - 8) as f32 * 0.01;
        }
        for i in 0..x.len() {
            x[i] = ((i as i32 % 11) - 5) as f32 * 0.01;
        }
        let qm = Int8Mat::quantize(&w, n, k);
        let cw = CpuWeight::new(w, n, k);
        // warmup
        let _ = linear(&x, &cw, m);
        let _ = linear_int8(&x, &qm, None, m);
        let iters = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = linear(&x, &cw, m);
        }
        let fp32 = t0.elapsed().as_secs_f64() / iters as f64;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = linear_int8(&x, &qm, None, m);
        }
        let int8 = t0.elapsed().as_secs_f64() / iters as f64;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        eprintln!(
            "1500x1280x1280  fp32 {:.3}s ({:.1} GFLOP/s)  int8 {:.3}s ({:.1} GFLOP/s)  speedup {:.2}x",
            fp32,
            flops / fp32 / 1e9,
            int8,
            flops / int8 / 1e9,
            fp32 / int8
        );
        // Also time fc1-like 1500 x 5120 x 1280
        let n2 = 5120;
        let mut w2 = vec![0.0f32; n2 * k];
        for i in 0..w2.len() {
            w2[i] = ((i as i32 % 17) - 8) as f32 * 0.01;
        }
        let qm2 = Int8Mat::quantize(&w2, n2, k);
        let cw2 = CpuWeight::new(w2, n2, k);
        let _ = linear(&x, &cw2, m);
        let _ = linear_int8(&x, &qm2, None, m);
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = linear(&x, &cw2, m);
        }
        let fp32 = t0.elapsed().as_secs_f64() / iters as f64;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = linear_int8(&x, &qm2, None, m);
        }
        let int8 = t0.elapsed().as_secs_f64() / iters as f64;
        let flops = 2.0 * m as f64 * n2 as f64 * k as f64;
        eprintln!(
            "1500x5120x1280  fp32 {:.3}s ({:.1} GFLOP/s)  int8 {:.3}s ({:.1} GFLOP/s)  speedup {:.2}x",
            fp32,
            flops / fp32 / 1e9,
            int8,
            flops / int8 / 1e9,
            fp32 / int8
        );
    }
}
