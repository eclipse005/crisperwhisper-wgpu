//! Specialized fp32 attention GEMMs for `head_dim = 64` (all official models).
//!
//! These beat a general packed GEMM on the skinny-K 1500×1500×64 shapes used
//! by the encoder: we pack a 16-row K/V panel once and stream Q/scores with
//! AVX-512 FMA, single-threaded per head (the caller already parallelizes
//! across heads).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const HEAD_DIM: usize = 64;

/// `C[m, n] = A[m, 64] @ B[n, 64]^T`. A/B have row stride `rs` (typically `d_model`).
/// C is row-major with row stride `n`.
pub unsafe fn gemm_bt_k64(
    m: usize,
    n: usize,
    c: *mut f32,
    a: *const f32,
    a_rs: usize,
    b: *const f32,
    b_rs: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            gemm_bt_k64_avx512(m, n, c, a, a_rs, b, b_rs);
            return;
        }
    }
    gemm_bt_k64_scalar(m, n, c, a, a_rs, b, b_rs);
}

/// `C[m, 64] = scores[m, n] @ V[n, 64]`. Scores are `[m, n]` row-major,
/// V has row stride `v_rs`, C has row stride `c_rs`.
pub unsafe fn gemm_av_k64(
    m: usize,
    n: usize,
    c: *mut f32,
    c_rs: usize,
    scores: *const f32,
    v: *const f32,
    v_rs: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            gemm_av_k64_avx512(m, n, c, c_rs, scores, v, v_rs);
            return;
        }
    }
    gemm_av_k64_scalar(m, n, c, c_rs, scores, v, v_rs);
}

unsafe fn gemm_bt_k64_scalar(
    m: usize,
    n: usize,
    c: *mut f32,
    a: *const f32,
    a_rs: usize,
    b: *const f32,
    b_rs: usize,
) {
    for i in 0..m {
        let ai = a.add(i * a_rs);
        for j in 0..n {
            let bj = b.add(j * b_rs);
            let mut acc = 0.0f32;
            for p in 0..HEAD_DIM {
                acc += *ai.add(p) * *bj.add(p);
            }
            *c.add(i * n + j) = acc;
        }
    }
}

unsafe fn gemm_av_k64_scalar(
    m: usize,
    n: usize,
    c: *mut f32,
    c_rs: usize,
    scores: *const f32,
    v: *const f32,
    v_rs: usize,
) {
    for i in 0..m {
        for p in 0..HEAD_DIM {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += *scores.add(i * n + j) * *v.add(j * v_rs + p);
            }
            *c.add(i * c_rs + p) = acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_bt_k64_avx512(
    m: usize,
    n: usize,
    c: *mut f32,
    a: *const f32,
    a_rs: usize,
    b: *const f32,
    b_rs: usize,
) {
    const NR: usize = 16;
    const MR: usize = 8;
    let n16 = n & !15;
    let mut packed = [0.0f32; HEAD_DIM * NR];
    let mut j = 0;
    while j < n16 {
        // packed[p][t] = B[j+t, p], so each p is 16 contiguous B values.
        for t in 0..NR {
            let src = b.add((j + t) * b_rs);
            for p in 0..HEAD_DIM {
                packed[p * NR + t] = *src.add(p);
            }
        }
        let mut i = 0;
        while i + MR <= m {
            let mut acc = [_mm512_setzero_ps(); MR];
            for p in 0..HEAD_DIM {
                let bv = _mm512_loadu_ps(packed.as_ptr().add(p * NR));
                for r in 0..MR {
                    let av = _mm512_set1_ps(*a.add((i + r) * a_rs + p));
                    acc[r] = _mm512_fmadd_ps(av, bv, acc[r]);
                }
            }
            for r in 0..MR {
                _mm512_storeu_ps(c.add((i + r) * n + j), acc[r]);
            }
            i += MR;
        }
        while i < m {
            let mut acc = _mm512_setzero_ps();
            let ai = a.add(i * a_rs);
            for p in 0..HEAD_DIM {
                let bv = _mm512_loadu_ps(packed.as_ptr().add(p * NR));
                acc = _mm512_fmadd_ps(_mm512_set1_ps(*ai.add(p)), bv, acc);
            }
            _mm512_storeu_ps(c.add(i * n + j), acc);
            i += 1;
        }
        j += NR;
    }
    if j < n {
        let nr = n - j;
        packed[..HEAD_DIM * NR].fill(0.0);
        for t in 0..nr {
            let src = b.add((j + t) * b_rs);
            for p in 0..HEAD_DIM {
                packed[p * NR + t] = *src.add(p);
            }
        }
        let mask = (1u16 << nr) - 1;
        for i in 0..m {
            let mut acc = _mm512_setzero_ps();
            let ai = a.add(i * a_rs);
            for p in 0..HEAD_DIM {
                let bv = _mm512_loadu_ps(packed.as_ptr().add(p * NR));
                acc = _mm512_fmadd_ps(_mm512_set1_ps(*ai.add(p)), bv, acc);
            }
            _mm512_mask_storeu_ps(c.add(i * n + j), mask, acc);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_av_k64_avx512(
    m: usize,
    n: usize,
    c: *mut f32,
    c_rs: usize,
    scores: *const f32,
    v: *const f32,
    v_rs: usize,
) {
    const NR: usize = 16;
    const MR: usize = 8;
    // Pre-pack V to [n, 64] so inner loads are contiguous.
    let mut vpack = vec![0.0f32; n * HEAD_DIM];
    for t in 0..n {
        let src = v.add(t * v_rs);
        let dst = vpack.as_mut_ptr().add(t * HEAD_DIM);
        // 64 floats = 4 zmm
        for p in 0..4 {
            _mm512_storeu_ps(dst.add(p * 16), _mm512_loadu_ps(src.add(p * 16)));
        }
    }
    let n16 = n & !15;
    let mut i = 0;
    while i + MR <= m {
        let mut acc = [[_mm512_setzero_ps(); 4]; MR];
        let mut j = 0;
        while j < n16 {
            for t in 0..NR {
                let vp = vpack.as_ptr().add((j + t) * HEAD_DIM);
                let v0 = _mm512_loadu_ps(vp);
                let v1 = _mm512_loadu_ps(vp.add(16));
                let v2 = _mm512_loadu_ps(vp.add(32));
                let v3 = _mm512_loadu_ps(vp.add(48));
                for r in 0..MR {
                    let s = _mm512_set1_ps(*scores.add((i + r) * n + j + t));
                    acc[r][0] = _mm512_fmadd_ps(s, v0, acc[r][0]);
                    acc[r][1] = _mm512_fmadd_ps(s, v1, acc[r][1]);
                    acc[r][2] = _mm512_fmadd_ps(s, v2, acc[r][2]);
                    acc[r][3] = _mm512_fmadd_ps(s, v3, acc[r][3]);
                }
            }
            j += NR;
        }
        while j < n {
            let vp = vpack.as_ptr().add(j * HEAD_DIM);
            let v0 = _mm512_loadu_ps(vp);
            let v1 = _mm512_loadu_ps(vp.add(16));
            let v2 = _mm512_loadu_ps(vp.add(32));
            let v3 = _mm512_loadu_ps(vp.add(48));
            for r in 0..MR {
                let s = _mm512_set1_ps(*scores.add((i + r) * n + j));
                acc[r][0] = _mm512_fmadd_ps(s, v0, acc[r][0]);
                acc[r][1] = _mm512_fmadd_ps(s, v1, acc[r][1]);
                acc[r][2] = _mm512_fmadd_ps(s, v2, acc[r][2]);
                acc[r][3] = _mm512_fmadd_ps(s, v3, acc[r][3]);
            }
            j += 1;
        }
        for r in 0..MR {
            let dst = c.add((i + r) * c_rs);
            _mm512_storeu_ps(dst, acc[r][0]);
            _mm512_storeu_ps(dst.add(16), acc[r][1]);
            _mm512_storeu_ps(dst.add(32), acc[r][2]);
            _mm512_storeu_ps(dst.add(48), acc[r][3]);
        }
        i += MR;
    }
    while i < m {
        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        let mut a2 = _mm512_setzero_ps();
        let mut a3 = _mm512_setzero_ps();
        for j in 0..n {
            let s = _mm512_set1_ps(*scores.add(i * n + j));
            let vp = vpack.as_ptr().add(j * HEAD_DIM);
            a0 = _mm512_fmadd_ps(s, _mm512_loadu_ps(vp), a0);
            a1 = _mm512_fmadd_ps(s, _mm512_loadu_ps(vp.add(16)), a1);
            a2 = _mm512_fmadd_ps(s, _mm512_loadu_ps(vp.add(32)), a2);
            a3 = _mm512_fmadd_ps(s, _mm512_loadu_ps(vp.add(48)), a3);
        }
        let dst = c.add(i * c_rs);
        _mm512_storeu_ps(dst, a0);
        _mm512_storeu_ps(dst.add(16), a1);
        _mm512_storeu_ps(dst.add(32), a2);
        _mm512_storeu_ps(dst.add(48), a3);
        i += 1;
    }
}


/// `scores[h, j] = Q[h, 64] · K[j, h*64:(h+1)*64]` for a single query
/// (`tq == 1`). Walks each K row once across all heads so the decoder
/// does not stream the 1500×d cache 20 times through gemm 0.18 packing.
pub unsafe fn scores_t1_k64(
    n_heads: usize,
    tk: usize,
    scores: *mut f32,
    q: *const f32,
    k: *const f32,
    d: usize,
) {
    debug_assert_eq!(d, n_heads * HEAD_DIM);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            scores_t1_k64_avx512(n_heads, tk, scores, q, k, d);
            return;
        }
    }
    scores_t1_k64_scalar(n_heads, tk, scores, q, k, d);
}

/// `ctx[h, 64] = sum_j scores[h, j] * V[j, h*64:(h+1)*64]` for `tq == 1`.
pub unsafe fn av_t1_k64(
    n_heads: usize,
    tk: usize,
    ctx: *mut f32,
    d: usize,
    scores: *const f32,
    v: *const f32,
) {
    debug_assert_eq!(d, n_heads * HEAD_DIM);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            av_t1_k64_avx512(n_heads, tk, ctx, d, scores, v);
            return;
        }
    }
    av_t1_k64_scalar(n_heads, tk, ctx, d, scores, v);
}

unsafe fn scores_t1_k64_scalar(
    n_heads: usize,
    tk: usize,
    scores: *mut f32,
    q: *const f32,
    k: *const f32,
    d: usize,
) {
    for j in 0..tk {
        let krow = k.add(j * d);
        for h in 0..n_heads {
            let qh = q.add(h * HEAD_DIM);
            let kh = krow.add(h * HEAD_DIM);
            let mut acc = 0.0f32;
            for p in 0..HEAD_DIM {
                acc += *qh.add(p) * *kh.add(p);
            }
            *scores.add(h * tk + j) = acc;
        }
    }
}

unsafe fn av_t1_k64_scalar(
    n_heads: usize,
    tk: usize,
    ctx: *mut f32,
    d: usize,
    scores: *const f32,
    v: *const f32,
) {
    for i in 0..d {
        *ctx.add(i) = 0.0;
    }
    for j in 0..tk {
        let vrow = v.add(j * d);
        for h in 0..n_heads {
            let s = *scores.add(h * tk + j);
            let dst = ctx.add(h * HEAD_DIM);
            let src = vrow.add(h * HEAD_DIM);
            for p in 0..HEAD_DIM {
                *dst.add(p) += s * *src.add(p);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scores_t1_k64_avx512(
    n_heads: usize,
    tk: usize,
    scores: *mut f32,
    q: *const f32,
    k: *const f32,
    d: usize,
) {
    use rayon::prelude::*;
    #[derive(Copy, Clone)]
    struct Ptr(*const f32);
    unsafe impl Send for Ptr {}
    unsafe impl Sync for Ptr {}
    impl Ptr {
        unsafe fn add(self, off: usize) -> *const f32 {
            self.0.add(off)
        }
    }
    #[derive(Copy, Clone)]
    struct MutPtr(*mut f32);
    unsafe impl Send for MutPtr {}
    unsafe impl Sync for MutPtr {}
    impl MutPtr {
        unsafe fn add(self, off: usize) -> *mut f32 {
            self.0.add(off)
        }
    }
    let qp = Ptr(q);
    let kp = Ptr(k);
    let sp = MutPtr(scores);
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = ((tk + nthreads - 1) / nthreads).max(1);
    (0..nthreads).into_par_iter().for_each(|ci| {
        let j0 = ci * chunk;
        if j0 >= tk {
            return;
        }
        let j1 = (j0 + chunk).min(tk);
        for j in j0..j1 {
            if j + 1 < j1 {
                _mm_prefetch(kp.add((j + 1) * d) as *const i8, _MM_HINT_T0);
            }
            let krow = kp.add(j * d);
            for h in 0..n_heads {
                let qh = qp.add(h * HEAD_DIM);
                let kh = krow.add(h * HEAD_DIM);
                let a0 = _mm512_fmadd_ps(_mm512_loadu_ps(qh), _mm512_loadu_ps(kh), _mm512_setzero_ps());
                let a1 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(qh.add(16)),
                    _mm512_loadu_ps(kh.add(16)),
                    _mm512_setzero_ps(),
                );
                let a2 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(qh.add(32)),
                    _mm512_loadu_ps(kh.add(32)),
                    _mm512_setzero_ps(),
                );
                let a3 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(qh.add(48)),
                    _mm512_loadu_ps(kh.add(48)),
                    _mm512_setzero_ps(),
                );
                *sp.add(h * tk + j) = _mm512_reduce_add_ps(a0)
                    + _mm512_reduce_add_ps(a1)
                    + _mm512_reduce_add_ps(a2)
                    + _mm512_reduce_add_ps(a3);
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn av_t1_k64_avx512(
    n_heads: usize,
    tk: usize,
    ctx: *mut f32,
    d: usize,
    scores: *const f32,
    v: *const f32,
) {
    use rayon::prelude::*;
    #[derive(Copy, Clone)]
    struct Ptr(*const f32);
    unsafe impl Send for Ptr {}
    unsafe impl Sync for Ptr {}
    impl Ptr {
        unsafe fn add(self, off: usize) -> *const f32 {
            self.0.add(off)
        }
    }
    let sp = Ptr(scores);
    let vp0 = Ptr(v);
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = ((tk + nthreads - 1) / nthreads).max(1);
    let acc_dim = n_heads * HEAD_DIM;
    let partials: Vec<Vec<f32>> = (0..nthreads)
        .into_par_iter()
        .map(|ci| {
            let j0 = ci * chunk;
            let mut acc = vec![0.0f32; acc_dim];
            if j0 >= tk {
                return acc;
            }
            let j1 = (j0 + chunk).min(tk);
            for j in j0..j1 {
                if j + 1 < j1 {
                    _mm_prefetch(vp0.add((j + 1) * d) as *const i8, _MM_HINT_T0);
                }
                let vrow = vp0.add(j * d);
                for h in 0..n_heads {
                    let s = _mm512_set1_ps(*sp.add(h * tk + j));
                    let ap = acc.as_mut_ptr().add(h * HEAD_DIM);
                    let vp = vrow.add(h * HEAD_DIM);
                    _mm512_storeu_ps(ap, _mm512_fmadd_ps(s, _mm512_loadu_ps(vp), _mm512_loadu_ps(ap)));
                    _mm512_storeu_ps(
                        ap.add(16),
                        _mm512_fmadd_ps(s, _mm512_loadu_ps(vp.add(16)), _mm512_loadu_ps(ap.add(16))),
                    );
                    _mm512_storeu_ps(
                        ap.add(32),
                        _mm512_fmadd_ps(s, _mm512_loadu_ps(vp.add(32)), _mm512_loadu_ps(ap.add(32))),
                    );
                    _mm512_storeu_ps(
                        ap.add(48),
                        _mm512_fmadd_ps(s, _mm512_loadu_ps(vp.add(48)), _mm512_loadu_ps(ap.add(48))),
                    );
                }
            }
            acc
        })
        .collect();
    for i in 0..d {
        *ctx.add(i) = 0.0;
    }
    for acc in &partials {
        for i in 0..d {
            *ctx.add(i) += acc[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bt_k64_matches_scalar() {
        let m = 11;
        let n = 37;
        let a_rs = 80;
        let b_rs = 80;
        let mut a = vec![0.0f32; m * a_rs];
        let mut b = vec![0.0f32; n * b_rs];
        for i in 0..a.len() {
            a[i] = ((i as i32 % 17) - 8) as f32 * 0.05;
        }
        for i in 0..b.len() {
            b[i] = ((i as i32 % 13) - 6) as f32 * 0.04;
        }
        let mut got = vec![0.0f32; m * n];
        let mut gold = vec![0.0f32; m * n];
        unsafe {
            gemm_bt_k64(m, n, got.as_mut_ptr(), a.as_ptr(), a_rs, b.as_ptr(), b_rs);
            gemm_bt_k64_scalar(m, n, gold.as_mut_ptr(), a.as_ptr(), a_rs, b.as_ptr(), b_rs);
        }
        let mut max = 0.0f32;
        for (x, y) in got.iter().zip(gold.iter()) {
            max = max.max((x - y).abs());
        }
        assert!(max < 2e-4, "qk max abs err {max}");
    }

    #[test]
    fn av_k64_matches_scalar() {
        let m = 11;
        let n = 37;
        let c_rs = 80;
        let v_rs = 80;
        let mut s = vec![0.0f32; m * n];
        let mut v = vec![0.0f32; n * v_rs];
        for i in 0..s.len() {
            s[i] = ((i as i32 % 11) - 5) as f32 * 0.02;
        }
        for i in 0..v.len() {
            v[i] = ((i as i32 % 19) - 9) as f32 * 0.03;
        }
        let mut got = vec![0.0f32; m * c_rs];
        let mut gold = vec![0.0f32; m * c_rs];
        unsafe {
            gemm_av_k64(m, n, got.as_mut_ptr(), c_rs, s.as_ptr(), v.as_ptr(), v_rs);
            gemm_av_k64_scalar(m, n, gold.as_mut_ptr(), c_rs, s.as_ptr(), v.as_ptr(), v_rs);
        }
        let mut max = 0.0f32;
        for r in 0..m {
            for p in 0..HEAD_DIM {
                max = max.max((got[r * c_rs + p] - gold[r * c_rs + p]).abs());
            }
        }
        assert!(max < 3e-4, "av max abs err {max}");
    }

    fn fill_pattern(buf: &mut [f32], modulus: i32, off: i32, scale: f32) {
        for i in 0..buf.len() {
            buf[i] = ((i as i32 % modulus) - off) as f32 * scale;
        }
    }

    #[test]
    fn t1_qk_av_match_scalar() {
        let n_heads = 20;
        let tk = 1500;
        let d = n_heads * HEAD_DIM;
        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; tk * d];
        let mut v = vec![0.0f32; tk * d];
        fill_pattern(&mut q, 17, 8, 0.05);
        fill_pattern(&mut k, 13, 6, 0.04);
        fill_pattern(&mut v, 19, 9, 0.03);
        let mut scores = vec![0.0f32; n_heads * tk];
        let mut scores_g = vec![0.0f32; n_heads * tk];
        let mut ctx = vec![0.0f32; d];
        let mut ctx_g = vec![0.0f32; d];
        unsafe {
            scores_t1_k64(n_heads, tk, scores.as_mut_ptr(), q.as_ptr(), k.as_ptr(), d);
            scores_t1_k64_scalar(n_heads, tk, scores_g.as_mut_ptr(), q.as_ptr(), k.as_ptr(), d);
            av_t1_k64(n_heads, tk, ctx.as_mut_ptr(), d, scores.as_ptr(), v.as_ptr());
            av_t1_k64_scalar(n_heads, tk, ctx_g.as_mut_ptr(), d, scores_g.as_ptr(), v.as_ptr());
        }
        let mut qk_max = 0.0f32;
        for (x, y) in scores.iter().zip(scores_g.iter()) {
            qk_max = qk_max.max((x - y).abs());
        }
        let mut av_max = 0.0f32;
        for (x, y) in ctx.iter().zip(ctx_g.iter()) {
            av_max = av_max.max((x - y).abs());
        }
        assert!(qk_max < 2e-4, "t1 qk vs scalar max abs err {qk_max}");
        assert!(av_max < 5e-4, "t1 av vs scalar max abs err {av_max}");
    }

    #[test]
    fn t1_close_to_gemm018() {
        use crate::cpu_tensor::{gemm_bt_strided, gemm_strided};
        use gemm::Parallelism;
        let n_heads = 20;
        let tk = 1500;
        let d = n_heads * HEAD_DIM;
        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; tk * d];
        let mut v = vec![0.0f32; tk * d];
        fill_pattern(&mut q, 17, 8, 0.05);
        fill_pattern(&mut k, 13, 6, 0.04);
        fill_pattern(&mut v, 19, 9, 0.03);
        let mut scores = vec![0.0f32; n_heads * tk];
        let mut ctx = vec![0.0f32; d];
        unsafe {
            scores_t1_k64(n_heads, tk, scores.as_mut_ptr(), q.as_ptr(), k.as_ptr(), d);
            av_t1_k64(n_heads, tk, ctx.as_mut_ptr(), d, scores.as_ptr(), v.as_ptr());
        }
        let mut qk_max = 0.0f32;
        let mut av_max = 0.0f32;
        for h in 0..n_heads {
            let mut gemm_s = vec![0.0f32; tk];
            let mut gemm_c = vec![0.0f32; d];
            unsafe {
                gemm_bt_strided(
                    1,
                    tk,
                    HEAD_DIM,
                    gemm_s.as_mut_ptr(),
                    1,
                    tk as isize,
                    q.as_ptr().add(h * HEAD_DIM),
                    1,
                    d as isize,
                    k.as_ptr().add(h * HEAD_DIM),
                    d as isize,
                    1,
                    Parallelism::None,
                );
                gemm_strided(
                    1,
                    HEAD_DIM,
                    tk,
                    gemm_c.as_mut_ptr().add(h * HEAD_DIM),
                    1,
                    d as isize,
                    gemm_s.as_ptr(),
                    1,
                    tk as isize,
                    v.as_ptr().add(h * HEAD_DIM),
                    1,
                    d as isize,
                    Parallelism::None,
                );
            }
            for j in 0..tk {
                qk_max = qk_max.max((scores[h * tk + j] - gemm_s[j]).abs());
            }
            for p in 0..HEAD_DIM {
                av_max = av_max.max((ctx[h * HEAD_DIM + p] - gemm_c[h * HEAD_DIM + p]).abs());
            }
        }
        eprintln!("t1 vs gemm 0.18: qk max {qk_max:.6e} av max {av_max:.6e}");
        // Association differs from gemm 0.18; this documents the gap. Product
        // default still requires a timestamp regression before shipping.
        assert!(qk_max < 5e-4, "t1 qk vs gemm max abs err {qk_max}");
        assert!(av_max < 2e-3, "t1 av vs gemm max abs err {av_max}");
    }

    #[test]
    #[ignore]
    fn bench_t1_vs_gemm() {
        use crate::cpu_tensor::{gemm_bt_strided, gemm_strided};
        use gemm::Parallelism;
        use std::time::Instant;
        let n_heads = 20;
        let tk = 1500;
        let d = n_heads * HEAD_DIM;
        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; tk * d];
        let mut v = vec![0.0f32; tk * d];
        fill_pattern(&mut q, 17, 8, 0.05);
        fill_pattern(&mut k, 13, 6, 0.04);
        fill_pattern(&mut v, 19, 9, 0.03);
        let mut scores = vec![0.0f32; n_heads * tk];
        let mut ctx = vec![0.0f32; d];
        let iters = 200;
        unsafe {
            scores_t1_k64(n_heads, tk, scores.as_mut_ptr(), q.as_ptr(), k.as_ptr(), d);
            av_t1_k64(n_heads, tk, ctx.as_mut_ptr(), d, scores.as_ptr(), v.as_ptr());
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                scores_t1_k64(n_heads, tk, scores.as_mut_ptr(), q.as_ptr(), k.as_ptr(), d);
                av_t1_k64(n_heads, tk, ctx.as_mut_ptr(), d, scores.as_ptr(), v.as_ptr());
            }
        }
        let t1 = t0.elapsed();
        let t2 = Instant::now();
        for _ in 0..iters {
            for h in 0..n_heads {
                unsafe {
                    gemm_bt_strided(
                        1,
                        tk,
                        HEAD_DIM,
                        scores.as_mut_ptr().add(h * tk),
                        1,
                        tk as isize,
                        q.as_ptr().add(h * HEAD_DIM),
                        1,
                        d as isize,
                        k.as_ptr().add(h * HEAD_DIM),
                        d as isize,
                        1,
                        Parallelism::None,
                    );
                    gemm_strided(
                        1,
                        HEAD_DIM,
                        tk,
                        ctx.as_mut_ptr().add(h * HEAD_DIM),
                        1,
                        d as isize,
                        scores.as_ptr().add(h * tk),
                        1,
                        tk as isize,
                        v.as_ptr().add(h * HEAD_DIM),
                        1,
                        d as isize,
                        Parallelism::None,
                    );
                }
            }
        }
        let t3 = t2.elapsed();
        eprintln!(
            "t1 fused-par {iters} iters: {:.3} ms  gemm 0.18 serial heads: {:.3} ms  speedup {:.2}x",
            t1.as_secs_f64() * 1e3,
            t3.as_secs_f64() * 1e3,
            t3.as_secs_f64() / t1.as_secs_f64()
        );
    }
}
