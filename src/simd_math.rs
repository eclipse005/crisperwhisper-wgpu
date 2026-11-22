//! AVX-512 math kernels for encoder hot paths: `expf`, `erff`, softmax, GELU.
//!
//! The default path uses glibc libmvec (`_ZGVeN16v_expf` / `_ZGVeN16v_erff`) via
//! a C wrapper, so we stay close to scalar libm without calling `expf`/`erff`
//! one lane at a time. A custom polynomial `expf`/`erff` is kept as a faster
//! fallback (`expf_avx512` / `erff_avx512`) but is not the default: it moved
//! two audio3 word timestamps past the 20 ms soft threshold.
//!
//! Any change here must be re-checked against official gold (JFK, audio1 30s,
//! and for math changes also audio1 full + audio3).

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn cw_libmvec_expf16(input: *const f32, output: *mut f32);
    fn cw_libmvec_erff16(input: *const f32, output: *mut f32);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn libmvec_expf(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let mut tmp = [0.0f32; 16];
    _mm512_storeu_ps(tmp.as_mut_ptr(), x);
    cw_libmvec_expf16(tmp.as_ptr(), tmp.as_mut_ptr());
    _mm512_loadu_ps(tmp.as_ptr())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
unsafe fn libmvec_erff(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let mut tmp = [0.0f32; 16];
    _mm512_storeu_ps(tmp.as_mut_ptr(), x);
    cw_libmvec_erff16(tmp.as_ptr(), tmp.as_mut_ptr());
    _mm512_loadu_ps(tmp.as_ptr())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
pub unsafe fn expf_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    // Clamp to the finite range of expf. Softmax causal masks are -inf and
    // become ~0 after the lower clamp — close enough to exp(-inf) = 0.
    let x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.33655)), _mm512_set1_ps(88.37626));
    // n = round(x / ln2), r = x - n * ln2  (ln2 split for accuracy)
    const LOG2EF: f32 = 1.4426950408889634;
    const LN2_HI: f32 = 0.693359375; // 0x1.62e4p-1
    const LN2_LO: f32 = -2.121_944_40e-4;
    let n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(LOG2EF)), 0);
    // r = (x - n*LN2_HI) - n*LN2_LO
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_HI), x);
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_LO), r);
    // minimax for exp(r) on [-ln2/2, ln2/2], cephes-style
    const C0: f32 = 1.0;
    const C1: f32 = 1.0;
    const C2: f32 = 0.5;
    const C3: f32 = 1.66666666666666019037e-1;
    const C4: f32 = 4.16666666662260795726e-2;
    const C5: f32 = 8.33333333329956620655e-3;
    const C6: f32 = 1.38888885403571713845e-3;
    const C7: f32 = 1.98412697444574797948e-4;
    let mut p = _mm512_set1_ps(C7);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C6));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C5));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C4));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C2));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C1));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C0));
    let ni = _mm512_cvtps_epi32(n);
    let two_n = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_add_epi32(ni, _mm512_set1_epi32(127)),
        23,
    ));
    _mm512_mul_ps(p, two_n)
}

/// Approximate `erff` (Abramowitz & Stegun 7.1.26 is ~1.5e-7; this is a
/// higher-degree odd polynomial on a reduced argument, plus a tail).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
pub unsafe fn erff_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    // Algorithm: erf(x) = sign * (1 - t * exp(-x^2) * poly(t)), t = 1/(1+p|x|)
    // Coefficients from A&S 7.1.26, plus a small extra term for tighter error.
    let sign = _mm512_and_ps(x, _mm512_set1_ps(-0.0));
    let ax = _mm512_andnot_ps(_mm512_set1_ps(-0.0), x);
    let t = _mm512_rcp14_ps(_mm512_fmadd_ps(ax, _mm512_set1_ps(0.3275911), _mm512_set1_ps(1.0)));
    // one Newton rcp refinement: t = t * (2 - a*t)
    let den = _mm512_fmadd_ps(ax, _mm512_set1_ps(0.3275911), _mm512_set1_ps(1.0));
    let t = _mm512_mul_ps(t, _mm512_fnmadd_ps(den, t, _mm512_set1_ps(2.0)));
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    let mut poly = _mm512_set1_ps(A5);
    poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(A4));
    poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(A3));
    poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(A2));
    poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(A1));
    poly = _mm512_mul_ps(poly, t);
    let e = expf_avx512(_mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), ax), ax));
    let y = _mm512_fnmadd_ps(poly, e, _mm512_set1_ps(1.0));
    _mm512_or_ps(sign, y)
}

pub fn softmax_inplace_rows(mat: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(mat.len(), rows * cols);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && cols >= 16 {
            unsafe { softmax_avx512(mat, cols) };
            return;
        }
    }
    mat.par_chunks_mut(cols).for_each(softmax_row_scalar);
}

fn softmax_row_scalar(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_avx512(mat: &mut [f32], cols: usize) {
    use std::arch::x86_64::*;
    let cols16 = cols & !15;
    let rem = cols - cols16;
    let mask = if rem == 0 { 0 } else { (1u16 << rem) - 1 };
    mat.par_chunks_mut(cols).for_each(|row| {
        let ptr = row.as_mut_ptr();
        let mut maxv = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut p = 0;
        while p < cols16 {
            maxv = _mm512_max_ps(maxv, _mm512_loadu_ps(ptr.add(p)));
            p += 16;
        }
        let mut max = _mm512_reduce_max_ps(maxv);
        if rem != 0 {
            let tail = _mm512_maskz_loadu_ps(mask, ptr.add(cols16));
            // masked lanes are 0, which may exceed a fully-negative max — use a
            // masked max against -inf instead.
            let tail = _mm512_mask_blend_ps(mask, _mm512_set1_ps(f32::NEG_INFINITY), tail);
            max = max.max(_mm512_reduce_max_ps(tail));
        }
        let maxv = _mm512_set1_ps(max);
        let mut sumv = _mm512_setzero_ps();
        p = 0;
        while p < cols16 {
            // Subtract max in registers, then libmvec in place on the row
            // (avoids an extra 16-float bounce buffer).
            _mm512_storeu_ps(ptr.add(p), _mm512_sub_ps(_mm512_loadu_ps(ptr.add(p)), maxv));
            cw_libmvec_expf16(ptr.add(p), ptr.add(p));
            sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(ptr.add(p)));
            p += 16;
        }
        let mut sum = _mm512_reduce_add_ps(sumv);
        if rem != 0 {
            let e = libmvec_expf(_mm512_sub_ps(
                _mm512_mask_blend_ps(mask, maxv, _mm512_maskz_loadu_ps(mask, ptr.add(cols16))),
                maxv,
            ));
            _mm512_mask_storeu_ps(ptr.add(cols16), mask, e);
            sum += _mm512_mask_reduce_add_ps(mask, e);
        }
        let invv = _mm512_set1_ps(1.0 / sum);
        p = 0;
        while p < cols16 {
            _mm512_storeu_ps(ptr.add(p), _mm512_mul_ps(_mm512_loadu_ps(ptr.add(p)), invv));
            p += 16;
        }
        if rem != 0 {
            let v = _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, ptr.add(cols16)), invv);
            _mm512_mask_storeu_ps(ptr.add(cols16), mask, v);
        }
    });
}

pub fn gelu_erf_inplace(xs: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && xs.len() >= 16 {
            unsafe { gelu_avx512(xs) };
            return;
        }
    }
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    xs.par_iter_mut().for_each(|x| {
        *x = 0.5 * *x * (1.0 + libm::erff(*x * INV_SQRT2));
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gelu_avx512(xs: &mut [f32]) {
    use std::arch::x86_64::*;
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let inv_sqrt2 = _mm512_set1_ps(INV_SQRT2);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    let n = xs.len();
    let n16 = n & !15;
    // Work in reasonably large chunks so rayon pays off without splitting a
    // cache line mid-vector (chunks are 16-aligned).
    let chunk = if n16 >= 8192 { 4096 } else { n16.max(16) };
    let (head, tail) = xs.split_at_mut(n16);
    head.par_chunks_mut(chunk).for_each(|slab| {
        let mut p = 0;
        let ptr = slab.as_mut_ptr();
        let e = slab.len();
        // Scratch for libmvec erff so we can keep `x` in registers for the
        // 0.5 * x * (1 + erf) multiply (in-place erff would overwrite x).
        let mut erf_buf = [0.0f32; 16];
        while p < e {
            let x = _mm512_loadu_ps(ptr.add(p));
            _mm512_storeu_ps(erf_buf.as_mut_ptr(), _mm512_mul_ps(x, inv_sqrt2));
            cw_libmvec_erff16(erf_buf.as_ptr(), erf_buf.as_mut_ptr());
            let y = _mm512_loadu_ps(erf_buf.as_ptr());
            let out = _mm512_mul_ps(_mm512_mul_ps(half, x), _mm512_add_ps(one, y));
            _mm512_storeu_ps(ptr.add(p), out);
            p += 16;
        }
    });
    const INV: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for x in tail {
        *x = 0.5 * *x * (1.0 + libm::erff(*x * INV));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_close_to_scalar() {
        let rows = 7;
        let cols = 1500;
        let mut a = vec![0.0f32; rows * cols];
        let mut b = vec![0.0f32; rows * cols];
        for i in 0..a.len() {
            a[i] = ((i as i32 * 13) % 101) as f32 * 0.05 - 2.0;
            if i % 17 == 0 {
                a[i] = f32::NEG_INFINITY;
            }
        }
        b.copy_from_slice(&a);
        softmax_inplace_rows(&mut a, rows, cols);
        b.chunks_mut(cols).for_each(softmax_row_scalar);
        let mut max = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max = max.max((x - y).abs());
        }
        eprintln!("softmax libmvec vs scalar max abs err {max:.3e}");
        assert!(max < 2e-6, "softmax max abs err {max}");
    }

    #[test]
    fn gelu_close_to_libm() {
        let mut xs: Vec<f32> = (-80..80).map(|i| i as f32 * 0.07).collect();
        xs.extend_from_slice(&[0.0, 1.0, -1.0, 3.5, -3.5, 0.0001]);
        // pad to 16
        while xs.len() % 16 != 0 {
            xs.push(0.25);
        }
        let gold: Vec<f32> = xs
            .iter()
            .map(|&x| 0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2)))
            .collect();
        let mut got = xs;
        gelu_erf_inplace(&mut got);
        let mut max = 0.0f32;
        for (x, y) in got.iter().zip(gold.iter()) {
            max = max.max((x - y).abs());
        }
        // A&S erf is about 1.5e-7; GELU magnifies a little on large |x|.
        eprintln!("gelu libmvec vs libm max abs err {max:.3e}");
        assert!(max < 5e-6, "gelu max abs err {max}");
    }
}
