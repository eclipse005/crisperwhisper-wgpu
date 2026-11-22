//! CT2 CPU math kernels used only on the optional `--compute-type int8` path.
//!
//! On this machine CTranslate2 4.7.1 reports AVX512 as available but
//! **selects ISA AVX2** (`Selected ISA: AVX2`). GELU / softmax therefore
//! use 8-wide `exp256_ps` from `third_party/avx_mathfun.h` (cephes, mul+add
//! not FMA) and `vec_erf` from `src/cpu/vec.h`. Softmax leftover lanes and
//! whole rows of length `<= 8` use scalar `std::exp`. Horizontal reductions
//! follow CT2's `reduce_m256` (permute2f128 / shuffle 0x4E / 0xB1).
//!
//! The default fp32 path keeps glibc libmvec and must not call these
//! functions.

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn reduce_m256_add(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let v1 = _mm256_permute2f128_ps(v, v, 0x1);
    let v = _mm256_add_ps(v, v1);
    let v1 = _mm256_shuffle_ps(v, v, 0x4E);
    let v = _mm256_add_ps(v, v1);
    let v1 = _mm256_shuffle_ps(v, v, 0xB1);
    let v = _mm256_add_ps(v, v1);
    _mm256_cvtss_f32(v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn reduce_m256_max(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let v1 = _mm256_permute2f128_ps(v, v, 0x1);
    let v = _mm256_max_ps(v, v1);
    let v1 = _mm256_shuffle_ps(v, v, 0x4E);
    let v = _mm256_max_ps(v, v1);
    let v1 = _mm256_shuffle_ps(v, v, 0xB1);
    let v = _mm256_max_ps(v, v1);
    _mm256_cvtss_f32(v)
}

/// CT2 `exp256_ps` (avx_mathfun.h). Uses mul/add, not FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
pub unsafe fn exp256_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949));
    let x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949));
    let fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341));
    let fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5));
    let tmp = _mm256_round_ps(fx, 0x9); // floor, no exc
    let mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
    let fx = _mm256_sub_ps(tmp, mask);
    let tmp = _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375));
    let z = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4));
    let x = _mm256_sub_ps(x, tmp);
    let x = _mm256_sub_ps(x, z);
    let z = _mm256_mul_ps(x, x);
    let mut y = _mm256_set1_ps(1.9875691500E-4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.3981999507E-3));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(8.3334519073E-3));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(4.1665795894E-2));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.6666665459E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(5.0000001201E-1));
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);
    let imm0 = _mm256_cvttps_epi32(fx);
    let imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    let imm0 = _mm256_slli_epi32(imm0, 23);
    _mm256_mul_ps(y, _mm256_castsi256_ps(imm0))
}

/// CT2 `vec_erf` (A&S 7.1.26, exact div, `exp256_ps(-x*x)`, FMA in the poly).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn vec_erf(a: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let neg_zero = _mm256_set1_ps(-0.0);
    let one = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_and_ps(neg_zero, a);
    let abs_vec = _mm256_xor_ps(sign_mask, a);
    let tmp0 = _mm256_fmadd_ps(_mm256_set1_ps(0.3275911), abs_vec, one);
    let t = _mm256_div_ps(one, tmp0);
    let mut r = _mm256_fmadd_ps(_mm256_set1_ps(1.061405429), t, _mm256_set1_ps(-1.453152027));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(1.421413741));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(-0.284496736));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(0.254829592));
    let pow_2 = _mm256_mul_ps(a, a);
    let neg_pow_2 = _mm256_xor_ps(neg_zero, pow_2);
    let tmp4 = exp256_ps(neg_pow_2);
    let tmp5 = _mm256_xor_ps(neg_zero, tmp4);
    let tmp6 = _mm256_mul_ps(tmp5, t);
    let tmp7 = _mm256_fmadd_ps(tmp6, r, one);
    _mm256_xor_ps(sign_mask, tmp7)
}

/// CT2 `gelu_func`: `0.5 * (v * (1 + erf(v * 0.7071067811865475)))`.
pub fn gelu_erf_ct2_inplace(xs: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && !xs.is_empty() {
            unsafe { gelu_ct2_avx2(xs) };
            return;
        }
    }
    xs.par_iter_mut().for_each(|x| *x = gelu_ct2_scalar(*x));
}

fn gelu_ct2_scalar(v: f32) -> f32 {
    const K: f32 = 0.7071067811865475;
    0.5 * (v * (1.0 + erf_as_scalar(v * K)))
}

fn erf_as_scalar(a: f32) -> f32 {
    let sign = if a.is_sign_negative() { -1.0 } else { 1.0 };
    let abs = a.abs();
    let t = 1.0 / (0.3275911 * abs + 1.0);
    let r = (((1.061405429 * t + -1.453152027) * t + 1.421413741) * t + -0.284496736) * t
        + 0.254829592;
    sign * (1.0 - r * t * (-(a * a)).exp())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn gelu_ct2_avx2(xs: &mut [f32]) {
    use std::arch::x86_64::*;
    const K: f32 = 0.7071067811865475;
    let k = _mm256_set1_ps(K);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let n = xs.len();
    let n8 = n & !7;
    let chunk = if n8 >= 8192 { 4096 } else { n8.max(8) };
    if n8 > 0 {
        let (head, _) = xs.split_at_mut(n8);
        head.par_chunks_mut(chunk).for_each(|slab| {
            let ptr = slab.as_mut_ptr();
            let mut p = 0;
            while p < slab.len() {
                let v = _mm256_loadu_ps(ptr.add(p));
                // CT2: u = k*v; u = 1+erf(u); u = v*u; u = 0.5*u
                let u = vec_erf(_mm256_mul_ps(v, k));
                let u = _mm256_add_ps(one, u);
                let u = _mm256_mul_ps(v, u);
                let y = _mm256_mul_ps(half, u);
                _mm256_storeu_ps(ptr.add(p), y);
                p += 8;
            }
        });
    }
    let rem = n - n8;
    if rem != 0 {
        let mut tmp = [0.0f32; 8];
        tmp[..rem].copy_from_slice(&xs[n8..]);
        let v = _mm256_loadu_ps(tmp.as_ptr());
        let u = vec_erf(_mm256_mul_ps(v, k));
        let u = _mm256_add_ps(one, u);
        let u = _mm256_mul_ps(v, u);
        let y = _mm256_mul_ps(half, u);
        _mm256_storeu_ps(tmp.as_mut_ptr(), y);
        xs[n8..].copy_from_slice(&tmp[..rem]);
    }
}

/// CT2 AVX2 softmax. Rows `<= 8` and leftover lanes use scalar `std::exp`.
pub fn softmax_ct2_inplace_rows(mat: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(mat.len(), rows * cols);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && cols > 8 {
            unsafe { softmax_ct2_avx2(mat, cols) };
            return;
        }
    }
    mat.par_chunks_mut(cols).for_each(softmax_row_scalar_libm);
}

fn softmax_row_scalar_libm(row: &mut [f32]) {
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
#[target_feature(enable = "avx,avx2")]
unsafe fn softmax_ct2_avx2(mat: &mut [f32], cols: usize) {
    use std::arch::x86_64::*;
    let cols8 = cols & !7;
    let rem = cols - cols8;
    mat.par_chunks_mut(cols).for_each(|row| {
        let ptr = row.as_mut_ptr();
        let mut maxv = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut p = 0;
        while p < cols8 {
            maxv = _mm256_max_ps(maxv, _mm256_loadu_ps(ptr.add(p)));
            p += 8;
        }
        let mut max = reduce_m256_max(maxv);
        if rem != 0 {
            for i in cols8..cols {
                if row[i] > max {
                    max = row[i];
                }
            }
        }
        let maxv = _mm256_set1_ps(max);
        let mut sumv = _mm256_setzero_ps();
        p = 0;
        while p < cols8 {
            let e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(ptr.add(p)), maxv));
            _mm256_storeu_ps(ptr.add(p), e);
            sumv = _mm256_add_ps(sumv, e);
            p += 8;
        }
        let mut sum = reduce_m256_add(sumv);
        if rem != 0 {
            for i in cols8..cols {
                let e = (row[i] - max).exp();
                row[i] = e;
                sum += e;
            }
        }
        let invv = _mm256_set1_ps(1.0 / sum);
        p = 0;
        while p < cols8 {
            _mm256_storeu_ps(ptr.add(p), _mm256_mul_ps(_mm256_loadu_ps(ptr.add(p)), invv));
            p += 8;
        }
        if rem != 0 {
            let inv = 1.0 / sum;
            for i in cols8..cols {
                row[i] *= inv;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp256_ps_matches_ct2_c_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: &[(f32, u32)] = &[
            (0.0, 0x3f800000),
            (1.0, 0x402df854),
            (-1.0, 0x3ebc5ab2),
            (0.5, 0x3fd3094c),
            (-0.5, 0x3f1b4598),
            (2.0, 0x40ec7326),
            (-2.0, 0x3e0a9555),
            (10.0, 0x46ac14ee),
            (-10.0, 0x383e6bce),
            (20.0, 0x4de75844),
            (-20.0, 0x310da433),
            (88.0, 0x7ef882b7),
            (-88.0, 0x00000000),
            (100.0, 0x7f3504a4),
            (-100.0, 0x00000000),
            (0.7071067811865475, 0x4001cca3),
            (-0.7071067811865475, 0x3efc7380),
            (std::f32::consts::PI, 0x41b92024),
            (-std::f32::consts::PI, 0x3d310113),
            (1e-8, 0x3f800000),
            (-1e-8, 0x3f800000),
            (16.0, 0x4b07975f),
            (-16.0, 0x33f1aade),
            (0.1, 0x3f8d763e),
            (-0.3, 0x3f3da643),
            (1.7, 0x40af2a94),
            (-4.2, 0x3c75b005),
            (8.5, 0x45999627),
        ];
        unsafe {
            for &(x, want) in cases {
                let mut tmp = [0.0f32; 8];
                tmp.fill(x);
                let got = exp256_ps(std::arch::x86_64::_mm256_loadu_ps(tmp.as_ptr()));
                std::arch::x86_64::_mm256_storeu_ps(tmp.as_mut_ptr(), got);
                assert_eq!(tmp[0].to_bits(), want, "exp256_ps({x}) got {:#x} want {want:#x}", tmp[0].to_bits());
            }
            let mut mix = [0.0f32; 8];
            for i in 0..8 {
                mix[i] = (i as f32 - 4.0) * 0.37;
            }
            let want_mix: [u32; 8] = [
                0x3e6919da, 0x3ea8bbf4, 0x3ef44846, 0x3f30d3f7, 0x3f800000, 0x3fb94f5e,
                0x400623cf, 0x404232ee,
            ];
            let got = exp256_ps(std::arch::x86_64::_mm256_loadu_ps(mix.as_ptr()));
            std::arch::x86_64::_mm256_storeu_ps(mix.as_mut_ptr(), got);
            for i in 0..8 {
                assert_eq!(mix[i].to_bits(), want_mix[i], "mix[{i}] got {:#x}", mix[i].to_bits());
            }
        }
    }

    #[test]
    fn gelu_ct2_leftover_matches_padded_simd() {
        let mut xs: Vec<f32> = (-40..41).map(|i| i as f32 * 0.17).collect(); // 81
        assert_eq!(xs.len() % 8, 1);
        let mut padded = xs.clone();
        padded.resize(xs.len().div_ceil(8) * 8, 0.0);
        gelu_erf_ct2_inplace(&mut xs);
        gelu_erf_ct2_inplace(&mut padded);
        assert!(xs.iter().all(|v| v.is_finite()));
        for (i, (&a, &b)) in xs.iter().zip(padded.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "lane {i}");
        }
    }
}
