//! Scalar stand-ins for `cw_libmvec_expf16` / `cw_libmvec_erff16`.
//!
//! Compiled only when the glibc C wrapper (`src/libmvec_wrap.c`) is not built,
//! so Linux+glibc+x86_64 keeps a single definition. The existing
//! `extern "C"` declarations in `simd_math` still link. These symbols are only
//! called under `is_x86_feature_detected!("avx512f")`, so the scalar body does
//! not change gold on machines without AVX-512.

#[no_mangle]
pub unsafe extern "C" fn cw_libmvec_expf16(input: *const f32, output: *mut f32) {
    for i in 0..16 {
        *output.add(i) = libm::expf(*input.add(i));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cw_libmvec_erff16(input: *const f32, output: *mut f32) {
    for i in 0..16 {
        *output.add(i) = libm::erff(*input.add(i));
    }
}
