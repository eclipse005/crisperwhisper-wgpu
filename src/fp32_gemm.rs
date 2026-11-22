//! Pre-packed fp32 GEMM for encoder linears.
//!
//! `C[m, n] = A[m, k] @ W[n, k]^T` with W stored as 16-column panels
//! `[n_tiles, k, 16]`. Official turbo/small/medium/large have n,k multiples of
//! 16, so packing is size-preserving and we can drop the row-major copy.
//!
//! Accumulation splits K into the same panels as gemm 0.18's stable FMA path
//! (`kernel_params` after the row-major-C transpose, MR=16, NR=6) and *adds*
//! each panel into C. With that association the kernel is bit-identical to
//! `gemm_row_major` on encoder-sized shapes, so it can be the product default.

use rayon::prelude::*;

pub const NR: usize = 16;
const MR: usize = 8;
/// K-panel length used by gemm 0.18's stable FMA kernel after it transposes a
/// row-major `C[seq, out]`. Must stay in lockstep with `gemm_row_major`.
fn gemm_fma_kc(n_out: usize, seq: usize, k: usize) -> usize {
    const MR: usize = 16;
    const NR: usize = 6;
    if n_out <= 64 && seq <= 64 {
        return k.min(512);
    }
    gemm_common::cache::kernel_params(n_out, seq, k, MR, NR, std::mem::size_of::<f32>()).kc.max(1)
}

pub struct Fp32Packed {
    pub packed: Vec<f32>,
    pub n: usize,
    pub k: usize,
    pub n_tiles: usize,
}

impl Fp32Packed {
    pub fn pack(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        let n_tiles = (n + NR - 1) / NR;
        let mut packed = vec![0.0f32; n_tiles * k * NR];
        for t in 0..n_tiles {
            for p in 0..k {
                let dst = (t * k + p) * NR;
                for jr in 0..NR {
                    let r = t * NR + jr;
                    if r < n {
                        packed[dst + jr] = w[r * k + p];
                    }
                }
            }
        }
        Self {
            packed,
            n,
            k,
            n_tiles,
        }
    }

    pub fn can_pack(n: usize, k: usize) -> bool {
        n > 0 && k > 0 && n % NR == 0
    }
}

pub fn linear_packed(x: &[f32], w: &Fp32Packed, seq: usize) -> Vec<f32> {
    assert_eq!(x.len(), seq * w.k, "linear_packed input len");
    let mut out = vec![0.0f32; seq * w.n];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { gemm_packed_avx512(x, w, &mut out, seq, gemm_fma_kc(w.n, seq, w.k)) };
            return out;
        }
    }
    gemm_packed_scalar(x, w, &mut out, seq);
    out
}

fn gemm_packed_scalar(x: &[f32], w: &Fp32Packed, out: &mut [f32], _m: usize) {
    let n = w.n;
    let k = w.k;
    let n_tiles = w.n_tiles;
    let packed = &w.packed;
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a = &x[i * k..(i + 1) * k];
        for t in 0..n_tiles {
            let nr = (n - t * NR).min(NR);
            for jr in 0..nr {
                let mut acc = 0.0f32;
                let mut bp = (t * k) * NR + jr;
                for p in 0..k {
                    acc += a[p] * packed[bp];
                    bp += NR;
                }
                row[t * NR + jr] = acc;
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_packed_avx512(x: &[f32], w: &Fp32Packed, out: &mut [f32], m: usize, kc: usize) {
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
    #[derive(Copy, Clone)]
    struct MPtr(*mut f32);
    unsafe impl Send for MPtr {}
    unsafe impl Sync for MPtr {}
    impl MPtr {
        unsafe fn add(self, off: usize) -> *mut f32 {
            self.0.add(off)
        }
    }

    let n = w.n;
    let k = w.k;
    let n_tiles = w.n_tiles;
    let a_ptr = CPtr(x.as_ptr());
    let b_ptr = CPtr(w.packed.as_ptr());
    let c_ptr = MPtr(out.as_mut_ptr());
    // n is packed with full 16-wide panels (callers require n % 16 == 0).
    debug_assert_eq!(n, n_tiles * NR);

    let m_main = m - m % MR;
    let m_tiles = m_main / MR;
    // Parallelize across N panels so a packed 16-column slab of W stays hot.
    // This is important for wide FFN (`n=5120`); the earlier m-parallel loop
    // was slower than gemm 0.18 on that shape.
    (0..n_tiles).into_par_iter().for_each(|t| {
        let mut k0 = 0;
        while k0 < k {
            let k1 = (k0 + kc).min(k);
            let first = k0 == 0;
            for mt in 0..m_tiles {
                let i0 = mt * MR;
                // Fresh acc per kc panel; later panels are added into C
                // (gemm 0.18's beta=1 update), not continued as one FMA chain.
                let mut acc = [_mm512_setzero_ps(); MR];
                let mut p = k0;
                while p + 4 <= k1 {
                    for u in 0..4 {
                        let bv = _mm512_loadu_ps(b_ptr.add((t * k + p + u) * NR));
                        for r in 0..MR {
                            let av = _mm512_set1_ps(*a_ptr.add((i0 + r) * k + p + u));
                            acc[r] = _mm512_fmadd_ps(av, bv, acc[r]);
                        }
                    }
                    p += 4;
                }
                while p < k1 {
                    let bv = _mm512_loadu_ps(b_ptr.add((t * k + p) * NR));
                    for r in 0..MR {
                        let av = _mm512_set1_ps(*a_ptr.add((i0 + r) * k + p));
                        acc[r] = _mm512_fmadd_ps(av, bv, acc[r]);
                    }
                    p += 1;
                }
                for r in 0..MR {
                    let dst = c_ptr.add((i0 + r) * n + t * NR);
                    if first {
                        _mm512_storeu_ps(dst, acc[r]);
                    } else {
                        _mm512_storeu_ps(dst, _mm512_add_ps(_mm512_loadu_ps(dst), acc[r]));
                    }
                }
            }
            for i in m_main..m {
                let mut acc = _mm512_setzero_ps();
                let mut p = k0;
                while p + 4 <= k1 {
                    for u in 0..4 {
                        let bv = _mm512_loadu_ps(b_ptr.add((t * k + p + u) * NR));
                        let av = _mm512_set1_ps(*a_ptr.add(i * k + p + u));
                        acc = _mm512_fmadd_ps(av, bv, acc);
                    }
                    p += 4;
                }
                while p < k1 {
                    let bv = _mm512_loadu_ps(b_ptr.add((t * k + p) * NR));
                    let av = _mm512_set1_ps(*a_ptr.add(i * k + p));
                    acc = _mm512_fmadd_ps(av, bv, acc);
                    p += 1;
                }
                let dst = c_ptr.add(i * n + t * NR);
                if first {
                    _mm512_storeu_ps(dst, acc);
                } else {
                    _mm512_storeu_ps(dst, _mm512_add_ps(_mm512_loadu_ps(dst), acc));
                }
            }
            k0 = k1;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_tensor::{gemm_row_major, CpuWeight};

    fn close(a: &[f32], b: &[f32], tol: f32) -> f32 {
        let mut max = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max = max.max((x - y).abs());
        }
        assert!(max <= tol, "max abs err {max} > {tol}");
        max
    }

    #[test]
    fn packed_matches_gemm_small() {
        let m = 20;
        let n = 32;
        let k = 24;
        let mut w = vec![0.0f32; n * k];
        let mut x = vec![0.0f32; m * k];
        for i in 0..w.len() {
            w[i] = ((i as i32 * 13) % 101) as f32 * 0.01 - 0.5;
        }
        for i in 0..x.len() {
            x[i] = ((i as i32 * 7) % 89) as f32 * 0.02 - 0.8;
        }
        let packed = Fp32Packed::pack(&w, n, k);
        let got = linear_packed(&x, &packed, m);
        let mut gold = vec![0.0f32; m * n];
        gemm_row_major(&mut gold, &x, &CpuWeight::new(w, n, k), m, 0.0);
        let err = close(&got, &gold, 2e-4);
        eprintln!("packed vs gemm 20x32x24 max abs err {err:.3e}");
    }

    #[test]
    fn packed_matches_gemm_tail_rows() {
        // 12 = 8 + 4 remainder, n multiple of 16
        let m = 12;
        let n = 16;
        let k = 16;
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.001 - 0.2).collect();
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.002 - 0.1).collect();
        let packed = Fp32Packed::pack(&w, n, k);
        let got = linear_packed(&x, &packed, m);
        let mut gold = vec![0.0f32; m * n];
        gemm_row_major(&mut gold, &x, &CpuWeight::new(w, n, k), m, 0.0);
        close(&got, &gold, 2e-4);
    }

    #[test]
    fn packed_bit_identical_to_gemm_encoder_shapes() {
        let m = 1500;
        let shapes = [(1280, 1280), (5120, 1280), (1280, 5120)];
        for (n, k) in shapes {
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i * 17) % 101) as f32 * 0.01 - 0.5)
                .collect();
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 11) % 89) as f32 * 0.02 - 0.8)
                .collect();
            let packed = Fp32Packed::pack(&w, n, k);
            let kc = gemm_fma_kc(n, m, k);
            let mut gold = vec![0.0f32; m * n];
            gemm_row_major(&mut gold, &x, &CpuWeight::new(w, n, k), m, 0.0);
            let got = linear_packed(&x, &packed, m);
            let mut max = 0.0f32;
            let mut n_diff = 0usize;
            for (a, b) in got.iter().zip(gold.iter()) {
                let d = (a - b).abs();
                if d != 0.0 {
                    n_diff += 1;
                }
                max = max.max(d);
            }
            eprintln!(
                "1500x{n}x{k} kc={kc}: maxerr {:.3e}  ndiff {}",
                max, n_diff
            );
            assert_eq!(n_diff, 0, "1500x{n}x{k} not bit-identical, maxerr {max}");
        }
    }

    #[test]
    #[ignore]
    fn bench_encoder_like_linear() {
        let m = 1500;
        let shapes = [(1280, 1280), (5120, 1280), (1280, 5120)];
        for (n, k) in shapes {
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i * 17) % 101) as f32 * 0.01 - 0.5)
                .collect();
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 11) % 89) as f32 * 0.02 - 0.8)
                .collect();
            let packed = Fp32Packed::pack(&w, n, k);
            let cw = CpuWeight::new(w, n, k);
            let mut gold = vec![0.0f32; m * n];
            // warmup
            let _ = linear_packed(&x, &packed, m);
            gemm_row_major(&mut gold, &x, &cw, m, 0.0);
            let t0 = std::time::Instant::now();
            let got = linear_packed(&x, &packed, m);
            let packed_s = t0.elapsed();
            let t0 = std::time::Instant::now();
            gemm_row_major(&mut gold, &x, &cw, m, 0.0);
            let gemm_s = t0.elapsed();
            let mut max = 0.0f32;
            for (a, b) in got.iter().zip(gold.iter()) {
                max = max.max((a - b).abs());
            }
            eprintln!(
                "1500x{n}x{k}: packed {:.3}s  gemm {:.3}s  speedup {:.2}x  maxerr {:.3e}",
                packed_s.as_secs_f64(),
                gemm_s.as_secs_f64(),
                gemm_s.as_secs_f64() / packed_s.as_secs_f64(),
                max
            );
            assert!(max < 5e-3, "encoder-like max err {max}");
        }
    }
}
