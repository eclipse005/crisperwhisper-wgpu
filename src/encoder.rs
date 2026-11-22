use crate::cpu_tensor::{compute_type, conv1d, conv1d_int8, gelu_erf_inplace, layer_norm, layer_norm_ct2};
use crate::error::{Error, Result};
use crate::weights::{AttentionWeights, EncoderWeights, LayerNormWeights};

pub fn encode(enc: &EncoderWeights, mel: &[f32], n_mels: usize, n_frames: usize, ln_eps: f32) -> Result<Vec<f32>> {
    if n_frames != 3000 {
        return Err(Error::Model(format!(
            "encoder expects 3000 mel frames (30s window), got {n_frames}"
        )));
    }
    if mel.len() != n_mels * n_frames {
        return Err(Error::Model(format!(
            "mel length {} != n_mels {n_mels} * n_frames {n_frames}",
            mel.len()
        )));
    }
    let d = enc.conv_out;
    let mut h = if let Some(q) = enc.conv1_int8.as_ref() {
        conv1d_int8(mel, n_mels, n_frames, q, &enc.conv1_bias, d, enc.conv_k, 1, 1)
    } else {
        conv1d(mel, n_mels, n_frames, &enc.conv1_weight, &enc.conv1_bias, d, enc.conv_k, 1, 1)
    };
    gelu_erf_inplace(&mut h);
    let t1 = n_frames;
    h = if let Some(q) = enc.conv2_int8.as_ref() {
        conv1d_int8(&h, d, t1, q, &enc.conv2_bias, d, enc.conv_k, 2, 1)
    } else {
        conv1d(&h, d, t1, &enc.conv2_weight, &enc.conv2_bias, d, enc.conv_k, 2, 1)
    };
    gelu_erf_inplace(&mut h);
    let t = t1 / 2; // 1500
    if enc.embed_positions.len() != t * d {
        return Err(Error::Model(format!(
            "encoder positions {} != {t}*{d}",
            enc.embed_positions.len()
        )));
    }
    // permute [d, t] -> [t, d] and add frozen positional embeddings
    let mut hidden = vec![0.0f32; t * d];
    {
        use rayon::prelude::*;
        hidden.par_chunks_mut(d).enumerate().for_each(|(ti, row)| {
            for c in 0..d {
                row[c] = h[c * t + ti] + enc.embed_positions[ti * d + c];
            }
        });
    }

    for layer in &enc.layers {
        encoder_layer(layer, &mut hidden, t, d, ln_eps);
    }
    Ok(ln(&enc.layer_norm, &hidden, t, d, ln_eps))
}

fn ln(w: &LayerNormWeights, x: &[f32], seq: usize, dim: usize, eps: f32) -> Vec<f32> {
    // Default fp32 keeps the official two-pass kernel. Optional INT8 keeps
    // the gold-close 8-wide one-pass leftover (FMA sumsq, mean=sum/n).
    // Official two-pass on INT8 moved JFK `country.` end 10.44 → 10.46
    // and 0 → 3 words over 20 ms vs gold; do not switch INT8 to that, and
    // do not restore the CT2 AVX2 .so association either.
    if compute_type().uses_int8() {
        layer_norm_ct2(x, seq, dim, &w.weight, &w.bias, eps)
    } else {
        layer_norm(x, seq, dim, &w.weight, &w.bias, eps)
    }
}

fn encoder_layer(layer: &crate::weights::EncoderLayerWeights, hidden: &mut [f32], t: usize, d: usize, eps: f32) {
    let normed = ln(&layer.self_attn_ln, hidden, t, d, eps);
    let attn = attention(&normed, t, &layer.self_attn, &normed, t, None, None, false, false, None);
    debug_assert_eq!(attn.output.len(), hidden.len());
    add_inplace(hidden, &attn.output);
    let normed = ln(&layer.final_ln, hidden, t, d, eps);
    let mut ff = layer.fc1.forward(&normed, t);
    gelu_erf_inplace(&mut ff);
    let ff = layer.fc2.forward(&ff, t);
    add_inplace(hidden, &ff);
}

fn add_inplace(dst: &mut [f32], src: &[f32]) {
    use rayon::prelude::*;
    debug_assert_eq!(dst.len(), src.len());
    if dst.len() >= 2048 {
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d += *s);
    } else {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += *s;
        }
    }
}

pub struct AttnOutput {
    pub output: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    /// Post-softmax weights for the last query, `[n_heads, tk]`. Present when requested.
    pub last_query_weights: Option<Vec<f32>>,
}

/// Multi-head attention matching HF `WhisperAttention` (eager):
/// `Q = q_proj(x) * head_dim^{-0.5}`, scores = Q @ K^T with no extra scale.
pub fn attention(
    queries: &[f32],
    tq: usize,
    weights: &AttentionWeights,
    kv_src: &[f32],
    tk_new: usize,
    past_k: Option<&[f32]>,
    past_v: Option<&[f32]>,
    causal: bool,
    return_weights: bool,
    // Decoder self/cross KV cache: new keys/values are appended in place
    // and the buffer is used directly, so decode steps do not recopy the
    // growing cache (or the 1500-token encoder-side cross cache).
    kv_cache: Option<(&mut Vec<f32>, &mut Vec<f32>)>,
) -> AttnOutput {
    let d = weights.q.weight.cols;
    let n_heads = weights.n_heads;
    let head_dim = weights.head_dim;
    let scale = (head_dim as f32).sqrt().recip();

    let mut q = weights.q.forward(queries, tq);
    for v in q.iter_mut() {
        *v *= scale;
    }
    let k_new = if tk_new == 0 {
        Vec::new()
    } else {
        weights.k.forward(kv_src, tk_new)
    };
    let v_new = if tk_new == 0 {
        Vec::new()
    } else {
        weights.v.forward(kv_src, tk_new)
    };

    let t_past = if let Some((ck, _)) = kv_cache.as_ref() {
        ck.len() / d
    } else {
        past_k.map(|p| p.len() / d).unwrap_or(0)
    };
    let tk = t_past + tk_new;
    // Encoder (and any first chunk with no cache) can keep the freshly
    // projected K/V without an extra concat copy. The decoder passes a
    // cache and we append into it, so later steps never recopy past K/V.
    let mut cache_ptrs: Option<(*const f32, *const f32)> = None;
    let (k, v) = if let Some((ck, cv)) = kv_cache {
        if tk_new > 0 {
            ck.reserve(k_new.len());
            cv.reserve(v_new.len());
            ck.extend_from_slice(&k_new);
            cv.extend_from_slice(&v_new);
        }
        debug_assert_eq!(ck.len(), tk * d);
        debug_assert_eq!(cv.len(), tk * d);
        cache_ptrs = Some((ck.as_ptr(), cv.as_ptr()));
        (Vec::new(), Vec::new())
    } else if t_past == 0 {
        (k_new, v_new)
    } else {
        let mut k = vec![0.0f32; tk * d];
        let mut v = vec![0.0f32; tk * d];
        if let (Some(pk), Some(pv)) = (past_k, past_v) {
            k[..t_past * d].copy_from_slice(pk);
            v[..t_past * d].copy_from_slice(pv);
        }
        k[t_past * d..].copy_from_slice(&k_new);
        v[t_past * d..].copy_from_slice(&v_new);
        (k, v)
    };

    // Output ctx [tq, d] written per head with strided GEMM — no Q/K/V pack.
    // Heads write disjoint `head_dim`-wide slices (always 64 floats / 256 B in
    // official models), so adjacent heads do not share a cache line.
    let mut ctx = vec![0.0f32; tq * d];
    let mut last_query_weights = if return_weights {
        Some(vec![0.0f32; n_heads * tk])
    } else {
        None
    };

    #[derive(Copy, Clone)]
    struct MutPtr(*mut f32);
    unsafe impl Send for MutPtr {}
    unsafe impl Sync for MutPtr {}
    impl MutPtr {
        fn is_null(self) -> bool {
            self.0.is_null()
        }
        unsafe fn add(self, off: usize) -> *mut f32 {
            self.0.add(off)
        }
    }
    #[derive(Copy, Clone)]
    struct ConstPtr(*const f32);
    unsafe impl Send for ConstPtr {}
    unsafe impl Sync for ConstPtr {}
    impl ConstPtr {
        unsafe fn add(self, off: usize) -> *const f32 {
            self.0.add(off)
        }
    }

    let ctx_ptr = MutPtr(ctx.as_mut_ptr());
    let last_ptr = MutPtr(
        last_query_weights
            .as_mut()
            .map(|w| w.as_mut_ptr())
            .unwrap_or(std::ptr::null_mut()),
    );
    let q_ptr = ConstPtr(q.as_ptr());
    let (k_ptr, v_ptr) = if let Some((kp, vp)) = cache_ptrs {
        (ConstPtr(kp), ConstPtr(vp))
    } else {
        (ConstPtr(k.as_ptr()), ConstPtr(v.as_ptr()))
    };

    use crate::cpu_tensor::{gemm_bt_strided, gemm_strided, softmax_inplace_rows};
    use gemm::Parallelism;
    use rayon::prelude::*;

    // Parallelize across heads; each GEMM stays single-threaded so we do not
    // oversubscribe the rayon pool (the previous pack-then-Rayon(0) path did).
    //
    // Reuse a per-worker scores buffer (1500×1500 fp32 ≈ 9 MiB / head) so the
    // encoder does not allocate 20 of them every layer. The Vec is *taken out*
    // of the RefCell for the whole head — softmax uses rayon, which can steal
    // another head onto this thread and would otherwise panic with
    // "RefCell already borrowed".
    thread_local! {
        static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    if tq == 1 && head_dim == crate::attn_kernels::HEAD_DIM {
        // Fused tq=1 path: one pass over K and V for all heads, no gemm 0.18
        // packing. Causal masking is a no-op for a single newest query
        // (it may attend to every cached key, i.e. 0..=tk-1).
        let mut scores = vec![0.0f32; n_heads * tk];
        unsafe {
            crate::attn_kernels::scores_t1_k64(
                n_heads,
                tk,
                scores.as_mut_ptr(),
                q_ptr.0,
                k_ptr.0,
                d,
            );
        }
        softmax_inplace_rows(&mut scores, n_heads, tk);
        if return_weights && !last_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(scores.as_ptr(), last_ptr.0, n_heads * tk);
            }
        }
        unsafe {
            crate::attn_kernels::av_t1_k64(n_heads, tk, ctx_ptr.0, d, scores.as_ptr(), v_ptr.0);
        }
    } else {
    let run_head = |h: usize| {
        let q_off = h * head_dim;
        let mut scores = SCORES.with(|slot| slot.replace(Vec::new()));
        scores.clear();
        scores.resize(tq * tk, 0.0);
        unsafe {
            // Skinny-K kernel wins on encoder-sized squares (1500×1500).
            // Decoder steps and teacher-forced timestamp passes (small tq)
            // stay on the general GEMM.
            if head_dim == crate::attn_kernels::HEAD_DIM && tq >= 256 && tk >= 256 {
                crate::attn_kernels::gemm_bt_k64(
                    tq,
                    tk,
                    scores.as_mut_ptr(),
                    q_ptr.add(q_off),
                    d,
                    k_ptr.add(q_off),
                    d,
                );
            } else {
                gemm_bt_strided(
                    tq,
                    tk,
                    head_dim,
                    scores.as_mut_ptr(),
                    1,
                    tk as isize,
                    q_ptr.add(q_off),
                    1,
                    d as isize,
                    k_ptr.add(q_off),
                    d as isize,
                    1,
                    Parallelism::None,
                );
            }
        }
        if causal {
            // Query i (absolute position t_past + i, if this chunk is the newest
            // `tq == tk_new` tokens) may attend to keys 0..=(t_past + i).
            for i in 0..tq {
                let max_key = t_past + i;
                let row = &mut scores[i * tk..(i + 1) * tk];
                for j in (max_key + 1)..tk {
                    row[j] = f32::NEG_INFINITY;
                }
            }
        }
        softmax_inplace_rows(&mut scores, tq, tk);
        if return_weights && !last_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scores[(tq - 1) * tk..].as_ptr(),
                    last_ptr.add(h * tk),
                    tk,
                );
            }
        }
        unsafe {
            if head_dim == crate::attn_kernels::HEAD_DIM && tq >= 256 && tk >= 256 {
                crate::attn_kernels::gemm_av_k64(
                    tq,
                    tk,
                    ctx_ptr.add(q_off),
                    d,
                    scores.as_ptr(),
                    v_ptr.add(q_off),
                    d,
                );
            } else {
                gemm_strided(
                    tq,
                    head_dim,
                    tk,
                    ctx_ptr.add(q_off),
                    1,
                    d as isize,
                    scores.as_ptr(),
                    1,
                    tk as isize,
                    v_ptr.add(q_off),
                    1,
                    d as isize,
                    Parallelism::None,
                );
            }
        }
        SCORES.with(|slot| {
            let leftover = slot.replace(scores);
            if leftover.capacity() > slot.borrow().capacity() {
                slot.replace(leftover);
            }
        });
    };
    // Always one rayon task per head. A serial tq==1 path looked cheaper
    // on spawn cost but left 20×2 skinny gemm 0.18 calls on one core;
    // the previous 8-way head parallelism is faster in practice.
    (0..n_heads).into_par_iter().for_each(run_head);

    }
    let output = weights.out.forward(&ctx, tq);
    AttnOutput {
        output,
        keys: k,
        values: v,
        last_query_weights,
    }
}

pub fn linear_logits(hidden_last: &[f32], embed_tokens: &[f32], vocab: usize, d: usize) -> Vec<f32> {
    linear_logits_maybe_int8(hidden_last, embed_tokens, vocab, d, None)
}

pub fn linear_logits_maybe_int8(
    hidden_last: &[f32],
    embed_tokens: &[f32],
    vocab: usize,
    d: usize,
    embed_int8: Option<&crate::int8_gemm::Int8Mat>,
) -> Vec<f32> {
    if let Some(q) = embed_int8 {
        assert_eq!(q.rows, vocab);
        assert_eq!(q.cols, d);
        return crate::int8_gemm::linear_int8(hidden_last, q, None, 1);
    }
    // tied proj_out: logits = hidden @ embed_tokens^T, embed is [vocab, d]
    crate::cpu_tensor::matmul_b_t(hidden_last, embed_tokens, 1, vocab, d)
}
