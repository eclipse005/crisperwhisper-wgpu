use crate::cpu_tensor::{compute_type, gelu_erf_inplace, layer_norm, layer_norm_ct2};
use crate::encoder::{attention, AttnOutput};
use crate::weights::{DecoderLayerWeights, DecoderWeights, LayerNormWeights};

pub struct LayerKv {
    pub self_k: Vec<f32>,
    pub self_v: Vec<f32>,
    pub cross_k: Vec<f32>,
    pub cross_v: Vec<f32>,
    pub cross_ready: bool,
}

pub struct DecoderCache {
    pub layers: Vec<LayerKv>,
    pub seq_len: usize,
}

impl DecoderCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers)
                .map(|_| LayerKv {
                    self_k: Vec::new(),
                    self_v: Vec::new(),
                    cross_k: Vec::new(),
                    cross_v: Vec::new(),
                    cross_ready: false,
                })
                .collect(),
            seq_len: 0,
        }
    }
}

pub struct DecoderStep {
    pub hidden: Vec<f32>,
    /// Per decoder layer, last-query cross-attention weights `[n_heads, enc_len]`.
    pub cross_last: Option<Vec<Vec<f32>>>,
}

/// Forward a chunk of decoder tokens. Returns hidden states `[tq, d]` after the
/// final layer norm (ready for the tied `proj_out`).
pub fn decoder_forward(
    dec: &DecoderWeights,
    ids: &[u32],
    encoder_out: &[f32],
    enc_len: usize,
    cache: &mut DecoderCache,
    ln_eps: f32,
    collect_cross_attn: bool,
) -> DecoderStep {
    let d = dec.layer_norm.weight.len();
    let vocab = if let Some(q) = dec.embed_tokens_int8.as_ref() {
        q.rows
    } else {
        dec.embed_tokens.len() / d
    };
    let tq = ids.len();
    let pos0 = cache.seq_len;
    assert!(pos0 + tq <= dec.embed_positions.len() / d, "decoder positions overflow");

    let mut hidden = vec![0.0f32; tq * d];
    for (i, &id) in ids.iter().enumerate() {
        let id = id as usize;
        assert!(id < vocab, "token id {id} >= vocab {vocab}");
        let pos = pos0 + i;
        let p = &dec.embed_positions[pos * d..(pos + 1) * d];
        let row = &mut hidden[i * d..(i + 1) * d];
        if let Some(q) = dec.embed_tokens_int8.as_ref() {
            q.dequant_row(id, row);
            for c in 0..d {
                row[c] += p[c];
            }
        } else {
            let e = &dec.embed_tokens[id * d..(id + 1) * d];
            for c in 0..d {
                row[c] = e[c] + p[c];
            }
        }
    }

    let mut cross_last = if collect_cross_attn {
        Some(Vec::with_capacity(dec.layers.len()))
    } else {
        None
    };
    for (layer, kv) in dec.layers.iter().zip(cache.layers.iter_mut()) {
        let (h, xw) = decoder_layer(
            layer, &hidden, tq, d, encoder_out, enc_len, kv, ln_eps, collect_cross_attn,
        );
        hidden = h;
        if let (Some(ref mut acc), Some(w)) = (cross_last.as_mut(), xw) {
            acc.push(w);
        }
    }
    cache.seq_len += tq;
    DecoderStep {
        hidden: ln(&dec.layer_norm, &hidden, tq, d, ln_eps),
        cross_last,
    }
}

fn ln(w: &LayerNormWeights, x: &[f32], seq: usize, dim: usize, eps: f32) -> Vec<f32> {
    // See encoder.rs: INT8 keeps the gold-close 8-wide leftover LN.
    if compute_type().uses_int8() {
        layer_norm_ct2(x, seq, dim, &w.weight, &w.bias, eps)
    } else {
        layer_norm(x, seq, dim, &w.weight, &w.bias, eps)
    }
}

fn decoder_layer(
    layer: &DecoderLayerWeights,
    x: &[f32],
    tq: usize,
    d: usize,
    encoder_out: &[f32],
    enc_len: usize,
    kv: &mut LayerKv,
    eps: f32,
    collect_cross_attn: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    // Self-attention (causal). New K/V are appended into the cache in place.
    let normed = ln(&layer.self_attn_ln, x, tq, d, eps);
    let AttnOutput { output, .. } = attention(
        &normed,
        tq,
        &layer.self_attn,
        &normed,
        tq,
        None,
        None,
        true,
        false,
        Some((&mut kv.self_k, &mut kv.self_v)),
    );
    // Reuse the attention output buffer for the residual add.
    let mut hidden = output;
    for i in 0..tq * d {
        hidden[i] += x[i];
    }

    // Cross-attention. Encoder K/V are computed once, stored in the cache,
    // and reused on later steps without recopying the 1500-token buffer.
    let normed = ln(&layer.encoder_attn_ln, &hidden, tq, d, eps);
    let (kv_src, tk_new) = if kv.cross_ready {
        (&[] as &[f32], 0)
    } else {
        (encoder_out, enc_len)
    };
    let out = attention(
        &normed,
        tq,
        &layer.encoder_attn,
        kv_src,
        tk_new,
        None,
        None,
        false,
        collect_cross_attn,
        Some((&mut kv.cross_k, &mut kv.cross_v)),
    );
    kv.cross_ready = true;
    for i in 0..tq * d {
        hidden[i] += out.output[i];
    }

    let normed = ln(&layer.final_ln, &hidden, tq, d, eps);
    let mut ff = layer.fc1.forward(&normed, tq);
    gelu_erf_inplace(&mut ff);
    let ff = layer.fc2.forward(&ff, tq);
    for i in 0..tq * d {
        hidden[i] += ff[i];
    }
    (hidden, out.last_query_weights)
}
