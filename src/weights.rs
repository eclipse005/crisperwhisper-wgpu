//! Load official BF16 (or F16/F32) safetensors into fp32 CPU weights.
//! `encoder_blank_head` is recorded even though greedy decode does not use it.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use crate::config::ModelConfig;
use crate::cpu_tensor::{compute_type, CpuWeight, Linear};
use crate::error::{Error, Result};
use crate::int8_gemm::Int8Mat;

pub struct LayerNormWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

pub struct AttentionWeights {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
}

pub struct EncoderLayerWeights {
    pub self_attn_ln: LayerNormWeights,
    pub self_attn: AttentionWeights,
    pub final_ln: LayerNormWeights,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct EncoderWeights {
    pub conv1_weight: Vec<f32>, // [out, in, k]
    pub conv1_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub conv1_int8: Option<Int8Mat>,
    pub conv2_int8: Option<Int8Mat>,
    pub conv_in: usize,
    pub conv_out: usize,
    pub conv_k: usize,
    pub embed_positions: Vec<f32>, // [max_source, d]
    pub layers: Vec<EncoderLayerWeights>,
    pub layer_norm: LayerNormWeights,
}

pub struct DecoderLayerWeights {
    pub self_attn_ln: LayerNormWeights,
    pub self_attn: AttentionWeights,
    pub encoder_attn_ln: LayerNormWeights,
    pub encoder_attn: AttentionWeights,
    pub final_ln: LayerNormWeights,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct DecoderWeights {
    pub embed_tokens: Vec<f32>,    // [vocab, d]
    pub embed_positions: Vec<f32>, // [max_target, d]
    pub embed_tokens_int8: Option<Int8Mat>,
    pub layers: Vec<DecoderLayerWeights>,
    pub layer_norm: LayerNormWeights,
}

pub struct ModelWeights {
    pub config: ModelConfig,
    pub encoder: EncoderWeights,
    pub decoder: DecoderWeights,
    /// Present on official checkpoints; unused in the phase-1 greedy path.
    pub encoder_blank_head: Linear,
}

trait WeightSrc {
    fn has(&self, name: &str) -> bool;
    fn vec(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>>;

    fn linear(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        out: usize,
        inn: usize,
        is_encoder: bool,
    ) -> Result<Linear> {
        self.linear_force_fp32(weight_name, bias_name, out, inn, is_encoder, false)
    }

    fn linear_force_fp32(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        out: usize,
        inn: usize,
        is_encoder: bool,
        force_fp32: bool,
    ) -> Result<Linear> {
        let w = self.vec(weight_name, &[out, inn])?;
        let bias = match bias_name {
            Some(b) => Some(self.vec(b, &[out])?),
            None => None,
        };
        let mut lin = Linear {
            weight: CpuWeight::new(w, out, inn),
            bias,
            int8: None,
            fp32_packed: None,
            is_encoder,
        };
        if crate::cpu_tensor::compute_type().packs_int8() && !force_fp32 {
            lin.pack_int8();
        } else if is_encoder {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                lin.pack_fp32();
            }
        }
        Ok(lin)
    }

    fn ln(&self, prefix: &str, dim: usize) -> Result<LayerNormWeights> {
        Ok(LayerNormWeights {
            weight: self.vec(&format!("{prefix}.weight"), &[dim])?,
            bias: self.vec(&format!("{prefix}.bias"), &[dim])?,
        })
    }

    fn attn(
        &self,
        prefix: &str,
        d: usize,
        n_heads: usize,
        is_encoder: bool,
    ) -> Result<AttentionWeights> {
        self.attn_force_fp32(prefix, d, n_heads, is_encoder, false)
    }

    fn attn_force_fp32(
        &self,
        prefix: &str,
        d: usize,
        n_heads: usize,
        is_encoder: bool,
        force_fp32: bool,
    ) -> Result<AttentionWeights> {
        if d % n_heads != 0 {
            return Err(Error::Model(format!(
                "d_model {d} not divisible by n_heads {n_heads}"
            )));
        }
        Ok(AttentionWeights {
            q: self.linear_force_fp32(
                &format!("{prefix}.q_proj.weight"),
                Some(&format!("{prefix}.q_proj.bias")),
                d,
                d,
                is_encoder,
                force_fp32,
            )?,
            k: self.linear_force_fp32(
                &format!("{prefix}.k_proj.weight"),
                None,
                d,
                d,
                is_encoder,
                force_fp32,
            )?,
            v: self.linear_force_fp32(
                &format!("{prefix}.v_proj.weight"),
                Some(&format!("{prefix}.v_proj.bias")),
                d,
                d,
                is_encoder,
                force_fp32,
            )?,
            out: self.linear_force_fp32(
                &format!("{prefix}.out_proj.weight"),
                Some(&format!("{prefix}.out_proj.bias")),
                d,
                d,
                is_encoder,
                force_fp32,
            )?,
            n_heads,
            head_dim: d / n_heads,
        })
    }
}

struct StLoader<'a> {
    tensors: SafeTensors<'a>,
}

impl<'a> WeightSrc for StLoader<'a> {
    fn has(&self, name: &str) -> bool {
        self.tensors.tensor(name).is_ok()
    }

    fn vec(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
        let t = self
            .tensors
            .tensor(name)
            .map_err(|e| Error::Model(format!("missing tensor {name}: {e}")))?;
        if t.shape() != expected {
            return Err(Error::Model(format!(
                "tensor {name} shape {:?} != expected {:?}",
                t.shape(),
                expected
            )));
        }
        tensor_to_f32(t.data(), t.dtype(), name)
    }
}

/// File-backed safetensors source: reads only the JSON header plus one tensor
/// at a time via positional reads (`pread` / Windows `seek_read`), so the
/// 1.5 GiB official checkpoint is never kept resident as a whole mmap.
/// Used by `ModelWeights::load`.
struct FileLoader {
    file: File,
    data_start: u64,
    infos: HashMap<String, TensorLoc>,
}

struct TensorLoc {
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl FileLoader {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .map_err(|e| Error::Model(format!("safetensors header length: {e}")))?;
        let n = u64::from_le_bytes(len_buf) as usize;
        // Official turbo header is a few hundred KiB; reject obviously-wrong sizes.
        if n < 2 || n > 64 * 1024 * 1024 {
            return Err(Error::Model(format!("invalid safetensors header size {n}")));
        }
        let mut header = vec![0u8; n];
        file.read_exact(&mut header)
            .map_err(|e| Error::Model(format!("safetensors header: {e}")))?;
        let value: serde_json::Value = serde_json::from_slice(&header)
            .map_err(|e| Error::Model(format!("safetensors header json: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| Error::Model("safetensors header is not an object".into()))?;
        let mut infos = HashMap::new();
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = parse_st_dtype(
                info.get("dtype")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Model(format!("tensor {name} missing dtype")))?,
            )?;
            let shape = info
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Model(format!("tensor {name} missing shape")))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|n| n as usize)
                        .ok_or_else(|| Error::Model(format!("tensor {name} bad shape")))
                })
                .collect::<Result<Vec<_>>>()?;
            let offs = info
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Model(format!("tensor {name} missing data_offsets")))?;
            if offs.len() != 2 {
                return Err(Error::Model(format!("tensor {name} bad data_offsets")));
            }
            let start = offs[0]
                .as_u64()
                .ok_or_else(|| Error::Model(format!("tensor {name} bad data_offsets")))?
                as usize;
            let end = offs[1]
                .as_u64()
                .ok_or_else(|| Error::Model(format!("tensor {name} bad data_offsets")))?
                as usize;
            if end < start {
                return Err(Error::Model(format!("tensor {name} inverted data_offsets")));
            }
            infos.insert(
                name.clone(),
                TensorLoc {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            file,
            data_start: 8 + n as u64,
            infos,
        })
    }

    fn read_bytes(&self, start: usize, end: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; end - start];
        let off = self.data_start + start as u64;
        #[cfg(unix)]
        {
            self.file
                .read_exact_at(&mut buf, off)
                .map_err(|e| Error::Model(format!("pread tensor at {off}: {e}")))?;
            return Ok(buf);
        }
        #[cfg(windows)]
        {
            // seek_read is the pread analogue: it does not move the file
            // cursor, so later tensor reads stay independent. Short reads
            // are possible, so loop until the buffer is filled.
            let mut got = 0usize;
            while got < buf.len() {
                let n = self
                    .file
                    .seek_read(&mut buf[got..], off + got as u64)
                    .map_err(|e| {
                        Error::Model(format!("seek_read tensor at {}: {e}", off + got as u64))
                    })?;
                if n == 0 {
                    return Err(Error::Model(format!(
                        "seek_read tensor at {off}: unexpected EOF"
                    )));
                }
                got += n;
            }
            return Ok(buf);
        }
        #[cfg(not(any(unix, windows)))]
        {
            use std::io::{Seek, SeekFrom};
            let mut file = self
                .file
                .try_clone()
                .map_err(|e| Error::Model(format!("clone fd for tensor read at {off}: {e}")))?;
            file.seek(SeekFrom::Start(off))
                .map_err(|e| Error::Model(format!("seek tensor at {off}: {e}")))?;
            file.read_exact(&mut buf)
                .map_err(|e| Error::Model(format!("read tensor at {off}: {e}")))?;
            Ok(buf)
        }
    }
}

fn parse_st_dtype(s: &str) -> Result<Dtype> {
    match s {
        "F32" => Ok(Dtype::F32),
        "F16" => Ok(Dtype::F16),
        "BF16" => Ok(Dtype::BF16),
        other => Err(Error::Model(format!(
            "unsupported safetensors dtype {other}"
        ))),
    }
}

impl WeightSrc for FileLoader {
    fn has(&self, name: &str) -> bool {
        self.infos.contains_key(name)
    }

    fn vec(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
        let loc = self
            .infos
            .get(name)
            .ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
        if loc.shape != expected {
            return Err(Error::Model(format!(
                "tensor {name} shape {:?} != expected {:?}",
                loc.shape, expected
            )));
        }
        let bytes = self.read_bytes(loc.start, loc.end)?;
        tensor_to_f32(&bytes, loc.dtype, name)
    }
}

fn tensor_to_f32(data: &[u8], dtype: Dtype, name: &str) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if data.len() % 4 != 0 {
                return Err(Error::Model(format!("bad f32 length for {name}")));
            }
            let mut out = Vec::with_capacity(data.len() / 4);
            for c in data.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
            Ok(out)
        }
        Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(Error::Model(format!("bad f16 length for {name}")));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for c in data.chunks_exact(2) {
                let bits = u16::from_le_bytes(c.try_into().unwrap());
                out.push(f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err(Error::Model(format!("bad bf16 length for {name}")));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for c in data.chunks_exact(2) {
                let bits = u16::from_le_bytes(c.try_into().unwrap());
                out.push(bf16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        other => Err(Error::Model(format!(
            "unsupported dtype {other:?} for tensor {name}"
        ))),
    }
}

fn pack_int8_mat(weight: &[f32], rows: usize, cols: usize) -> Int8Mat {
    let mut q = Int8Mat::quantize(weight, rows, cols);
    q.discard_unpacked_if_vnni();
    q
}

fn build_weights(cfg: ModelConfig, loader: &impl WeightSrc) -> Result<ModelWeights> {
    let d = cfg.d_model;
    let n_mels = cfg.feature_size;
    let k = 3usize;
    let int8 = compute_type().packs_int8();

    let conv1_weight = loader.vec("model.encoder.conv1.weight", &[d, n_mels, k])?;
    let conv1_bias = loader.vec("model.encoder.conv1.bias", &[d])?;
    let conv2_weight = loader.vec("model.encoder.conv2.weight", &[d, d, k])?;
    let conv2_bias = loader.vec("model.encoder.conv2.bias", &[d])?;
    let (conv1_int8, conv1_weight, conv2_int8, conv2_weight) = if int8 {
        let q1 = pack_int8_mat(&conv1_weight, d, n_mels * k);
        let q2 = pack_int8_mat(&conv2_weight, d, d * k);
        (Some(q1), Vec::new(), Some(q2), Vec::new())
    } else {
        (None, conv1_weight, None, conv2_weight)
    };

    let encoder = EncoderWeights {
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        conv1_int8,
        conv2_int8,
        conv_in: n_mels,
        conv_out: d,
        conv_k: k,
        embed_positions: loader.vec(
            "model.encoder.embed_positions.weight",
            &[cfg.max_source_positions, d],
        )?,
        layers: (0..cfg.encoder_layers)
            .map(|i| {
                let p = format!("model.encoder.layers.{i}");
                // Optional INT8: last encoder layer stays official fp32 so the
                // features handed to the decoder are closer to gold. One turbo
                // layer still fits under the CT2 peak-RSS budget.
                // First-layer / first+last / fp32-conv variants improved
                // `popcor-` but moved JFK timestamps away from gold (including
                // `country.` end 10.44 → 10.46 on some of them). Do not add
                // those back without a JFK-and-full regression that does not
                // slip vs gold.
                let last_fp32 = int8 && i + 1 == cfg.encoder_layers;
                Ok(EncoderLayerWeights {
                    self_attn_ln: loader.ln(&format!("{p}.self_attn_layer_norm"), d)?,
                    self_attn: loader.attn_force_fp32(
                        &format!("{p}.self_attn"),
                        d,
                        cfg.encoder_attention_heads,
                        true,
                        last_fp32,
                    )?,
                    final_ln: loader.ln(&format!("{p}.final_layer_norm"), d)?,
                    fc1: loader.linear_force_fp32(
                        &format!("{p}.fc1.weight"),
                        Some(&format!("{p}.fc1.bias")),
                        cfg.encoder_ffn_dim,
                        d,
                        true,
                        last_fp32,
                    )?,
                    fc2: loader.linear_force_fp32(
                        &format!("{p}.fc2.weight"),
                        Some(&format!("{p}.fc2.bias")),
                        d,
                        cfg.encoder_ffn_dim,
                        true,
                        last_fp32,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        layer_norm: loader.ln("model.encoder.layer_norm", d)?,
    };

    let embed_tokens = loader.vec("model.decoder.embed_tokens.weight", &[cfg.vocab_size, d])?;
    let (embed_tokens_int8, embed_tokens) = if int8 {
        let q = pack_int8_mat(&embed_tokens, cfg.vocab_size, d);
        (Some(q), Vec::new())
    } else {
        (None, embed_tokens)
    };

    let decoder = DecoderWeights {
        embed_tokens,
        embed_tokens_int8,
        embed_positions: loader.vec(
            "model.decoder.embed_positions.weight",
            &[cfg.max_target_positions, d],
        )?,
        layers: (0..cfg.decoder_layers)
            .map(|i| {
                let p = format!("model.decoder.layers.{i}");
                Ok(DecoderLayerWeights {
                    self_attn_ln: loader.ln(&format!("{p}.self_attn_layer_norm"), d)?,
                    self_attn: loader.attn(
                        &format!("{p}.self_attn"),
                        d,
                        cfg.decoder_attention_heads,
                        false,
                    )?,
                    encoder_attn_ln: loader.ln(&format!("{p}.encoder_attn_layer_norm"), d)?,
                    encoder_attn: loader.attn(
                        &format!("{p}.encoder_attn"),
                        d,
                        cfg.decoder_attention_heads,
                        false,
                    )?,
                    final_ln: loader.ln(&format!("{p}.final_layer_norm"), d)?,
                    fc1: loader.linear(
                        &format!("{p}.fc1.weight"),
                        Some(&format!("{p}.fc1.bias")),
                        cfg.decoder_ffn_dim,
                        d,
                        false,
                    )?,
                    fc2: loader.linear(
                        &format!("{p}.fc2.weight"),
                        Some(&format!("{p}.fc2.bias")),
                        d,
                        cfg.decoder_ffn_dim,
                        false,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        layer_norm: loader.ln("model.decoder.layer_norm", d)?,
    };

    if !loader.has("encoder_blank_head.weight") || !loader.has("encoder_blank_head.bias") {
        return Err(Error::Model(
            "checkpoint is missing encoder_blank_head; official weights always include it".into(),
        ));
    }
    let encoder_blank_head = loader.linear(
        "encoder_blank_head.weight",
        Some("encoder_blank_head.bias"),
        1,
        d,
        true,
    )?;

    Ok(ModelWeights {
        config: cfg,
        encoder,
        decoder,
        encoder_blank_head,
    })
}

impl ModelWeights {
    pub fn load(cfg: ModelConfig) -> Result<Self> {
        let path = find_safetensors(&cfg.model_dir)?;
        let loader = FileLoader::open(&path)?;
        build_weights(cfg, &loader)
    }

    pub fn load_from_bytes(cfg: ModelConfig, bytes: &[u8]) -> Result<Self> {
        let loader = StLoader {
            tensors: SafeTensors::deserialize(bytes)?,
        };
        build_weights(cfg, &loader)
    }
}

fn find_safetensors(dir: &Path) -> Result<std::path::PathBuf> {
    let p = dir.join("model.safetensors");
    if p.exists() {
        return Ok(p);
    }
    // Some snapshots use a shard index; reject rather than silently picking one.
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        return Err(Error::Model(format!(
            "sharded safetensors at {} are not yet supported; flatten to model.safetensors",
            dir.display()
        )));
    }
    Err(Error::Model(format!(
        "no model.safetensors in {}",
        dir.display()
    )))
}
