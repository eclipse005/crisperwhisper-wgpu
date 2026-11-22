use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

/// Architecture + preprocessor + generation defaults loaded from a checkpoint directory.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub vocab_size: usize,
    pub activation_function: String,
    pub scale_embedding: bool,
    pub layer_norm_eps: f32,
    /// Mel bins. Must come from preprocessor_config.feature_size (128 turbo, 80 otherwise).
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub hop_length: usize,
    pub n_fft: usize,
    pub chunk_length: usize,
    pub n_samples: usize,
    pub nb_max_frames: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub decoder_start_token_id: u32,
    pub pad_token_id: u32,
    pub no_timestamps_token_id: Option<u32>,
    pub transcribe_token_id: Option<u32>,
    pub lang_to_id: HashMap<String, u32>,
    pub suppress_tokens: Vec<u32>,
    pub alignment_heads: Vec<(usize, usize)>,
    pub model_dir: std::path::PathBuf,
}

#[derive(Debug, Deserialize)]
struct HfConfig {
    d_model: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    encoder_attention_heads: usize,
    decoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    decoder_ffn_dim: usize,
    max_source_positions: usize,
    max_target_positions: usize,
    vocab_size: usize,
    #[serde(default = "default_gelu")]
    activation_function: String,
    #[serde(default)]
    scale_embedding: bool,
    #[serde(default = "default_ln_eps")]
    layer_norm_eps: f32,
    #[serde(default)]
    num_mel_bins: Option<usize>,
    bos_token_id: u32,
    eos_token_id: u32,
    decoder_start_token_id: u32,
    pad_token_id: u32,
}

fn default_gelu() -> String {
    "gelu".into()
}
fn default_ln_eps() -> f32 {
    1e-5
}

#[derive(Debug, Deserialize)]
struct PreprocessorConfig {
    feature_size: usize,
    #[serde(default = "default_sr")]
    sampling_rate: u32,
    #[serde(default = "default_hop")]
    hop_length: usize,
    #[serde(default = "default_nfft")]
    n_fft: usize,
    #[serde(default = "default_chunk")]
    chunk_length: usize,
    n_samples: Option<usize>,
    nb_max_frames: Option<usize>,
}

fn default_sr() -> u32 {
    16_000
}
fn default_hop() -> usize {
    160
}
fn default_nfft() -> usize {
    400
}
fn default_chunk() -> usize {
    30
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    no_timestamps_token_id: Option<u32>,
    #[serde(default)]
    lang_to_id: Option<HashMap<String, u32>>,
    #[serde(default)]
    task_to_id: Option<HashMap<String, u32>>,
    #[serde(default)]
    suppress_tokens: Option<Vec<i64>>,
    #[serde(default)]
    alignment_heads: Option<Vec<(usize, usize)>>,
}

impl ModelConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let hf: HfConfig = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        let pre: PreprocessorConfig = serde_json::from_str(&std::fs::read_to_string(
            model_dir.join("preprocessor_config.json"),
        )?)?;
        let gen: GenerationConfig = serde_json::from_str(&std::fs::read_to_string(
            model_dir.join("generation_config.json"),
        )?)?;

        if let Some(n) = hf.num_mel_bins {
            if n != pre.feature_size {
                return Err(Error::Model(format!(
                    "config.num_mel_bins={n} != preprocessor_config.feature_size={}",
                    pre.feature_size
                )));
            }
        }
        if pre.feature_size == 0 {
            return Err(Error::Model(
                "preprocessor_config.feature_size is missing or zero".into(),
            ));
        }

        let n_samples = pre.n_samples.unwrap_or(pre.chunk_length * pre.sampling_rate as usize);
        let nb_max_frames = pre.nb_max_frames.unwrap_or(n_samples / pre.hop_length);

        let mut lang_to_id = HashMap::new();
        if let Some(map) = gen.lang_to_id {
            for (token, id) in map {
                // "<|en|>" -> "en"
                let lang = token.trim_start_matches("<|").trim_end_matches("|>");
                if !lang.is_empty() {
                    lang_to_id.insert(lang.to_string(), id);
                }
            }
        }

        let transcribe_token_id = gen
            .task_to_id
            .as_ref()
            .and_then(|m| m.get("transcribe").copied());

        let suppress_tokens = gen
            .suppress_tokens
            .unwrap_or_default()
            .into_iter()
            .filter(|t| *t >= 0)
            .map(|t| t as u32)
            .collect();

        Ok(Self {
            d_model: hf.d_model,
            encoder_layers: hf.encoder_layers,
            decoder_layers: hf.decoder_layers,
            encoder_attention_heads: hf.encoder_attention_heads,
            decoder_attention_heads: hf.decoder_attention_heads,
            encoder_ffn_dim: hf.encoder_ffn_dim,
            decoder_ffn_dim: hf.decoder_ffn_dim,
            max_source_positions: hf.max_source_positions,
            max_target_positions: hf.max_target_positions,
            vocab_size: hf.vocab_size,
            activation_function: hf.activation_function,
            scale_embedding: hf.scale_embedding,
            layer_norm_eps: hf.layer_norm_eps,
            feature_size: pre.feature_size,
            sampling_rate: pre.sampling_rate,
            hop_length: pre.hop_length,
            n_fft: pre.n_fft,
            chunk_length: pre.chunk_length,
            n_samples,
            nb_max_frames,
            bos_token_id: hf.bos_token_id,
            eos_token_id: hf.eos_token_id,
            decoder_start_token_id: hf.decoder_start_token_id,
            pad_token_id: hf.pad_token_id,
            no_timestamps_token_id: gen.no_timestamps_token_id,
            transcribe_token_id,
            lang_to_id,
            suppress_tokens,
            alignment_heads: gen.alignment_heads.unwrap_or_default(),
            model_dir: model_dir.to_path_buf(),
        })
    }

    pub fn language_id(&self, language: &str) -> Result<u32> {
        self.lang_to_id.get(language).copied().ok_or_else(|| {
            Error::Model(format!(
                "unknown or unsupported language {language:?}; language must be specified"
            ))
        })
    }
}
