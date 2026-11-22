//! CrisperWhisper 2.0 — Rust CPU port (wgpu backend reserved, not implemented).

pub mod attn_kernels;
pub mod audio;
pub mod config;
pub mod cpu_tensor;
pub mod ct2_math;
pub mod decoder;
pub mod encoder;
pub mod fp32_gemm;
pub mod early_eot;
pub mod error;
pub mod fallback;
pub mod features;
pub mod int8_gemm;
pub mod loop_detection;
pub mod longform;
pub mod model;
pub mod prompt;
pub mod simd_math;
pub mod tokenizer;
pub mod weights;
pub mod word_timing;

pub use audio::{load_audio, SAMPLE_RATE};
pub use config::ModelConfig;
pub use cpu_tensor::{compute_type, int8_enabled, set_compute_type, set_int8_enabled, ComputeType};
pub use error::{Error, Result};
pub use features::FeatureExtractor;
pub use model::{
    resolve_model_dir, CrisperWhisper, TranscribeOptions, TranscribeResult, OFFICIAL_SIZES,
};
pub use prompt::{build_prompt, strip_prompt_artifacts, TranscriptMode};
pub use word_timing::WordTimestamp;
