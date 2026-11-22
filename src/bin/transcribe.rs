use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use crisperwhisper_wgpu::{
    resolve_model_dir, set_compute_type, ComputeType, CrisperWhisper, TranscriptMode,
    TranscribeOptions,
};

#[derive(Parser, Debug)]
#[command(name = "transcribe", about = "CrisperWhisper 2.0 Rust CPU transcriber")]
struct Args {
    /// Official size (large/turbo/medium/small) or a checkpoint directory.
    #[arg(long, default_value = "turbo")]
    model: String,

    /// Explicit checkpoint directory (overrides HF cache lookup).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Audio file (wav, flac, mp3, ...).
    #[arg(long)]
    audio: PathBuf,

    /// ISO 639-1 language code. Required — no auto detect.
    #[arg(long)]
    language: String,

    /// verbatim or intended.
    #[arg(long, default_value = "verbatim")]
    mode: String,

    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,

    /// Word-level timestamps (attention + mel blank + Viterbi).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    word_timestamps: bool,

    /// Official coverage-gated temperature fallback.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    temperature_fallback: bool,

    /// Official context-conditioned early-EOT recovery (longform only).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    early_eot_recovery: bool,

    /// Optional JSON dump (text, tokens, prompt, timings).
    #[arg(long)]
    json_out: Option<PathBuf>,

    /// Compute path: `fp32` is the exact official-transformers gold route (default).
    /// `int8` is a CT2-compatible path: per-output-row symmetric absmax
    /// (`scale = 127/amax`, ties-to-even `rint`) on encoder/decoder linears,
    /// conv1/conv2, and tied embeddings/logits, plus dynamic per-row activation
    /// quant (AVX-512 VNNI) and CT2-style `int8_float32` leftovers (float LN,
    /// dequant `acc*(1/sa)/sb`). It is allowed to drift from fp32 gold the same
    /// way CT2 INT8 does, and is not the product default.
    #[arg(long, default_value = "fp32")]
    compute_type: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let ctype = ComputeType::parse(&args.compute_type).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown --compute-type {} (expected fp32 or int8)",
            args.compute_type
        )
    })?;
    set_compute_type(ctype);
    eprintln!("compute-type: {}", ctype.label());
    let mode = TranscriptMode::parse(&args.mode).map_err(|e| anyhow::anyhow!("{e}"))?;
    let model_dir = resolve_model_dir(&args.model, args.model_dir.as_deref())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!("loading {}", model_dir.display());
    let t0 = Instant::now();
    let model = CrisperWhisper::load(&model_dir).map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!(
        "loaded {} (mel={}, enc={}, dec={}, vocab={}) in {:.2}s",
        model.weights.config.model_dir.display(),
        model.weights.config.feature_size,
        model.weights.config.encoder_layers,
        model.weights.config.decoder_layers,
        model.weights.config.vocab_size,
        t0.elapsed().as_secs_f64()
    );

    let opts = TranscribeOptions {
        language: args.language,
        mode,
        max_new_tokens: args.max_new_tokens,
        word_timestamps: args.word_timestamps,
        hallucination_mitigation: true,
        temperature_fallback: args.temperature_fallback,
        early_eot_recovery: args.early_eot_recovery,
        context: None,
    };
    let result = model
        .transcribe_file(&args.audio, &opts)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("transcribe {}", args.audio.display()))?;

    let total = result.encode_seconds + result.decode_seconds;
    let rtf = if total > 0.0 {
        result.audio_seconds / total
    } else {
        0.0
    };
    eprintln!(
        "audio {:.2}s  encode {:.2}s  decode {:.2}s  RTFx {:.2}  tokens {}",
        result.audio_seconds,
        result.encode_seconds,
        result.decode_seconds,
        rtf,
        result.tokens.len()
    );
    println!("{}", result.text);
    if let Some(path) = args.json_out {
        let mut rec = serde_json::to_value(&result)?;
        if let serde_json::Value::Object(ref mut map) = rec {
            map.insert("audio".into(), serde_json::Value::String(args.audio.display().to_string()));
            map.insert(
                "model_dir".into(),
                serde_json::Value::String(model_dir.display().to_string()),
            );
        }
        std::fs::write(&path, serde_json::to_string_pretty(&rec)?)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
