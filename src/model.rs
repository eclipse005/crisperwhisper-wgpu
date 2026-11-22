use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;

use crate::audio::{load_audio, SAMPLE_RATE};
use crate::config::ModelConfig;
use crate::decoder::{decoder_forward, DecoderCache};
use crate::word_timing::{extract_word_timings, TimingConfig, WordTimestamp};
use crate::encoder::{encode, linear_logits_maybe_int8};
use crate::error::{Error, Result};
use crate::features::FeatureExtractor;
use crate::early_eot::{self, EarlyEotConfig};
use crate::fallback;
use crate::longform::{
    chunk_start_sec, make_chunks, monotonize_words, overlap_drop_index, LongformConfig,
};
use crate::loop_detection::{default_repair_thresholds, find_token_loop};
use crate::prompt::{build_prompt, strip_prompt_artifacts, TranscriptMode};
use crate::tokenizer::WhisperTokenizer;
use crate::weights::ModelWeights;

pub const OFFICIAL_SIZES: &[&str] = &["large", "turbo", "medium", "small"];

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    pub language: String,
    pub mode: TranscriptMode,
    pub max_new_tokens: usize,
    pub word_timestamps: bool,
    pub hallucination_mitigation: bool,
    /// Official coverage-gated temperature fallback (default on).
    pub temperature_fallback: bool,
    /// Official context-conditioned early-EOT recovery (default on, continuation only).
    pub early_eot_recovery: bool,
    /// Optional continuation context (already used by longform; unused for short clips).
    pub context: Option<String>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: String::new(),
            mode: TranscriptMode::Verbatim,
            max_new_tokens: 256,
            word_timestamps: true,
            hallucination_mitigation: true,
            temperature_fallback: true,
            early_eot_recovery: true,
            context: None,
        }
    }
}


#[derive(Clone, Debug, Default)]
struct DecodeOpts {
    collect_attn: bool,
    ban_first: Option<u32>,
    min_new_tokens: usize,
    temperature: Option<f32>,
    sample_seed: u64,
}

struct DecodeOutput {
    tokens: Vec<u32>,
    attn: Option<Vec<f32>>,
    stop_prob: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscribeResult {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: Vec<u32>,
    pub language: String,
    pub mode: String,
    pub audio_seconds: f64,
    pub encode_seconds: f64,
    pub decode_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WordTimestamp>>,
}

pub struct CrisperWhisper {
    pub weights: ModelWeights,
    pub tokenizer: WhisperTokenizer,
    pub features: FeatureExtractor,
}

impl CrisperWhisper {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let cfg = ModelConfig::load(model_dir)?;
        if cfg.activation_function != "gelu" {
            return Err(Error::Model(format!(
                "unsupported activation {}; only gelu (erf) is implemented",
                cfg.activation_function
            )));
        }
        if cfg.scale_embedding {
            return Err(Error::Model(
                "scale_embedding=true is not implemented (official 2.0 checkpoints are false)".into(),
            ));
        }
        let features = FeatureExtractor::with_params(
            cfg.feature_size,
            cfg.sampling_rate,
            cfg.hop_length,
            cfg.chunk_length,
            cfg.n_fft,
        );
        let tokenizer = WhisperTokenizer::load(model_dir, &cfg)?;
        let weights = ModelWeights::load(cfg)?;
        Ok(Self {
            weights,
            tokenizer,
            features,
        })
    }

    pub fn transcribe_file(&self, audio_path: &Path, opts: &TranscribeOptions) -> Result<TranscribeResult> {
        let audio = load_audio(audio_path)?;
        self.transcribe_samples(&audio, opts)
    }

    pub fn transcribe_samples(&self, audio: &[f32], opts: &TranscribeOptions) -> Result<TranscribeResult> {
        if opts.language.is_empty() {
            return Err(Error::msg(
                "language is required (no automatic language detection in phase 1)",
            ));
        }
        let audio_seconds = audio.len() as f64 / SAMPLE_RATE as f64;
        if audio_seconds > 30.0 {
            return self.transcribe_longform(audio, audio_seconds, opts);
        }
        let cfg = &self.weights.config;
        let mel = self.features.extract_chunk(audio);
        let n_frames = mel.len() / cfg.feature_size;

        let t0 = Instant::now();
        let enc = encode(
            &self.weights.encoder,
            &mel,
            cfg.feature_size,
            n_frames,
            cfg.layer_norm_eps,
        )?;
        let encode_seconds = t0.elapsed().as_secs_f64();

        let prompt = build_prompt(
            &self.tokenizer,
            cfg,
            opts.mode,
            &opts.language,
            opts.context.as_deref(),
        )?;

        let sibling = build_prompt(
            &self.tokenizer,
            cfg,
            sibling_mode(opts.mode),
            &opts.language,
            opts.context.as_deref(),
        )?;
        let t1 = Instant::now();
        let (tokens, attn) = self.generate_with_repair(
            &enc,
            cfg.max_source_positions,
            &prompt,
            opts.max_new_tokens,
            opts.word_timestamps,
            opts.hallucination_mitigation,
            None,
        )?;
        let (tokens, attn) = self.apply_coverage_fallback(
            &enc,
            cfg.max_source_positions,
            &mel,
            cfg.feature_size,
            n_frames,
            &prompt,
            Some(&sibling),
            tokens,
            attn,
            opts.max_new_tokens,
            opts.word_timestamps,
            opts.temperature_fallback,
        )?;
        let decode_seconds = t1.elapsed().as_secs_f64();

        // Official generate lists omit the trailing EOT for timing/text tokens.
        let mut content = tokens.clone();
        if content.last().copied() == Some(self.tokenizer.eot_id) {
            content.pop();
        }
        let raw = self.tokenizer.decode(&content, true)?;
        let text = strip_prompt_artifacts(&raw);

        let words = if opts.word_timestamps {
            if let Some(attn) = attn {
                let f_enc = cfg.max_source_positions;
                let n_frames = mel.len() / cfg.feature_size;
                Some(extract_word_timings(
                    &self.tokenizer,
                    &content,
                    &attn,
                    f_enc,
                    &mel,
                    cfg.feature_size,
                    n_frames,
                    Some(opts.language.as_str()),
                    Some(audio_seconds.min(cfg.chunk_length as f64)),
                    &TimingConfig::default(),
                )?)
            } else {
                Some(Vec::new())
            }
        } else {
            None
        };

        Ok(TranscribeResult {
            text,
            tokens,
            prompt_tokens: prompt,
            language: opts.language.clone(),
            mode: match opts.mode {
                TranscriptMode::Verbatim => "verbatim".into(),
                TranscriptMode::Intended => "intended".into(),
            },
            audio_seconds,
            encode_seconds,
            decode_seconds,
            words,
        })
    }

    fn decode_inner(
        &self,
        encoder_out: &[f32],
        enc_len: usize,
        prompt: &[u32],
        max_new_tokens: usize,
        opts: DecodeOpts,
    ) -> Result<DecodeOutput> {
        let cfg = &self.weights.config;
        let d = cfg.d_model;
        let vocab = cfg.vocab_size;
        let eot = self.tokenizer.eot_id;
        let mut cache = DecoderCache::new(cfg.decoder_layers);

        if prompt.is_empty() {
            return Err(Error::msg("decoder prompt is empty"));
        }

        let step = decoder_forward(
            &self.weights.decoder,
            prompt,
            encoder_out,
            enc_len,
            &mut cache,
            cfg.layer_norm_eps,
            opts.collect_attn,
        );
        let hidden = step.hidden;
        let mut logits = linear_logits_maybe_int8(
            &hidden[(prompt.len() - 1) * d..],
            &self.weights.decoder.embed_tokens,
            vocab,
            d,
            self.weights.decoder.embed_tokens_int8.as_ref(),
        );
        apply_suppress(&mut logits, &cfg.suppress_tokens);
        if let Some(ban) = opts.ban_first {
            apply_suppress(&mut logits, &[ban]);
        }

        let mut attn_rows: Vec<Vec<f32>> = Vec::new();
        if opts.collect_attn {
            if let Some(ref layers) = step.cross_last {
                attn_rows.push(average_alignment_heads(layers, &cfg.alignment_heads, enc_len));
            }
        }

        let mut generated = Vec::new();
        let mut stop_prob = None;
        let mut rng = opts.sample_seed;
        for _ in 0..max_new_tokens {
            if generated.len() < opts.min_new_tokens {
                apply_suppress(&mut logits, &[eot]);
            }
            let next = if let Some(temp) = opts.temperature {
                fallback::sample_from_logits(&logits, temp, &mut rng)
            } else {
                argmax(&logits)
            };
            if next == eot {
                stop_prob = Some(fallback::softmax_prob(&logits, eot));
                generated.push(next);
                break;
            }
            generated.push(next);
            let step = decoder_forward(
                &self.weights.decoder,
                &[next],
                encoder_out,
                enc_len,
                &mut cache,
                cfg.layer_norm_eps,
                opts.collect_attn,
            );
            if opts.collect_attn {
                if let Some(ref layers) = step.cross_last {
                    attn_rows.push(average_alignment_heads(layers, &cfg.alignment_heads, enc_len));
                }
            }
            logits = linear_logits_maybe_int8(
                &step.hidden,
                &self.weights.decoder.embed_tokens,
                vocab,
                d,
                self.weights.decoder.embed_tokens_int8.as_ref(),
            );
            apply_suppress(&mut logits, &cfg.suppress_tokens);
        }

        let attn = if opts.collect_attn && !attn_rows.is_empty() {
            let n_content = if generated.last().copied() == Some(eot) {
                generated.len() - 1
            } else {
                generated.len()
            };
            let rows = if attn_rows.len() > n_content {
                &attn_rows[..n_content]
            } else {
                &attn_rows[..]
            };
            let mut flat = Vec::with_capacity(rows.len() * enc_len);
            for r in rows {
                if r.len() >= enc_len {
                    flat.extend_from_slice(&r[..enc_len]);
                } else {
                    flat.extend_from_slice(r);
                    flat.resize(flat.len() + (enc_len - r.len()), 0.0);
                }
            }
            Some(flat)
        } else {
            None
        };
        Ok(DecodeOutput {
            tokens: generated,
            attn,
            stop_prob,
        })
    }

    pub fn greedy_decode(
        &self,
        encoder_out: &[f32],
        enc_len: usize,
        prompt: &[u32],
        max_new_tokens: usize,
        collect_attn: bool,
        ban_first: Option<u32>,
    ) -> Result<(Vec<u32>, Option<Vec<f32>>)> {
        let out = self.decode_inner(
            encoder_out,
            enc_len,
            prompt,
            max_new_tokens,
            DecodeOpts {
                collect_attn,
                ban_first,
                ..DecodeOpts::default()
            },
        )?;
        Ok((out.tokens, out.attn))
    }

    pub fn generate_with_repair(
        &self,
        encoder_out: &[f32],
        enc_len: usize,
        prompt: &[u32],
        max_new_tokens: usize,
        collect_attn: bool,
        enabled: bool,
        ban_first: Option<u32>,
    ) -> Result<(Vec<u32>, Option<Vec<f32>>)> {
        let (mut generated, mut attn) = self.greedy_decode(
            encoder_out, enc_len, prompt, max_new_tokens, collect_attn, ban_first,
        )?;
        if !enabled {
            return Ok((generated, attn));
        }
        let thresholds = default_repair_thresholds();
        const KEEP_REPS: usize = 1;
        const MAX_REPAIRS: usize = 3;
        for _attempt in 1..=MAX_REPAIRS {
            let hit = find_token_loop(&strip_eot(&generated, self.tokenizer.eot_id), &thresholds);
            let Some((loop_start, gram)) = hit else { break };
            let keep_end = loop_start + gram.len() * KEEP_REPS;
            let mut trimmed = generated[..keep_end.min(generated.len())].to_vec();
            if trimmed.last().copied() == Some(self.tokenizer.eot_id) {
                trimmed.pop();
            }
            let remaining = max_new_tokens.saturating_sub(trimmed.len());
            if remaining == 0 {
                let n = trimmed.len();
                generated = trimmed;
                if let Some(ref mut a) = attn {
                    a.truncate(n * enc_len);
                }
                break;
            }
            let mut prefix = prompt.to_vec();
            prefix.extend_from_slice(&trimmed);
            let (tail, tail_attn) = self.greedy_decode(
                encoder_out, enc_len, &prefix, remaining, collect_attn, Some(gram[0]),
            )?;
            generated = trimmed.clone();
            generated.extend(tail);
            if collect_attn {
                let f = enc_len;
                let mut kept = attn.unwrap_or_default();
                kept.truncate(trimmed.len() * f);
                if let Some(ta) = tail_attn {
                    kept.extend(ta);
                }
                attn = Some(kept);
            }
        }
        Ok((generated, attn))
    }


    fn word_count(&self, gen_ids: &[u32]) -> Result<usize> {
        let content = strip_eot(gen_ids, self.tokenizer.eot_id);
        let raw = self.tokenizer.decode(&content, true)?;
        Ok(strip_prompt_artifacts(&raw).split_whitespace().count())
    }

    fn eot_probability(
        &self,
        encoder_out: &[f32],
        enc_len: usize,
        prompt: &[u32],
        gen_ids: &[u32],
    ) -> Result<Option<f32>> {
        let content = strip_eot(gen_ids, self.tokenizer.eot_id);
        if content.is_empty() {
            return Ok(None);
        }
        let cfg = &self.weights.config;
        let d = cfg.d_model;
        let vocab = cfg.vocab_size;
        let mut ids = prompt.to_vec();
        ids.extend_from_slice(&content);
        let mut cache = DecoderCache::new(cfg.decoder_layers);
        let step = decoder_forward(
            &self.weights.decoder,
            &ids,
            encoder_out,
            enc_len,
            &mut cache,
            cfg.layer_norm_eps,
            false,
        );
        let mut logits = linear_logits_maybe_int8(
            &step.hidden[(ids.len() - 1) * d..],
            &self.weights.decoder.embed_tokens,
            vocab,
            d,
            self.weights.decoder.embed_tokens_int8.as_ref(),
        );
        apply_suppress(&mut logits, &cfg.suppress_tokens);
        Ok(Some(fallback::softmax_prob(&logits, self.tokenizer.eot_id)))
    }

    fn greedy_stops_and_decode(
        &self,
        encoder_out: &[f32],
        enc_len: usize,
        prompt: &[u32],
        max_new_tokens: usize,
        min_new_tokens: usize,
    ) -> Result<(Vec<u32>, Option<f32>)> {
        let out = self.decode_inner(
            encoder_out,
            enc_len,
            prompt,
            max_new_tokens,
            DecodeOpts {
                min_new_tokens,
                ..DecodeOpts::default()
            },
        )?;
        Ok((strip_eot(&out.tokens, self.tokenizer.eot_id), out.stop_prob))
    }

    fn cross_attention_for_tokens(
        &self,
        encoder_out: &[f32],
        enc_len: usize,
        prompt: &[u32],
        gen_ids: &[u32],
    ) -> Result<Vec<f32>> {
        let cfg = &self.weights.config;
        let content = strip_eot(gen_ids, self.tokenizer.eot_id);
        if content.is_empty() {
            return Ok(Vec::new());
        }
        let mut cache = DecoderCache::new(cfg.decoder_layers);
        let step = decoder_forward(
            &self.weights.decoder,
            prompt,
            encoder_out,
            enc_len,
            &mut cache,
            cfg.layer_norm_eps,
            true,
        );
        let mut attn_rows = Vec::new();
        if let Some(ref layers) = step.cross_last {
            attn_rows.push(average_alignment_heads(layers, &cfg.alignment_heads, enc_len));
        }
        for (i, &tok) in content.iter().enumerate() {
            if i + 1 == content.len() {
                break;
            }
            let step = decoder_forward(
                &self.weights.decoder,
                &[tok],
                encoder_out,
                enc_len,
                &mut cache,
                cfg.layer_norm_eps,
                true,
            );
            if let Some(ref layers) = step.cross_last {
                attn_rows.push(average_alignment_heads(layers, &cfg.alignment_heads, enc_len));
            }
        }
        let mut flat = Vec::with_capacity(attn_rows.len() * enc_len);
        for r in &attn_rows {
            if r.len() >= enc_len {
                flat.extend_from_slice(&r[..enc_len]);
            } else {
                flat.extend_from_slice(r);
                flat.resize(flat.len() + (enc_len - r.len()), 0.0);
            }
        }
        Ok(flat)
    }

    fn apply_coverage_fallback(
        &self,
        encoder_out: &[f32],
        enc_len: usize,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        prompt: &[u32],
        sibling_prompt: Option<&[u32]>,
        gen_ids: Vec<u32>,
        attention: Option<Vec<f32>>,
        max_new_tokens: usize,
        want_attention: bool,
        enabled: bool,
    ) -> Result<(Vec<u32>, Option<Vec<f32>>)> {
        if !enabled {
            return Ok((gen_ids, attention));
        }
        let base_n = self.word_count(&gen_ids)?;
        let speech = fallback::speech_active_seconds(mel, n_mels, n_frames);
        if !fallback::is_undercovered(base_n, speech) {
            return Ok((gen_ids, attention));
        }
        if let Some(sib) = sibling_prompt {
            let (ref_ids, _) = self.greedy_decode(
                encoder_out, enc_len, sib, max_new_tokens, false, None,
            )?;
            let ref_n = self.word_count(&ref_ids)?;
            if !fallback::discrepant(base_n, ref_n) {
                return Ok((gen_ids, attention));
            }
        }

        let mut best_ids = gen_ids.clone();
        let mut best_n = base_n;
        let mut best_is_sampled = false;
        let mut seed = 0u64;
        let mut done = false;
        for &temp in &fallback::FALLBACK_TEMPERATURES {
            if done {
                break;
            }
            for _ in 0..fallback::SAMPLES_PER_TEMP {
                let cur_seed = seed;
                seed += 1;
                let sampled = self.decode_inner(
                    encoder_out,
                    enc_len,
                    prompt,
                    max_new_tokens,
                    DecodeOpts {
                        temperature: Some(temp),
                        sample_seed: cur_seed,
                        ..DecodeOpts::default()
                    },
                )?;
                if fallback::sampled_has_loop(&strip_eot(&sampled.tokens, self.tokenizer.eot_id)) {
                    continue;
                }
                let n = self.word_count(&sampled.tokens)?;
                if n > best_n {
                    best_ids = sampled.tokens.clone();
                    best_n = n;
                    best_is_sampled = true;
                }
                if !fallback::is_undercovered(n, speech) {
                    best_ids = sampled.tokens;
                    best_n = n;
                    best_is_sampled = true;
                    done = true;
                    break;
                }
            }
        }
        if !best_is_sampled {
            return Ok((gen_ids, attention));
        }
        let new_attn = if want_attention {
            Some(self.cross_attention_for_tokens(encoder_out, enc_len, prompt, &best_ids)?)
        } else {
            attention
        };
        eprintln!(
            "temperature fallback recovered {best_n} words (was {base_n}, speech {speech:.2}s)"
        );
        Ok((best_ids, new_attn))
    }

    fn recover_early_eot(
        &self,
        encoder_out: &[f32],
        enc_len: usize,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        prompt: &[u32],
        gen_ids: Vec<u32>,
        word_ts: &[WordTimestamp],
        is_last: bool,
        max_new_tokens: usize,
        config: &EarlyEotConfig,
    ) -> Result<Vec<u32>> {
        if !config.enabled {
            return Ok(gen_ids);
        }
        let last_end = early_eot::last_word_end(word_ts);
        let Some(last_end) = last_end else {
            return Ok(gen_ids);
        };
        let tail_min = if is_last {
            config.tail_min_final
        } else {
            config.tail_min_nonfinal
        };
        let gap = fallback::speech_active_after(mel, n_mels, n_frames, last_end);
        if gap < tail_min {
            return Ok(gen_ids);
        }
        let stop_prob = self.eot_probability(encoder_out, enc_len, prompt, &gen_ids)?;
        if stop_prob.is_none() || stop_prob.unwrap() >= config.stop_prob_threshold {
            return Ok(gen_ids);
        }
        let content_len = early_eot::content_len(&gen_ids, self.tokenizer.eot_id);
        let (ext_ids, ext_prob) = self.greedy_stops_and_decode(
            encoder_out,
            enc_len,
            prompt,
            max_new_tokens,
            content_len + 1,
        )?;
        if ext_prob.unwrap_or(0.0) >= config.confident_prob && ext_ids.len() > content_len {
            eprintln!(
                "early-EOT recovery{}: stop_p={:.2} gap={:.1}s -> +{} tokens (new stop_p={:.2})",
                if is_last { " [final]" } else { "" },
                stop_prob.unwrap(),
                gap,
                ext_ids.len() - content_len,
                ext_prob.unwrap(),
            );
            let mut out = ext_ids;
            out.push(self.tokenizer.eot_id);
            return Ok(out);
        }
        Ok(gen_ids)
    }

    fn extract_chunk_words(
        &self,
        content: &[u32],
        attn: Option<&[f32]>,
        mel: &[f32],
        n_frames: usize,
        language: &str,
        chunk_dur: f64,
        keep_unplaceable: bool,
    ) -> Result<Vec<WordTimestamp>> {
        let Some(attn) = attn else {
            return Ok(Vec::new());
        };
        let cfg = &self.weights.config;
        let mut timing_cfg = TimingConfig::default();
        timing_cfg.keep_unplaceable = keep_unplaceable;
        extract_word_timings(
            &self.tokenizer,
            content,
            attn,
            cfg.max_source_positions,
            mel,
            cfg.feature_size,
            n_frames,
            Some(language),
            Some(chunk_dur),
            &timing_cfg,
        )
    }

    fn transcribe_longform(
        &self,
        audio: &[f32],
        audio_seconds: f64,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let cfg = &self.weights.config;
        let lf = LongformConfig {
            max_new_tokens: opts.max_new_tokens,
            temperature_fallback: opts.temperature_fallback,
            early_eot: EarlyEotConfig {
                enabled: opts.early_eot_recovery,
                ..EarlyEotConfig::default()
            },
            ..LongformConfig::default()
        };
        let chunks = make_chunks(audio, &lf);
        let n_chunks = chunks.len();
        let mut confirmed_words: Vec<String> = Vec::new();
        let mut global_words: Vec<WordTimestamp> = Vec::new();
        let mut encode_seconds = 0.0;
        let mut decode_seconds = 0.0;
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut first_prompt = Vec::new();

        for (i, (start_sample, chunk)) in chunks.iter().enumerate() {
            let _ = start_sample;
            let is_last = i + 1 == n_chunks;
            let start_sec = chunk_start_sec(i, &lf);
            let chunk_dur = chunk.len() as f64 / SAMPLE_RATE as f64;
            let context = if i == 0 {
                None
            } else {
                let ctx: Vec<&str> = confirmed_words
                    .iter()
                    .rev()
                    .take(lf.context_words)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .map(|s| s.as_str())
                    .collect();
                if ctx.is_empty() { None } else { Some(ctx.join(" ")) }
            };
            let prompt = build_prompt(
                &self.tokenizer, cfg, opts.mode, &opts.language, context.as_deref(),
            )?;
            if i == 0 {
                first_prompt = prompt.clone();
            }
            let mel = self.features.extract_chunk(chunk);
            let n_frames = mel.len() / cfg.feature_size;
            let t0 = Instant::now();
            let enc = encode(
                &self.weights.encoder, &mel, cfg.feature_size, n_frames, cfg.layer_norm_eps,
            )?;
            encode_seconds += t0.elapsed().as_secs_f64();
            let sibling = build_prompt(
                &self.tokenizer, cfg, sibling_mode(opts.mode), &opts.language, context.as_deref(),
            )?;
            let t1 = Instant::now();
            let (tokens, attn) = self.generate_with_repair(
                &enc,
                cfg.max_source_positions,
                &prompt,
                lf.max_new_tokens,
                true, // need attention for timestamp-aware drop
                opts.hallucination_mitigation,
                None,
            )?;
            let (mut tokens, mut attn) = self.apply_coverage_fallback(
                &enc,
                cfg.max_source_positions,
                &mel,
                cfg.feature_size,
                n_frames,
                &prompt,
                Some(&sibling),
                tokens,
                attn,
                lf.max_new_tokens,
                true,
                lf.temperature_fallback,
            )?;
            let mut content = strip_eot(&tokens, self.tokenizer.eot_id);
            let mut word_ts_local = self.extract_chunk_words(
                &content,
                attn.as_deref(),
                &mel,
                n_frames,
                &opts.language,
                chunk_dur,
                true,
            )?;
            if lf.early_eot.enabled {
                let recovered = self.recover_early_eot(
                    &enc,
                    cfg.max_source_positions,
                    &mel,
                    cfg.feature_size,
                    n_frames,
                    &prompt,
                    tokens.clone(),
                    &word_ts_local,
                    is_last,
                    lf.max_new_tokens,
                    &lf.early_eot,
                )?;
                if recovered.len() != tokens.len() {
                    tokens = recovered;
                    content = strip_eot(&tokens, self.tokenizer.eot_id);
                    attn = Some(self.cross_attention_for_tokens(
                        &enc, cfg.max_source_positions, &prompt, &tokens,
                    )?);
                    word_ts_local = self.extract_chunk_words(
                        &content,
                        attn.as_deref(),
                        &mel,
                        n_frames,
                        &opts.language,
                        chunk_dur,
                        true,
                    )?;
                }
            }
            decode_seconds += t1.elapsed().as_secs_f64();
            all_tokens.extend_from_slice(&content);
            if word_ts_local.is_empty() {
                continue;
            }
            let keep = overlap_drop_index(
                &word_ts_local,
                lf.stride,
                lf.drop_words,
                lf.timestamp_aware_drop,
                is_last,
            );
            for wt in word_ts_local.iter().take(keep) {
                confirmed_words.push(wt.word.clone());
                if let (Some(s), Some(e)) = (wt.start, wt.end) {
                    global_words.push(WordTimestamp {
                        word: wt.word.clone(),
                        start: Some(((s + start_sec) * 1000.0).round() / 1000.0),
                        end: Some(((e + start_sec) * 1000.0).round() / 1000.0),
                    });
                }
            }
        }
        monotonize_words(&mut global_words);
        let text = confirmed_words.join(" ");
        Ok(TranscribeResult {
            text,
            tokens: all_tokens,
            prompt_tokens: first_prompt,
            language: opts.language.clone(),
            mode: match opts.mode {
                TranscriptMode::Verbatim => "verbatim".into(),
                TranscriptMode::Intended => "intended".into(),
            },
            audio_seconds,
            encode_seconds,
            decode_seconds,
            words: if opts.word_timestamps { Some(global_words) } else { None },
        })
    }
}

fn strip_eot(ids: &[u32], eot: u32) -> Vec<u32> {
    let mut v = ids.to_vec();
    while v.last().copied() == Some(eot) {
        v.pop();
    }
    v
}

fn sibling_mode(mode: TranscriptMode) -> TranscriptMode {
    match mode {
        TranscriptMode::Verbatim => TranscriptMode::Intended,
        TranscriptMode::Intended => TranscriptMode::Verbatim,
    }
}


fn average_alignment_heads(layer_weights: &[Vec<f32>], heads: &[(usize, usize)], enc_len: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; enc_len];
    let mut n = 0usize;
    for &(layer, head) in heads {
        if layer >= layer_weights.len() {
            continue;
        }
        let w = &layer_weights[layer];
        let n_heads = if enc_len == 0 { 0 } else { w.len() / enc_len };
        if n_heads == 0 || head >= n_heads {
            continue;
        }
        let row = &w[head * enc_len..(head + 1) * enc_len];
        for i in 0..enc_len {
            acc[i] += row[i];
        }
        n += 1;
    }
    if n > 0 {
        let inv = 1.0 / n as f32;
        for a in acc.iter_mut() {
            *a *= inv;
        }
    }
    acc
}

fn apply_suppress(logits: &mut [f32], ids: &[u32]) {
    for &id in ids {
        let i = id as usize;
        if i < logits.len() {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Resolve a size shorthand (`turbo`) or a directory to a local checkpoint path.
pub fn resolve_model_dir(model: &str, model_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = model_dir {
        if dir.join("config.json").exists() && dir.join("model.safetensors").exists() {
            return Ok(dir.to_path_buf());
        }
        return Err(Error::Model(format!(
            "model-dir {} is missing config.json or model.safetensors",
            dir.display()
        )));
    }
    let p = PathBuf::from(model);
    if p.is_dir() && p.join("config.json").exists() {
        return Ok(p);
    }
    if !OFFICIAL_SIZES.contains(&model) {
        return Err(Error::Model(format!(
            "unknown model {model:?}; pass an official size ({}) or a checkpoint directory",
            OFFICIAL_SIZES.join(", ")
        )));
    }
    if let Some(found) = find_hf_snapshot(model) {
        return Ok(found);
    }
    Err(Error::Model(format!(
        "could not find CrisperWhisper2.0_{model}. Download it with scripts/download_models.py \
         or pass --model-dir. Weights are not in this repo (Nyra non-commercial research license)."
    )))
}

fn find_hf_snapshot(size: &str) -> Option<PathBuf> {
    let repo = format!("models--nyralabs--CrisperWhisper2.0_{size}");
    let mut roots = Vec::new();
    if let Ok(h) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(h).join("hub"));
    }
    if let Ok(h) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        roots.push(PathBuf::from(h));
    }
    roots.push(PathBuf::from("/workspace/crisperwhisper-run/hf-cache/hub"));
    roots.push(PathBuf::from("models"));
    roots.push(PathBuf::from("hf-cache/hub"));

    for root in roots {
        let base = if root.ends_with("hub") {
            root.join(&repo)
        } else {
            root.join(&repo)
        };
        let snaps = base.join("snapshots");
        if snaps.is_dir() {
            if let Ok(rd) = std::fs::read_dir(&snaps) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.join("config.json").exists() && p.join("model.safetensors").exists() {
                        return Some(p);
                    }
                }
            }
        }
        if base.join("config.json").exists() && base.join("model.safetensors").exists() {
            return Some(base);
        }
    }
    None
}
