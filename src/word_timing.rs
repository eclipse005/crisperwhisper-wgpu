//! Word-level timestamps from cross-attention + mel-energy blanks + Viterbi.
//! Port of official `crisperwhisper.word_timing` (defaults: sharpen=5,
//! blank_source=mel, blank_gamma=3, blank_penalty=3, split_gap_max_s=0.1,
//! 20 ms encoder frames).

use crate::error::Result;
use crate::tokenizer::WhisperTokenizer;
use serde::Serialize;

pub const FRAME_DURATION_S: f64 = 0.02;
pub const SPACE_TOKEN_ID: u32 = 220;
pub const SPACELESS_LANGUAGES: &[&str] = &["zh", "ja", "th", "lo", "my", "yue"];

const PROMPT_TAG_PREFIXES: &[&str] = &["[verbatim_", "[intended_"];
const PROMPT_MARKERS: &[&str] = &[
    "<ctx>", "<ectx>", "<htx>", "<ehtx>", "<vtx>", "<evtx>", "<sot>", "<eot>",
];

#[derive(Debug, Clone, Serialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TimingConfig {
    pub sharpen: f64,
    pub frame_duration: f64,
    pub blank_source: BlankSource,
    pub blank_gamma: f64,
    pub blank_penalty: f64,
    pub split_gap_max_s: f64,
    pub clip_to_audio: bool,
    pub keep_unplaceable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlankSource {
    Mel,
    Space,
    Auto,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            sharpen: 5.0,
            frame_duration: FRAME_DURATION_S,
            blank_source: BlankSource::Mel,
            blank_gamma: 3.0,
            blank_penalty: 3.0,
            split_gap_max_s: 0.1,
            clip_to_audio: false,
            keep_unplaceable: false,
        }
    }
}

fn is_special_piece(piece: &str) -> bool {
    if piece.starts_with("<|") && piece.ends_with("|>") {
        return true;
    }
    if PROMPT_TAG_PREFIXES.iter().any(|p| piece.starts_with(p)) {
        return true;
    }
    PROMPT_MARKERS.iter().any(|m| piece == *m)
}

fn is_space_token(token_id: u32, piece: &str) -> bool {
    token_id == SPACE_TOKEN_ID || piece == " "
}

fn starts_new_word(token_id: u32, piece: &str) -> bool {
    is_space_token(token_id, piece) || piece.starts_with(' ')
}

pub fn group_tokens_into_words(
    token_ids: &[u32],
    token_pieces: &[String],
) -> (Vec<Vec<usize>>, Vec<String>) {
    let mut word_token_indices: Vec<Vec<usize>> = Vec::new();
    let mut word_texts: Vec<String> = Vec::new();
    let mut cur_idx: Vec<usize> = Vec::new();
    let mut cur_text = String::new();

    let flush = |cur_idx: &mut Vec<usize>,
                 cur_text: &mut String,
                 word_token_indices: &mut Vec<Vec<usize>>,
                 word_texts: &mut Vec<String>| {
        if cur_text.trim().is_empty() || cur_idx.is_empty() {
            cur_idx.clear();
            cur_text.clear();
            return;
        }
        word_token_indices.push(cur_idx.clone());
        word_texts.push(cur_text.trim().to_string());
        cur_idx.clear();
        cur_text.clear();
    };

    for (i, (tid, piece)) in token_ids.iter().zip(token_pieces.iter()).enumerate() {
        if is_special_piece(piece) || is_space_token(*tid, piece) {
            flush(&mut cur_idx, &mut cur_text, &mut word_token_indices, &mut word_texts);
            continue;
        }
        if starts_new_word(*tid, piece) && !cur_text.is_empty() {
            flush(&mut cur_idx, &mut cur_text, &mut word_token_indices, &mut word_texts);
        }
        cur_idx.push(i);
        cur_text.push_str(piece);
    }
    flush(&mut cur_idx, &mut cur_text, &mut word_token_indices, &mut word_texts);
    (word_token_indices, word_texts)
}

pub fn group_tokens_into_words_unicode(
    tok: &WhisperTokenizer,
    token_ids: &[u32],
    token_pieces: &[String],
) -> Result<(Vec<Vec<usize>>, Vec<String>)> {
    let replacement = '\u{FFFD}';
    let kept: Vec<usize> = token_ids
        .iter()
        .zip(token_pieces.iter())
        .enumerate()
        .filter(|(_, (tid, piece))| !is_special_piece(piece) && !is_space_token(**tid, piece))
        .map(|(i, _)| i)
        .collect();
    let kept_ids: Vec<u32> = kept.iter().map(|&i| token_ids[i]).collect();
    let full_text = tok.decode(&kept_ids, true)?;

    let mut word_token_indices = Vec::new();
    let mut word_texts = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut offset = 0usize;

    let mut flush = |cur: &mut Vec<usize>, decoded: &str, offset: &mut usize| {
        let text = decoded.trim();
        if !text.is_empty() {
            word_token_indices.push(cur.clone());
            word_texts.push(text.to_string());
        }
        cur.clear();
        *offset += decoded.len();
    };

    for i in kept {
        cur.push(i);
        let ids: Vec<u32> = cur.iter().map(|&j| token_ids[j]).collect();
        let decoded = tok.decode(&ids, true)?;
        if let Some(pos) = decoded.find(replacement) {
            let genuine = offset + pos < full_text.len()
                && full_text[offset + pos..].starts_with(replacement);
            if genuine {
                flush(&mut cur, &decoded, &mut offset);
            }
        } else {
            flush(&mut cur, &decoded, &mut offset);
        }
    }
    if !cur.is_empty() {
        let ids: Vec<u32> = cur.iter().map(|&j| token_ids[j]).collect();
        let decoded = tok.decode(&ids, true)?;
        flush(&mut cur, &decoded, &mut offset);
    }
    Ok((word_token_indices, word_texts))
}

pub fn segment_tokens_into_words(
    tok: &WhisperTokenizer,
    token_ids: &[u32],
    token_pieces: &[String],
    language: Option<&str>,
) -> Result<(Vec<Vec<usize>>, Vec<String>)> {
    if language.is_some_and(|l| SPACELESS_LANGUAGES.contains(&l)) {
        group_tokens_into_words_unicode(tok, token_ids, token_pieces)
    } else {
        Ok(group_tokens_into_words(token_ids, token_pieces))
    }
}

fn token_logp_from_attention(attention: &[f32], t: usize, f: usize, sharpen: f64, eps: f64) -> Vec<f32> {
    assert_eq!(attention.len(), t * f);
    let mut out = vec![0.0f32; t * f];
    for i in 0..t {
        let row = &attention[i * f..(i + 1) * f];
        let mut acc = vec![0.0f64; f];
        let mut sum = 0.0f64;
        for j in 0..f {
            let mut v = row[j] as f64;
            if v < 0.0 {
                v = 0.0;
            }
            if (sharpen - 1.0).abs() > f64::EPSILON {
                v = v.powf(sharpen);
            }
            acc[j] = v;
            sum += v;
        }
        let s = sum.max(eps);
        for j in 0..f {
            out[i * f + j] = (acc[j] / s + eps).ln() as f32;
        }
    }
    out
}

fn resample_1d(x: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if x.len() == target_len {
        return x.to_vec();
    }
    if x.is_empty() {
        return vec![0.0; target_len];
    }
    if x.len() == 1 {
        return vec![x[0]; target_len];
    }
    let mut out = vec![0.0f32; target_len];
    let n = (x.len() - 1) as f32;
    let m = (target_len - 1) as f32;
    for i in 0..target_len {
        let pos = if m == 0.0 { 0.0 } else { i as f32 / m * n };
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(x.len() - 1);
        let w = pos - lo as f32;
        out[i] = x[lo] * (1.0 - w) + x[hi] * w;
    }
    out
}

fn percentile(xs: &[f32], p: f64) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.len() == 1 {
        return v[0];
    }
    let idx = (v.len() - 1) as f64 * (p / 100.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let w = (idx - lo as f64) as f32;
        v[lo] * (1.0 - w) + v[hi] * w
    }
}

pub fn blank_logp_from_mel_energy(
    mel: &[f32],
    n_mels: usize,
    n_mel_frames: usize,
    target_frames: usize,
    audio_frames: Option<usize>,
    eps: f64,
    gamma: f64,
    penalty: f64,
) -> Vec<f32> {
    assert_eq!(mel.len(), n_mels * n_mel_frames);
    let used_mel_frames = if let Some(af) = audio_frames {
        n_mel_frames.min(af * 2)
    } else {
        n_mel_frames
    };
    let mut energy = vec![0.0f32; used_mel_frames];
    for t in 0..used_mel_frames {
        let mut s = 0.0f32;
        for m in 0..n_mels {
            s += mel[m * n_mel_frames + t];
        }
        energy[t] = s / n_mels as f32;
    }
    let p10 = percentile(&energy, 10.0);
    let p90 = percentile(&energy, 90.0);
    let denom = (p90 - p10).max(1e-6);
    for e in energy.iter_mut() {
        *e = ((*e - p10) / denom).clamp(0.0, 1.0);
    }
    let energy_norm = resample_1d(&energy, target_frames);
    energy_norm
        .into_iter()
        .map(|en| {
            let mut blank_p = (1.0 - en as f64).clamp(1e-4, 1.0 - 1e-4);
            if (gamma - 1.0).abs() > f64::EPSILON {
                blank_p = blank_p.powf(gamma).clamp(1e-4, 1.0);
            }
            let mut logp = (blank_p + eps).ln();
            if penalty != 0.0 {
                logp -= penalty;
            }
            logp as f32
        })
        .collect()
}

fn blank_logp_from_space_attention(
    attention: &[f32],
    t: usize,
    f: usize,
    gen_ids: &[u32],
    target_frames: usize,
    sharpen: f64,
    eps: f64,
) -> Option<Vec<f32>> {
    let space_idx: Vec<usize> = gen_ids
        .iter()
        .enumerate()
        .filter(|(i, id)| **id == SPACE_TOKEN_ID && *i < t)
        .map(|(i, _)| i)
        .collect();
    if space_idx.is_empty() {
        return None;
    }
    let tf = target_frames;
    let n = space_idx.len();
    let mut sp = vec![0.0f64; n * tf];
    for (r, &i) in space_idx.iter().enumerate() {
        let mut sum = 0.0f64;
        for j in 0..tf {
            let mut v = attention[i * f + j] as f64;
            if v < 0.0 {
                v = 0.0;
            }
            if (sharpen - 1.0).abs() > f64::EPSILON {
                v = v.powf(sharpen);
            }
            sp[r * tf + j] = v;
            sum += v;
        }
        let s = sum.max(eps);
        for j in 0..tf {
            sp[r * tf + j] /= s;
        }
    }
    let mut blank = vec![0.0f64; tf];
    let mut peak = 0.0f64;
    for j in 0..tf {
        let mut m = 0.0f64;
        for r in 0..n {
            m = m.max(sp[r * tf + j]);
        }
        blank[j] = m;
        peak = peak.max(m);
    }
    if peak > 0.0 {
        for b in blank.iter_mut() {
            *b /= peak;
        }
    }
    Some(
        blank
            .into_iter()
            .map(|b| (b.clamp(1e-4, 1.0 - 1e-4) + eps).ln() as f32)
            .collect(),
    )
}

fn viterbi_align_tokens_with_blanks(
    token_logp: &[f32],
    t: usize,
    f: usize,
    blank_logp: &[f32],
    frame_duration: f64,
) -> Vec<(Option<f64>, Option<f64>)> {
    if t == 0 || f == 0 {
        return Vec::new();
    }
    assert_eq!(token_logp.len(), t * f);
    assert_eq!(blank_logp.len(), f);
    let s = 2 * t + 1;
    let neg_inf = -1e9f32;
    let mut emit = vec![0.0f32; s * f];
    for st in 0..s {
        for j in 0..f {
            emit[st * f + j] = if st % 2 == 0 {
                blank_logp[j]
            } else {
                token_logp[(st / 2) * f + j]
            };
        }
    }
    let mut dp = vec![neg_inf; s * f];
    let mut back = vec![0u8; s * f];
    dp[0] = emit[0];
    let mut adv_row = vec![neg_inf; s];
    for fr in 1..f {
        for st in 0..s {
            let stay = dp[st * f + (fr - 1)];
            let adv = if st == 0 { neg_inf } else { dp[(st - 1) * f + (fr - 1)] };
            adv_row[st] = adv;
            let take_adv = adv > stay;
            dp[st * f + fr] = if take_adv { adv } else { stay } + emit[st * f + fr];
            back[st * f + fr] = if take_adv { 1 } else { 0 };
        }
    }
    let mut end_state = 0usize;
    let mut best = f32::NEG_INFINITY;
    for st in 0..s {
        let score = dp[st * f + (f - 1)] + st as f32 * 1e-4;
        if score > best {
            best = score;
            end_state = st;
        }
    }
    let mut states = vec![0i32; f];
    let mut st = end_state as i32;
    for fr in (0..f).rev() {
        states[fr] = st;
        if fr == 0 {
            break;
        }
        if back[st as usize * f + fr] == 1 {
            st -= 1;
        }
    }
    let mut out = Vec::with_capacity(t);
    for tok in 0..t {
        let tok_state = (2 * tok + 1) as i32;
        let mut first = None;
        let mut last = None;
        for (fr, &sstate) in states.iter().enumerate() {
            if sstate == tok_state {
                if first.is_none() {
                    first = Some(fr);
                }
                last = Some(fr);
            }
        }
        out.push(match (first, last) {
            (Some(a), Some(b)) => (Some(a as f64 * frame_duration), Some(b as f64 * frame_duration)),
            _ => (None, None),
        });
    }
    out
}

fn logaddexp(a: f32, b: f32) -> f32 {
    let (max, min) = if a > b { (a, b) } else { (b, a) };
    if !max.is_finite() {
        return max;
    }
    max + (min - max).exp().ln_1p()
}

fn viterbi_align_words_with_blanks(
    token_logp: &[f32],
    t: usize,
    f: usize,
    blank_logp: &[f32],
    word_token_indices: &[Vec<usize>],
    frame_duration: f64,
) -> Vec<(Option<f64>, Option<f64>)> {
    let w = word_token_indices.len();
    if w == 0 {
        return Vec::new();
    }
    let mut word_logp = vec![-1e9f32; w * f];
    for (wi, idxs) in word_token_indices.iter().enumerate() {
        let valid: Vec<usize> = idxs.iter().copied().filter(|&i| i < t).collect();
        if valid.is_empty() {
            continue;
        }
        for j in 0..f {
            let mut acc = token_logp[valid[0] * f + j];
            for &i in &valid[1..] {
                acc = logaddexp(acc, token_logp[i * f + j]);
            }
            word_logp[wi * f + j] = acc;
        }
    }
    viterbi_align_tokens_with_blanks(&word_logp, w, f, blank_logp, frame_duration)
}

fn split_interword_gaps(words: &mut [WordTimestamp], max_gap_s: f64) {
    if max_gap_s <= 0.0 || words.len() < 2 {
        return;
    }
    for i in 0..words.len() - 1 {
        let (a_end, b_start) = match (words[i].end, words[i + 1].start) {
            (Some(e), Some(s)) => (e, s),
            _ => continue,
        };
        let gap = b_start - a_end;
        if gap > 0.0 && gap <= max_gap_s {
            let mid = a_end + gap / 2.0;
            words[i].end = Some(mid);
            words[i + 1].start = Some(mid);
        }
    }
}

/// `attention` is `[n_tokens, f_enc]` post-softmax, 1-to-1 with `gen_ids`.
/// `mel` is `[n_mels, n_mel_frames]`.
pub fn extract_word_timings(
    tok: &WhisperTokenizer,
    gen_ids: &[u32],
    attention: &[f32],
    f_enc: usize,
    mel: &[f32],
    n_mels: usize,
    n_mel_frames: usize,
    language: Option<&str>,
    audio_duration_s: Option<f64>,
    cfg: &TimingConfig,
) -> Result<Vec<WordTimestamp>> {
    let t = gen_ids.len();
    if t == 0 || attention.is_empty() || f_enc == 0 {
        return Ok(Vec::new());
    }
    assert_eq!(attention.len(), t * f_enc);

    let pieces = tok.decode_pieces(gen_ids)?;
    let (word_token_indices, _) = segment_tokens_into_words(tok, gen_ids, &pieces, language)?;
    if word_token_indices.is_empty() {
        return Ok(Vec::new());
    }
    let mut word_texts = Vec::with_capacity(word_token_indices.len());
    for idxs in &word_token_indices {
        let ids: Vec<u32> = idxs.iter().map(|&i| gen_ids[i]).collect();
        word_texts.push(tok.decode(&ids, true)?.trim().to_string());
    }

    let (actual_frames, mel_audio_frames) = if cfg.clip_to_audio {
        if let Some(dur) = audio_duration_s {
            let af = f_enc.min((dur / cfg.frame_duration).round().max(1.0) as usize);
            (af, Some(af))
        } else {
            (f_enc, None)
        }
    } else {
        (f_enc, None)
    };

    let mut attn = vec![0.0f32; t * actual_frames];
    for i in 0..t {
        attn[i * actual_frames..(i + 1) * actual_frames]
            .copy_from_slice(&attention[i * f_enc..i * f_enc + actual_frames]);
    }
    let tok_logp = token_logp_from_attention(&attn, t, actual_frames, cfg.sharpen, 1e-8);

    let mut blank_logp = None;
    if matches!(cfg.blank_source, BlankSource::Auto | BlankSource::Space) {
        blank_logp = blank_logp_from_space_attention(
            &attn,
            t,
            actual_frames,
            gen_ids,
            actual_frames,
            cfg.sharpen,
            1e-8,
        );
    }
    if blank_logp.is_none() {
        blank_logp = Some(blank_logp_from_mel_energy(
            mel,
            n_mels,
            n_mel_frames,
            actual_frames,
            mel_audio_frames,
            1e-6,
            cfg.blank_gamma,
            cfg.blank_penalty,
        ));
    }
    let blank_logp = blank_logp.unwrap();
    let word_timings = viterbi_align_words_with_blanks(
        &tok_logp,
        t,
        actual_frames,
        &blank_logp,
        &word_token_indices,
        cfg.frame_duration,
    );

    let mut placed = Vec::new();
    let mut placed_idx = Vec::new();
    for (idx, (word, (start, end))) in word_texts.iter().zip(word_timings.iter()).enumerate() {
        if start.is_none() || end.is_none() {
            continue;
        }
        placed.push(WordTimestamp {
            word: word.clone(),
            start: *start,
            end: *end,
        });
        placed_idx.push(idx);
    }
    split_interword_gaps(&mut placed, cfg.split_gap_max_s);

    if !cfg.keep_unplaceable {
        return Ok(placed);
    }
    let mut full: Vec<WordTimestamp> = word_texts
        .into_iter()
        .map(|word| WordTimestamp {
            word,
            start: None,
            end: None,
        })
        .collect();
    for (wt, idx) in placed.into_iter().zip(placed_idx) {
        full[idx] = wt;
    }
    Ok(full)
}
