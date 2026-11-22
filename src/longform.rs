//! Default continuation longform (30 s window, 26 s stride, context_words=12,
//! drop_words=2, timestamp_aware_drop=True).

use crate::audio::SAMPLE_RATE;
use crate::word_timing::WordTimestamp;

#[derive(Debug, Clone)]
pub struct LongformConfig {
    pub chunk_duration: f64,
    pub stride: f64,
    pub context_words: usize,
    pub drop_words: usize,
    pub max_new_tokens: usize,
    pub timestamp_aware_drop: bool,
    pub temperature_fallback: bool,
    pub early_eot: crate::early_eot::EarlyEotConfig,
}

impl Default for LongformConfig {
    fn default() -> Self {
        Self {
            chunk_duration: 30.0,
            stride: 26.0,
            context_words: 12,
            drop_words: 2,
            max_new_tokens: 256,
            timestamp_aware_drop: true,
            temperature_fallback: true,
            early_eot: crate::early_eot::EarlyEotConfig::default(),
        }
    }
}

pub fn make_chunks(audio: &[f32], cfg: &LongformConfig) -> Vec<(usize, Vec<f32>)> {
    let chunk_samples = (cfg.chunk_duration * SAMPLE_RATE as f64).round() as usize;
    let stride_samples = (cfg.stride * SAMPLE_RATE as f64).round() as usize;
    let total = audio.len();
    if total <= chunk_samples {
        return vec![(0, audio.to_vec())];
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + chunk_samples).min(total);
        chunks.push((start, audio[start..end].to_vec()));
        if end >= total {
            break;
        }
        start += stride_samples;
    }
    chunks
}

/// Index up to which words are confirmed; `words[idx:]` are dropped.
pub fn overlap_drop_index(
    word_ts: &[WordTimestamp],
    stride_sec: f64,
    drop_words: usize,
    timestamp_aware_drop: bool,
    is_last: bool,
) -> usize {
    let n = word_ts.len();
    if is_last {
        return n;
    }
    let legacy = n.saturating_sub(drop_words);
    let has_placeable = word_ts.iter().any(|w| w.start.is_some());
    if !timestamp_aware_drop || !has_placeable {
        return legacy;
    }
    let mut first_overlap = n;
    for (idx, wt) in word_ts.iter().enumerate() {
        if let Some(s) = wt.start {
            if s >= stride_sec {
                first_overlap = idx;
                break;
            }
        }
    }
    legacy.max(first_overlap)
}

pub fn monotonize_words(words: &mut [WordTimestamp]) {
    for j in 1..words.len() {
        if let (Some(prev_end), Some(start), Some(end)) =
            (words[j - 1].end, words[j].start, words[j].end)
        {
            if start < prev_end {
                words[j].start = Some(prev_end);
                words[j].end = Some(prev_end.max(end));
            }
        }
    }
}

pub fn chunk_start_sec(chunk_index: usize, cfg: &LongformConfig) -> f64 {
    chunk_index as f64 * cfg.stride
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::word_timing::WordTimestamp;

    fn wt(word: &str, start: f64, end: f64) -> WordTimestamp {
        WordTimestamp { word: word.into(), start: Some(start), end: Some(end) }
    }

    #[test]
    fn overlap_drop_keeps_pre_stride() {
        let words = [wt("a", 24.0, 24.5), wt("b", 25.0, 25.4), wt("c", 26.1, 26.4), wt("d", 27.0, 27.3)];
        let keep = overlap_drop_index(&words, 26.0, 2, true, false);
        assert_eq!(keep, 2);
        assert_eq!(overlap_drop_index(&words, 26.0, 2, true, true), 4);
    }

    #[test]
    fn overlap_drop_keeps_all_if_no_overlap_words() {
        let words = [wt("a", 10.0, 11.0), wt("b", 20.0, 21.0), wt("c", 24.0, 25.0)];
        let keep = overlap_drop_index(&words, 26.0, 2, true, false);
        assert_eq!(keep, 3);
    }
}
