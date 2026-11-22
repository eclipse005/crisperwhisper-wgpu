//! Context-conditioned early-EOT recovery, matching official `longform.early_eot`.

#[derive(Debug, Clone)]
pub struct EarlyEotConfig {
    pub enabled: bool,
    pub stop_prob_threshold: f32,
    pub confident_prob: f32,
    pub tail_min_final: f64,
    pub tail_min_nonfinal: f64,
}

impl Default for EarlyEotConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            stop_prob_threshold: 0.7,
            confident_prob: 0.9,
            tail_min_final: 2.0,
            tail_min_nonfinal: 4.0,
        }
    }
}

pub fn content_len(ids: &[u32], eot: u32) -> usize {
    let mut n = ids.len();
    while n > 0 && ids[n - 1] == eot {
        n -= 1;
    }
    n
}

pub fn last_word_end(words: &[crate::word_timing::WordTimestamp]) -> Option<f64> {
    words.iter().filter_map(|w| w.end).max_by(|a, b| a.partial_cmp(b).unwrap())
}
