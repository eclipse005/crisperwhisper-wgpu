//! Coverage-gated temperature fallback, matching official `fallback.py`.
//!
//! Sampling uses a local xorshift RNG and will *not* bit-match PyTorch, so a
//! fallback-triggered chunk cannot be token-aligned with the official gold.
//! When the pre-filter does not fire, the greedy path is unchanged.

use crate::loop_detection::find_token_loop;

pub const FALLBACK_TEMPERATURES: [f32; 4] = [0.4, 0.6, 0.8, 1.0];
pub const SAMPLES_PER_TEMP: usize = 3;
pub const MEL_FRAME_S: f64 = 0.01;
pub const MIN_SPEECH_SEC: f64 = 5.0;
pub const MIN_WORDS_PER_SEC: f64 = 0.5;
pub const DISCREPANCY_RATIO: f64 = 0.5;
pub const MIN_REF_WORDS: usize = 8;

pub fn percentile(xs: &[f32], p: f64) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    let idx = ((p / 100.0) * (n - 1) as f64).clamp(0.0, (n - 1) as f64);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let t = (idx - lo as f64) as f32;
        v[lo] * (1.0 - t) + v[hi] * t
    }
}

pub fn speech_active_seconds(mel: &[f32], n_mels: usize, n_frames: usize) -> f64 {
    if n_mels == 0 || n_frames == 0 || mel.len() < n_mels * n_frames {
        return 0.0;
    }
    let mut energy = vec![0.0f32; n_frames];
    for t in 0..n_frames {
        let mut s = 0.0f32;
        for m in 0..n_mels {
            s += mel[m * n_frames + t];
        }
        energy[t] = s / n_mels as f32;
    }
    let floor = percentile(&energy, 10.0);
    let peak = percentile(&energy, 95.0);
    if peak - floor < 1e-3 {
        return 0.0;
    }
    let thr = floor + 0.2 * (peak - floor);
    let active = energy.iter().filter(|&&e| e > thr).count();
    active as f64 * MEL_FRAME_S
}

pub fn speech_active_after(mel: &[f32], n_mels: usize, n_frames: usize, start_s: f64) -> f64 {
    if n_mels == 0 || n_frames == 0 || mel.len() < n_mels * n_frames {
        return 0.0;
    }
    let mut energy = vec![0.0f32; n_frames];
    for t in 0..n_frames {
        let mut s = 0.0f32;
        for m in 0..n_mels {
            s += mel[m * n_frames + t];
        }
        energy[t] = s / n_mels as f32;
    }
    let floor = percentile(&energy, 10.0);
    let peak = percentile(&energy, 95.0);
    if peak - floor < 1e-3 {
        return 0.0;
    }
    let thr = floor + 0.2 * (peak - floor);
    let lo = ((start_s / MEL_FRAME_S).floor() as usize).min(energy.len());
    let active = energy[lo..].iter().filter(|&&e| e > thr).count();
    active as f64 * MEL_FRAME_S
}

pub fn is_undercovered(n_words: usize, speech_sec: f64) -> bool {
    if speech_sec < MIN_SPEECH_SEC {
        return false;
    }
    (n_words as f64) < MIN_WORDS_PER_SEC * speech_sec
}

pub fn discrepant(n_words: usize, ref_words: usize) -> bool {
    ref_words >= MIN_REF_WORDS && (n_words as f64) < DISCREPANCY_RATIO * ref_words as f64
}

/// Loop check used when rejecting sampled fallback draws (1..=5 grams, 8 reps).
pub fn sampled_has_loop(ids: &[u32]) -> bool {
    let th: Vec<(usize, usize)> = (1..=5).map(|n| (n, 8)).collect();
    find_token_loop(ids, &th).is_some()
}

pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E3779B97F4A7C15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Temperature softmax sample. `temp` must be > 0.
pub fn sample_from_logits(logits: &[f32], temp: f32, state: &mut u64) -> u32 {
    let t = temp.max(1e-6);
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max && v.is_finite() {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    let mut probs = vec![0.0f64; logits.len()];
    for (i, &v) in logits.iter().enumerate() {
        let p = if v.is_finite() {
            ((v - max) / t).exp() as f64
        } else {
            0.0
        };
        probs[i] = p;
        sum += p;
    }
    if sum <= 0.0 {
        return 0;
    }
    let u = (xorshift64(state) as f64) / (u64::MAX as f64);
    let mut acc = 0.0;
    let thresh = u * sum;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if acc >= thresh {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

pub fn softmax_prob(logits: &[f32], id: u32) -> f32 {
    let i = id as usize;
    if i >= logits.len() {
        return 0.0;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max && v.is_finite() {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    let mut target = 0.0f64;
    for (j, &v) in logits.iter().enumerate() {
        let e = if v.is_finite() {
            (v - max).exp() as f64
        } else {
            0.0
        };
        sum += e;
        if j == i {
            target = e;
        }
    }
    if sum <= 0.0 {
        0.0
    } else {
        (target / sum) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undercovered_requires_enough_speech() {
        assert!(!is_undercovered(0, 4.0));
        assert!(is_undercovered(0, 10.0));
        assert!(!is_undercovered(20, 10.0));
    }

    #[test]
    fn discrepancy_needs_min_ref() {
        assert!(!discrepant(1, 4));
        assert!(discrepant(3, 10));
        assert!(!discrepant(8, 10));
    }
}
