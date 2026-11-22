//! Log-mel feature extraction matching official `crisperwhisper.features.FeatureExtractor`
//! (vendored faster-whisper 1.2.1 implementation).
//!
//! Pipeline:
//! - optional trailing pad (`padding`, default 160 in the Python class; the
//!   official engines pass `padding=0` after already padding the waveform to 30s)
//! - reflect-pad `n_fft/2` on each side (STFT `center=True`)
//! - periodic Hann window `np.hanning(n_fft + 1)[:-1]`
//! - unnormalized real FFT, hop 160, n_fft 400
//! - drop the last time frame (`stft[..., :-1]`)
//! - power spectrogram, Slaney mel filterbank, `log10(clip(., 1e-10))`
//! - `max(log_spec, log_spec.max() - 8)`, then `(x + 4) / 4`

use rustfft::{num_complex::Complex, FftPlanner};

use crate::error::Result;

pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_HOP_LENGTH: usize = 160;
pub const DEFAULT_N_FFT: usize = 400;
pub const DEFAULT_CHUNK_LENGTH: usize = 30;

#[derive(Clone)]
pub struct FeatureExtractor {
    pub n_fft: usize,
    pub hop_length: usize,
    pub sampling_rate: u32,
    pub n_mels: usize,
    pub n_samples: usize,
    pub nb_max_frames: usize,
    mel_filters: Vec<f32>, // [n_mels, n_fft/2+1]
    window: Vec<f32>,
}

impl FeatureExtractor {
    pub fn new(n_mels: usize) -> Self {
        Self::with_params(
            n_mels,
            DEFAULT_SAMPLE_RATE,
            DEFAULT_HOP_LENGTH,
            DEFAULT_CHUNK_LENGTH,
            DEFAULT_N_FFT,
        )
    }

    pub fn with_params(
        n_mels: usize,
        sampling_rate: u32,
        hop_length: usize,
        chunk_length: usize,
        n_fft: usize,
    ) -> Self {
        let n_samples = chunk_length * sampling_rate as usize;
        let nb_max_frames = n_samples / hop_length;
        let n_freqs = n_fft / 2 + 1;
        let mel_filters = mel_filters(sampling_rate, n_fft, n_mels);
        assert_eq!(mel_filters.len(), n_mels * n_freqs);
        let window = hann_periodic(n_fft);
        Self {
            n_fft,
            hop_length,
            sampling_rate,
            n_mels,
            n_samples,
            nb_max_frames,
            mel_filters,
            window,
        }
    }

    /// Compute log-mel of shape `[n_mels, n_frames]`.
    ///
    /// `padding` is trailing zeros appended *before* the centered STFT, matching
    /// the Python `__call__(waveform, padding=...)` argument.
    pub fn extract(&self, waveform: &[f32], padding: usize) -> Vec<f32> {
        let mut wave = waveform.to_vec();
        if padding > 0 {
            wave.resize(wave.len() + padding, 0.0);
        }
        let (spec, n_frames, n_freqs) = stft_power(&wave, self.n_fft, self.hop_length, &self.window);
        // Drop last time frame: `np.abs(stft[..., :-1]) ** 2`
        let used_frames = n_frames.saturating_sub(1);
        let mut magnitudes = vec![0.0f32; n_freqs * used_frames];
        for f in 0..n_freqs {
            for t in 0..used_frames {
                magnitudes[f * used_frames + t] = spec[f * n_frames + t];
            }
        }
        // mel_spec = filters[n_mels, n_freqs] @ magnitudes[n_freqs, used_frames]
        let mut mel = vec![0.0f32; self.n_mels * used_frames];
        for m in 0..self.n_mels {
            let filt = &self.mel_filters[m * n_freqs..(m + 1) * n_freqs];
            for t in 0..used_frames {
                let mut acc = 0.0f32;
                for f in 0..n_freqs {
                    acc += filt[f] * magnitudes[f * used_frames + t];
                }
                mel[m * used_frames + t] = acc;
            }
        }
        log_mel_normalize(&mut mel);
        mel
    }

    /// Pad/truncate to a 30s window, then extract with `padding=0`, yielding
    /// `[n_mels, 3000]` — the layout the Whisper encoder requires.
    pub fn extract_chunk(&self, waveform: &[f32]) -> Vec<f32> {
        if waveform.len() > self.n_samples {
            self.extract(&waveform[..self.n_samples], 0)
        } else if waveform.len() == self.n_samples {
            self.extract(waveform, 0)
        } else {
            let mut padded = waveform.to_vec();
            padded.resize(self.n_samples, 0.0);
            self.extract(&padded, 0)
        }
    }

    pub fn n_frames_for(&self, n_mel_elems: usize) -> usize {
        n_mel_elems / self.n_mels
    }
}

fn log_mel_normalize(mel: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        *v = libm::log10f((*v).max(1e-10));
        if *v > max {
            max = *v;
        }
    }
    let floor = max - 8.0;
    for v in mel.iter_mut() {
        if *v < floor {
            *v = floor;
        }
        *v = (*v + 4.0) / 4.0;
    }
}

/// `np.hanning(n_fft + 1)[:-1].astype("float32")` — computed in f64, then cast.
pub fn hann_periodic(n_fft: usize) -> Vec<f32> {
    let m = n_fft + 1;
    let mut w = vec![0.0f32; n_fft];
    if m > 1 {
        let denom = (m - 1) as f64;
        for n in 0..n_fft {
            let s = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / denom).cos();
            w[n] = s as f32;
        }
    }
    w
}

/// Official `FeatureExtractor.get_mel_filters` (Slaney-style, max_mel fixed).
/// Computed in f64 to match NumPy, then stored as float32.
pub fn mel_filters(sr: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    // fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0/sr)
    let mut fftfreqs = vec![0.0f64; n_freqs];
    for i in 0..n_freqs {
        fftfreqs[i] = i as f64 * (sr as f64) / n_fft as f64;
    }

    let min_mel = 0.0f64;
    let max_mel = 45.245640471924965_f64;
    let n = n_mels + 2;
    let mut mels = vec![0.0f64; n];
    for i in 0..n {
        mels[i] = min_mel + (max_mel - min_mel) * i as f64 / (n - 1) as f64;
    }

    let f_min = 0.0f64;
    let f_sp = 200.0 / 3.0;
    let mut freqs: Vec<f64> = mels.iter().map(|m| f_min + f_sp * m).collect();

    let min_log_hz = 1000.0f64;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    for i in 0..n {
        if mels[i] >= min_log_mel {
            freqs[i] = min_log_hz * (logstep * (mels[i] - min_log_mel)).exp();
        }
    }

    let mut fdiff = vec![0.0f64; n - 1];
    for i in 0..n - 1 {
        fdiff[i] = freqs[i + 1] - freqs[i];
    }

    let mut weights_f64 = vec![0.0f64; n_mels * n_freqs];
    for i in 0..n_mels {
        let fd_lo = fdiff[i];
        let fd_hi = fdiff[i + 1];
        for j in 0..n_freqs {
            let lower = -(freqs[i] - fftfreqs[j]) / fd_lo;
            let upper = (freqs[i + 2] - fftfreqs[j]) / fd_hi;
            weights_f64[i * n_freqs + j] = lower.min(upper).max(0.0);
        }
    }

    // enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])
    for i in 0..n_mels {
        let enorm = 2.0 / (freqs[i + 2] - freqs[i]);
        for j in 0..n_freqs {
            weights_f64[i * n_freqs + j] *= enorm;
        }
    }
    weights_f64.into_iter().map(|v| v as f32).collect()
}

/// Reflect pad a 1-D signal by `pad` samples on each side (`np.pad(..., mode="reflect")`).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 {
        return x.to_vec();
    }
    // numpy reflect: the edge is not repeated. For length L, the mirrored
    // index is folded over the open interval (0, L-1).
    let n = x.len();
    assert!(n >= 2, "reflect pad needs at least 2 samples, got {n}");
    let mut out = vec![0.0f32; n + 2 * pad];
    for i in 0..out.len() {
        out[i] = x[reflect_index(i as isize - pad as isize, n)];
    }
    out
}

fn reflect_index(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    // Period of the un-repeated-edge reflection is 2*(n-1).
    let period = 2 * (n - 1);
    i = i.rem_euclid(period);
    if i >= n {
        (period - i) as usize
    } else {
        i as usize
    }
}

/// Power spectrogram `[n_freqs, n_frames]` (does **not** drop the last frame).
fn stft_power(wave: &[f32], n_fft: usize, hop: usize, window: &[f32]) -> (Vec<f32>, usize, usize) {
    let padded = reflect_pad(wave, n_fft / 2);
    let length = padded.len();
    let n_frames = 1 + (length - n_fft) / hop;
    let n_freqs = n_fft / 2 + 1;

    let mut power = vec![0.0f32; n_freqs * n_frames];

    // Sequential FFT plans share scratch; parallelize by giving each worker its own plan.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = (n_frames + n_threads - 1) / n_threads;
    let ranges: Vec<(usize, usize)> = (0..n_frames)
        .step_by(chunk.max(1))
        .map(|s| (s, (s + chunk).min(n_frames)))
        .collect();

    // Use a mutex-free approach: compute each range into a local buffer then copy.
    // Simpler and correct: just do it sequentially first if n_frames is small; 3001 is fine
    // to parallelize with per-thread planners.
    use rayon::prelude::*;
    let parts: Vec<(usize, Vec<f32>)> = ranges
        .into_par_iter()
        .map(|(start, end)| {
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(n_fft);
            let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
            let mut local = vec![0.0f32; n_freqs * (end - start)];
            for (li, t) in (start..end).enumerate() {
                let off = t * hop;
                for i in 0..n_fft {
                    buf[i] = Complex::new(padded[off + i] * window[i], 0.0);
                }
                fft.process(&mut buf);
                for f in 0..n_freqs {
                    let c = buf[f];
                    local[f * (end - start) + li] = c.re * c.re + c.im * c.im;
                }
            }
            (start, local)
        })
        .collect();

    for (start, local) in parts {
        let end = start + local.len() / n_freqs;
        let span = end - start;
        for f in 0..n_freqs {
            for li in 0..span {
                power[f * n_frames + (start + li)] = local[f * span + li];
            }
        }
    }
    (power, n_frames, n_freqs)
}

pub fn write_f32_le(path: &std::path::Path, data: &[f32]) -> Result<()> {
    use std::io::Write;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::File::create(path)?.write_all(&bytes)?;
    Ok(())
}

pub fn read_f32_le(path: &std::path::Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(crate::error::Error::msg(format!(
            "f32 file {} length {} is not a multiple of 4",
            path.display(),
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_matches_numpy_formula() {
        let w = hann_periodic(400);
        assert_eq!(w.len(), 400);
        assert!((w[0] - 0.0).abs() < 1e-7);
        // last sample of np.hanning(401)[:-1] is not zero
        assert!(w[399] > 0.0);
    }

    #[test]
    fn extract_chunk_shape() {
        let fe = FeatureExtractor::new(128);
        let audio = vec![0.0f32; 16_000]; // 1s, will be padded
        let mel = fe.extract_chunk(&audio);
        assert_eq!(mel.len(), 128 * 3000);
    }

    #[test]
    fn hann_and_filters_match_numpy() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let hann = read_f32_le(&dir.join("hann400.f32")).unwrap();
        let got = hann_periodic(400);
        let mut max = 0.0f32;
        for (a, b) in got.iter().zip(hann.iter()) {
            max = max.max((a - b).abs());
        }
        assert!(max < 1e-7, "hann max_abs {max}");

        for n_mels in [80usize, 128] {
            let exp = read_f32_le(&dir.join(format!("mel_filters_{n_mels}.f32"))).unwrap();
            let got = mel_filters(16_000, 400, n_mels);
            assert_eq!(got.len(), exp.len());
            let mut max = 0.0f32;
            let mut arg = 0usize;
            for (i, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
                let d = (a - b).abs();
                if d > max {
                    max = d;
                    arg = i;
                }
            }
            eprintln!("mel_filters {n_mels}: max_abs={max:.3e} at {arg}");
            assert!(max < 1e-7, "mel_filters {n_mels} max_abs {max}");
        }
    }
}
