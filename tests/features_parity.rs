//! Numerical parity vs official `crisperwhisper.features.FeatureExtractor`.
//! Fixtures are produced by `scripts/dump_features.py`.

use std::path::PathBuf;

use crisperwhisper_wgpu::features::{read_f32_le, FeatureExtractor};
use serde::Deserialize;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[derive(Debug, Deserialize)]
struct Meta {
    fixtures: Vec<Fix>,
}

#[derive(Debug, Deserialize)]
struct Fix {
    name: String,
    n_mels: usize,
    shape: Vec<usize>,
    wave_file: String,
    mel_file: String,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let s: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    s / a.len() as f32
}

#[test]
fn log_mel_matches_official_features_py() {
    let dir = fixtures_dir();
    let meta_path = dir.join("features_meta.json");
    assert!(
        meta_path.exists(),
        "missing {}; run scripts/dump_features.py first",
        meta_path.display()
    );
    let meta: Meta = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert!(!meta.fixtures.is_empty());

    let mut worst = 0.0f32;
    for fix in &meta.fixtures {
        let wave = read_f32_le(&dir.join(&fix.wave_file)).unwrap();
        let expected = read_f32_le(&dir.join(&fix.mel_file)).unwrap();
        assert_eq!(fix.shape.len(), 2);
        let (n_mels, n_frames) = (fix.shape[0], fix.shape[1]);
        assert_eq!(n_mels, fix.n_mels);
        assert_eq!(expected.len(), n_mels * n_frames);

        let fe = FeatureExtractor::new(n_mels);
        let got = fe.extract_chunk(&wave);
        assert_eq!(got.len(), expected.len(), "clip {}", fix.name);
        let max = max_abs_diff(&got, &expected);
        let mean = mean_abs_diff(&got, &expected);
        eprintln!(
            "{} n_mels={}  max_abs={:.3e}  mean_abs={:.3e}",
            fix.name, n_mels, max, mean
        );
        // rustfft vs NumPy rfft is not bit-identical; speech clips land around
        // 1e-5..2e-5.  That is tighter than HF WhisperFeatureExtractor vs
        // features.py on the same files (~2e-5..4e-5).  Mean error is ~1e-8.
        assert!(
            max < 5e-5,
            "{} n_mels={}: max_abs_diff {max} exceeds 5e-5 (mean {mean})",
            fix.name,
            n_mels
        );
        worst = worst.max(max);
    }
    eprintln!("worst max_abs_diff={worst:.3e}");
}
