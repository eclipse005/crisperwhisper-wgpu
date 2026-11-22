#!/usr/bin/env python3
"""Dump official crisperwhisper.features.FeatureExtractor outputs for Rust tests.

Also records the difference versus HuggingFace WhisperFeatureExtractor, which
the transformers gold path uses.  The Rust extractor is required to match
features.py; any HF delta is documented in the sidecar JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CHUNK = 30 * 16_000

# Make the official package importable.
sys.path.insert(0, "/workspace/CrisperWhisper")
from crisperwhisper.features import FeatureExtractor  # noqa: E402
from crisperwhisper.audio import load_audio  # noqa: E402


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def hf_extract(audio: np.ndarray, n_mels: int) -> np.ndarray | None:
    try:
        from transformers import WhisperFeatureExtractor
    except Exception as e:
        print(f"  transformers extractor unavailable: {e}")
        return None
    fe = WhisperFeatureExtractor(
        feature_size=n_mels,
        sampling_rate=16_000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    # Pad to 30s like the official engines, then let HF pad/truncate to max_length.
    wav = audio
    if len(wav) > CHUNK:
        wav = wav[:CHUNK]
    elif len(wav) < CHUNK:
        wav = np.pad(wav, (0, CHUNK - len(wav)))
    out = fe(wav, sampling_rate=16_000, return_tensors=None, padding="max_length", truncation=True)
    feats = np.asarray(out["input_features"], dtype=np.float32)
    if feats.ndim == 3:
        feats = feats[0]
    return feats


def dump_one(name: str, audio: np.ndarray, n_mels: int, src: str) -> dict:
    fe = FeatureExtractor(
        feature_size=n_mels,
        sampling_rate=16_000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    wav = audio.astype(np.float32)
    if len(wav) > CHUNK:
        wav = wav[:CHUNK]
    elif len(wav) < CHUNK:
        wav = np.pad(wav, (0, CHUNK - len(wav)))
    # Official engines: pad to 30s then FeatureExtractor(..., padding=0)
    mel = fe(wav, padding=0).astype(np.float32)
    assert mel.shape == (n_mels, 3000), mel.shape
    stem = f"{name}_mel{n_mels}"
    write_f32(FIXTURES / f"{stem}.f32", mel)
    wave_name = f"{name}_wave.f32"
    write_f32(FIXTURES / wave_name, wav)
    hf = hf_extract(audio.astype(np.float32), n_mels)
    rec = {
        "name": name,
        "n_mels": n_mels,
        "shape": list(mel.shape),
        "source": src,
        "extractor": "crisperwhisper.features.FeatureExtractor",
        "padding": 0,
        "padded_to_samples": CHUNK,
        "wave_file": wave_name,
        "mel_file": f"{stem}.f32",
        "mel_min": float(mel.min()),
        "mel_max": float(mel.max()),
        "mel_mean": float(mel.mean()),
    }
    if hf is not None:
        if hf.shape != mel.shape:
            rec["hf_shape"] = list(hf.shape)
            rec["hf_note"] = "shape mismatch vs features.py"
        else:
            diff = np.abs(hf - mel)
            rec["hf_max_abs_diff"] = float(diff.max())
            rec["hf_mean_abs_diff"] = float(diff.mean())
            rec["hf_allclose_1e5"] = bool(np.allclose(hf, mel, rtol=0, atol=1e-5))
            rec["hf_allclose_1e4"] = bool(np.allclose(hf, mel, rtol=0, atol=1e-4))
    return rec


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    records = []
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(CHUNK).astype(np.float32) * 0.1)
    silence = np.zeros(CHUNK, dtype=np.float32)
    items = [
        ("noise", noise, "rng(0) * 0.1, 30s"),
        ("silence", silence, "zeros, 30s"),
    ]
    jfk = Path("/workspace/crisperwhisper-run/audio/jfk.flac")
    if jfk.exists():
        audio = load_audio(jfk)
        items.append(("jfk", audio, str(jfk)))
    jfk6 = Path("/workspace/crisperwhisper-run/audio/jfk_6s.wav")
    if jfk6.exists():
        audio = load_audio(jfk6)
        items.append(("jfk6s", audio, str(jfk6)))

    for name, audio, src in items:
        for n_mels in (80, 128):
            rec = dump_one(name, audio, n_mels, src)
            records.append(rec)
            hf = rec.get("hf_max_abs_diff")
            print(f"wrote {rec['mel_file']}  shape={rec['shape']}  hf_max_abs={hf}")

    meta = {"fixtures": records}
    (FIXTURES / "features_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", FIXTURES / "features_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
