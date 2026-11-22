#!/usr/bin/env python3
"""Compare Rust CPU fp32 RTFx vs official CT2 int8 on the same machine/audio.

Timing excludes model load. Includes feature extract + encode + decode.
Word timestamps can be toggled; default ON to match the product path, with an
additional text-only pass so the decode-only cost is visible too.

This is a measurement tool, not a guarantee that Rust will beat CT2.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/CrisperWhisper")
os.environ.setdefault("HF_HOME", "/workspace/crisperwhisper-run/hf-cache")
os.environ.setdefault("CRISPERWHISPER_CACHE", "/workspace/crisperwhisper-run/cache")

ROOT = Path(__file__).resolve().parents[1]
RUST_BIN = ROOT / "target/release/transcribe"


def rust_run(model: str, audio: Path, language: str, word_ts: bool, compute_type: str = "fp32") -> dict:
    tag = "wts" if word_ts else "nots"
    out = Path("/tmp/cw-logs") / f"bench_rust_{compute_type}_{model}_{audio.stem}_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(RUST_BIN), "--model", model, "--audio", str(audio),
        "--language", language, "--mode", "verbatim",
        "--word-timestamps", "true" if word_ts else "false",
        "--compute-type", compute_type,
        "--json-out", str(out),
    ]
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/workspace/crisperwhisper-run/hf-cache")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"rust failed: {p.stderr[-800:]}")
    rec = json.loads(out.read_text())
    compute = rec["encode_seconds"] + rec["decode_seconds"]
    audio_s = rec["audio_seconds"]
    return {
        "engine": f"rust_{compute_type}",
        "compute_type": compute_type,
        "word_timestamps": word_ts,
        "audio_seconds": audio_s,
        "encode_seconds": rec["encode_seconds"],
        "decode_seconds": rec["decode_seconds"],
        "compute_seconds": compute,  # excludes load
        "wall_including_load": wall,
        "rtfx_exclude_load": audio_s / compute if compute else None,
        "text": rec.get("text"),
        "n_words": len(rec.get("words") or []),
        "stderr_tail": p.stderr.strip().splitlines()[-3:],
    }


def ct2_run(model_id: str, audio: Path, language: str, word_ts: bool, warmup: bool = True) -> dict:
    from crisperwhisper import CrisperWhisperModel
    from crisperwhisper.audio import load_audio, SAMPLE_RATE

    t_load = time.perf_counter()
    model = CrisperWhisperModel(
        model_id,
        backend="ct2",
        device="cpu",
        compute_type="int8",
        cache_dir="/workspace/crisperwhisper-run/cache",
    )
    load_s = time.perf_counter() - t_load
    samples = load_audio(audio)
    audio_s = len(samples) / SAMPLE_RATE
    if warmup:
        model.transcribe(
            samples, language=language, mode="verbatim",
            word_timestamps=word_ts,
        )
    t0 = time.perf_counter()
    result = model.transcribe(
        samples, language=language, mode="verbatim",
        word_timestamps=word_ts,
    )
    compute = time.perf_counter() - t0
    text = result.text if hasattr(result, "text") else str(result)
    n_words = len(getattr(result, "words", None) or [])
    return {
        "engine": "ct2_int8",
        "word_timestamps": word_ts,
        "audio_seconds": audio_s,
        "load_seconds": load_s,
        "compute_seconds": compute,
        "rtfx_exclude_load": audio_s / compute if compute else None,
        "text": text if isinstance(text, str) else None,
        "n_words": n_words,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--audio", action="append", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-ct2", action="store_true")
    ap.add_argument("--skip-rust", action="store_true")
    ap.add_argument("--compute-type", action="append", default=None,
                    help="Rust compute type (fp32, int8). Repeat to run several. Default: fp32.")
    args = ap.parse_args()
    compute_types = args.compute_type or ["fp32"]

    rows = []
    for audio in args.audio:
        audio = Path(audio)
        print(f"== {audio}", flush=True)
        rec = {"audio": str(audio)}
        if not args.skip_rust:
            for ctype in compute_types:
                for wts in (True, False):
                    print(f"  rust {ctype} word_ts={wts}", flush=True)
                    r = rust_run(args.model, audio, args.language, wts, compute_type=ctype)
                    print(f"    rust {ctype} compute={r['compute_seconds']:.3f}s RTFx={r['rtfx_exclude_load']:.2f}", flush=True)
                    key = f"rust_wts_{wts}" if ctype == "fp32" else f"rust_{ctype}_wts_{wts}"
                    rec[key] = r
        if not args.skip_ct2:
            for wts in (True, False):
                print(f"  ct2  word_ts={wts}", flush=True)
                r = ct2_run(args.model, audio, args.language, wts)
                print(f"    ct2  compute={r['compute_seconds']:.3f}s RTFx={r['rtfx_exclude_load']:.2f}", flush=True)
                rec[f"ct2_wts_{wts}"] = r
        rows.append(rec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
