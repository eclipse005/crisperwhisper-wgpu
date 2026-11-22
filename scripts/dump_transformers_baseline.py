#!/usr/bin/env python3
"""Freeze official transformers fp32 gold transcripts (text + token ids).

This is the accuracy gold standard. CT2 int8 is a speed baseline only.

Word timestamps (`--word-timestamps`) call `generate(output_attentions=True)`.
On CPU hosts with ~15 GiB RAM this OOMs on turbo/large because HuggingFace also
materializes encoder self-attentions. Default is text/tokens only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/CrisperWhisper")
os.environ.setdefault("HF_HOME", "/workspace/crisperwhisper-run/hf-cache")


def resolve_model(size: str) -> str:
    from huggingface_hub import snapshot_download

    if Path(size).is_dir() and (Path(size) / "config.json").exists():
        return str(Path(size))
    repo = f"nyralabs/CrisperWhisper2.0_{size}"
    return snapshot_download(repo_id=repo)


def load_clip(path: Path, seconds: float):
    from crisperwhisper.audio import SAMPLE_RATE, load_audio

    audio = load_audio(path)
    full = len(audio) / SAMPLE_RATE
    if seconds and seconds > 0:
        audio = audio[: int(seconds * SAMPLE_RATE)]
    return audio, full, len(audio) / SAMPLE_RATE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--audio", required=True, action="append", help="Repeatable.")
    ap.add_argument("--language", default="en")
    ap.add_argument("--mode", default="verbatim", choices=["verbatim", "intended"])
    ap.add_argument("--seconds", type=float, default=30.0, help="0 = full file")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Also request official word timestamps (high memory; may OOM on CPU).",
    )
    args = ap.parse_args()

    from crisperwhisper.prompt import PromptBuilder, strip_prompt_artifacts
    from crisperwhisper.transformers_engine import TransformersEngine

    model_path = resolve_model(args.model)
    print(f"loading transformers fp32 {model_path} on cpu...", flush=True)
    t0 = time.perf_counter()
    engine = TransformersEngine(model_path, device="cpu", compute_type="float32")
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    builder = PromptBuilder(engine, language=args.language)
    prompt = builder.verbatim() if args.mode == "verbatim" else builder.intended()
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in args.audio:
        audio_path = Path(audio_path)
        audio, full, used = load_clip(audio_path, args.seconds)
        print(f"transcribing {audio_path} ({used:.2f}s of {full:.2f}s) mode={args.mode}", flush=True)
        feats = engine.extract_features(audio)
        t1 = time.perf_counter()
        if args.word_timestamps:
            from crisperwhisper.model import CrisperWhisperModel

            # High-memory path; kept for when a bigger machine is available.
            model = CrisperWhisperModel(
                model_path, backend="transformers", compute_type="float32", device="cpu",
            )
            result = model.transcribe(
                audio,
                language=args.language,
                mode=args.mode,
                word_timestamps=True,
                timing_method="attention",
            )
            wall = time.perf_counter() - t1
            text = result.text
            tokens = None
            words = []
            for w in result.words or []:
                words.append({
                    "word": getattr(w, "word", None) or getattr(w, "text", None),
                    "start": getattr(w, "start", None),
                    "end": getattr(w, "end", None),
                })
        else:
            gen_ids = engine.generate(feats, [prompt], max_length=256, beam_size=1)[0]
            wall = time.perf_counter() - t1
            text = strip_prompt_artifacts(engine.decode_tokens(gen_ids, skip_special=True))
            tokens = [int(t) for t in gen_ids]
            words = None

        rec = {
            "gold": "transformers_fp32",
            "backend": "transformers",
            "compute_type": "float32",
            "word_timestamps": bool(args.word_timestamps),
            "timing_method": "attention",
            "model": args.model,
            "model_path": model_path,
            "audio": str(audio_path),
            "audio_seconds_full": full,
            "audio_seconds_used": used,
            "language": args.language,
            "mode": args.mode,
            "prompt_tokens": [int(t) for t in prompt],
            "tokens": tokens,
            "text": text,
            "words": words,
            "wall_seconds": wall,
            "rtfx": (used / wall) if wall > 0 else None,
        }
        out = out_dir / f"gold_{args.model}_{audio_path.stem}_{args.mode}.json"
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f"wrote {out}  {used:.2f}s in {wall:.1f}s  tokens={len(tokens or [])}  {text[:140]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
