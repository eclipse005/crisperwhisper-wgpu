#!/usr/bin/env python3
"""Memory-safe official transformers fp32 continuation-longform gold.

`generate_with_repair_and_attention` / `output_attentions=True` on the full
model OOMs on a 15 GiB CPU host (encoder self-attentions).  This script uses
the official longform control flow, but:

1. `generate_with_repair` (tokens only, no attentions)
2. teacher-forced *decoder-only* cross-attention (same alignment heads,
   same row convention as `cross_attention_for_tokens`)
3. official `extract_word_timings`, `_overlap_drop_index`, `monotonize_words`

Temperature fallback and early-EOT default to off so the core greedy
gold stays deterministic.  Pass `--fallback` / `--early-eot` (or
`--full-defaults`) to dump official default longform, including the
sibling-mode confirmation decode and early-EOT recovery.  Sampling in
fallback will not bit-match a Rust RNG.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/CrisperWhisper")
os.environ.setdefault("HF_HOME", "/workspace/crisperwhisper-run/hf-cache")


def resolve_model(size: str) -> str:
    from huggingface_hub import snapshot_download

    if Path(size).is_dir() and (Path(size) / "config.json").exists():
        return str(Path(size))
    return snapshot_download(repo_id=f"nyralabs/CrisperWhisper2.0_{size}")


def decoder_only_cross_attn(engine, features, prompt_tokens, gen_ids, heads):
    """[T, F_enc] head-averaged cross-attn, row k predicts gen_ids[k]."""
    import torch

    if not gen_ids:
        return np.zeros((0, 0), dtype=np.float32)
    with torch.no_grad():
        enc = engine.model.model.encoder(input_features=features)
        hidden = enc.last_hidden_state
        del enc
        full = list(prompt_tokens) + list(gen_ids)
        dec = torch.tensor([full], device=engine.device, dtype=torch.long)
        out = engine.model.model.decoder(
            input_ids=dec,
            encoder_hidden_states=hidden,
            output_attentions=True,
            use_cache=False,
        )
        # decoder cross_attentions: tuple[L] of [1, H, L_full, F_enc]
        cross = out.cross_attentions
        T = len(gen_ids)
        start = len(prompt_tokens) - 1
        F_enc = cross[0].shape[-1]
        acc = torch.zeros((T, F_enc), dtype=torch.float32, device=engine.device)
        for layer, head in heads:
            acc += cross[layer][0, head, start:start + T].to(torch.float32)
        acc /= max(len(heads), 1)
        arr = acc.cpu().numpy().astype(np.float32)
        del out, cross, acc, hidden, dec
    gc.collect()
    return arr


def write_partial(path, **kw):
    partial = {
        "model": kw["model"],
        "audio": str(kw["audio"]),
        "language": kw["language"],
        "mode": kw["mode"],
        "audio_seconds": kw["used"],
        "elapsed_seconds": kw["elapsed"],
        "temperature_fallback": bool(kw["fallback"]),
        "early_eot": bool(kw["early_eot"]),
        "n_chunks": kw["n_chunks"],
        "next_chunk": kw["next_chunk"],
        "confirmed_words": kw["confirmed_words"],
        "chunks": kw["chunk_results"],
        "words": kw["global_words"],
        "partial": True,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(partial, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--mode", default="verbatim", choices=["verbatim", "intended"])
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = full file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fallback", action="store_true", help="enable temperature fallback")
    ap.add_argument("--early-eot", action="store_true", help="enable early-EOT recovery")
    ap.add_argument("--full-defaults", action="store_true", help="enable official default fallback + early-EOT")
    ap.add_argument("--resume", action="store_true", help="resume from {out}.partial.json if present")
    args = ap.parse_args()
    if args.full_defaults:
        args.fallback = True
        args.early_eot = True

    from crisperwhisper.audio import SAMPLE_RATE, load_audio
    from crisperwhisper.longform.base import LongformConfig, make_chunks
    from crisperwhisper.longform.continuation import _overlap_drop_index
    from crisperwhisper.prompt import PromptBuilder
    from crisperwhisper.transformers_engine import TransformersEngine
    from crisperwhisper.word_timing import extract_word_timings, monotonize_words

    model_path = resolve_model(args.model)
    print(f"loading transformers fp32 {model_path} on cpu...", flush=True)
    t0 = time.perf_counter()
    engine = TransformersEngine(model_path, device="cpu", compute_type="float32")
    heads = engine.enable_attention()
    print(f"loaded in {time.perf_counter() - t0:.1f}s  heads={len(heads)}", flush=True)

    audio = load_audio(args.audio)
    full = len(audio) / SAMPLE_RATE
    if args.seconds and args.seconds > 0:
        audio = audio[: int(args.seconds * SAMPLE_RATE)]
    used = len(audio) / SAMPLE_RATE

    cfg = LongformConfig()
    cfg.temperature_fallback = bool(args.fallback)
    cfg.early_eot.enabled = bool(args.early_eot)

    chunks = make_chunks(audio, cfg)
    n_chunks = len(chunks)
    print(f"longform-safe {args.audio} ({used:.2f}s of {full:.2f}s) chunks={n_chunks} mode={args.mode} fallback={cfg.temperature_fallback} early_eot={cfg.early_eot.enabled}", flush=True)

    builder = PromptBuilder(engine, language=args.language)
    confirmed_words = []
    chunk_results = []
    global_words = []
    t1 = time.perf_counter()
    start_i = 0
    elapsed_prev = 0.0
    partial_path = Path(str(args.out) + ".partial.json")
    if args.resume and partial_path.exists():
        partial = json.loads(partial_path.read_text())
        same = (
            partial.get("audio") == str(args.audio)
            and partial.get("model") == args.model
            and partial.get("mode") == args.mode
            and partial.get("language") == args.language
            and abs(float(partial.get("audio_seconds", 0)) - used) < 1e-3
            and bool(partial.get("temperature_fallback")) == bool(cfg.temperature_fallback)
            and bool(partial.get("early_eot")) == bool(cfg.early_eot.enabled)
            and int(partial.get("n_chunks", -1)) == n_chunks
        )
        if same:
            confirmed_words = list(partial.get("confirmed_words") or [])
            chunk_results = list(partial.get("chunks") or [])
            global_words = list(partial.get("words") or [])
            start_i = int(partial.get("next_chunk", len(chunk_results)))
            elapsed_prev = float(partial.get("elapsed_seconds") or 0.0)
            print(f"resuming from chunk {start_i+1}/{n_chunks}  confirmed_words={len(confirmed_words)}", flush=True)
        else:
            print("partial exists but metadata does not match; starting from scratch", flush=True)

    for i, chunk in enumerate(chunks):
        if i < start_i:
            continue
        is_last = i == n_chunks - 1
        start_sec = i * cfg.stride
        chunk_dur = len(chunk) / SAMPLE_RATE
        if i == 0:
            context_text = None
        else:
            ctx_words = confirmed_words[-cfg.context_words:] if confirmed_words else []
            context_text = " ".join(ctx_words) if ctx_words else None
        if args.mode == "verbatim":
            prompt_tokens = builder.verbatim(context=context_text)
        else:
            prompt_tokens = builder.intended(context=context_text)

        features, mel = engine.extract_features_with_mel(chunk)
        print(f"  chunk {i+1}/{n_chunks} start={start_sec:.1f}s dur={chunk_dur:.2f}s ctx={bool(context_text)} generate...", flush=True)
        gen_ids = engine.generate_with_repair(
            features, prompt_tokens,
            max_length=cfg.max_new_tokens,
            hallucination_mitigation=True,
        )
        if args.mode == "verbatim":
            sibling_prompt = builder.intended(context=context_text)
        else:
            sibling_prompt = builder.verbatim(context=context_text)
        if cfg.temperature_fallback:
            from crisperwhisper.fallback import decode_with_coverage_fallback
            print(f"    coverage fallback...", flush=True)
            gen_ids, _ignored_attn = decode_with_coverage_fallback(
                engine, features, mel, prompt_tokens, gen_ids, None,
                max_length=cfg.max_new_tokens, want_attention=False,
                enabled=True,
                ref_prompt_tokens=sibling_prompt,
            )
        # timing gold omits trailing EOT (official extract also tolerates it)
        eot = getattr(engine, "eot_id", 50257)
        if gen_ids and gen_ids[-1] == eot:
            gen_ids = gen_ids[:-1]
        print(f"    gen_ids={len(gen_ids)}  teacher-force attn...", flush=True)
        attention = decoder_only_cross_attn(engine, features, prompt_tokens, gen_ids, heads)
        word_ts_local = extract_word_timings(
            engine, gen_ids, attention, mel,
            audio_duration_s=chunk_dur,
            keep_unplaceable=True,
            language=builder.language,
        )
        if cfg.early_eot.enabled:
            from crisperwhisper.longform.early_eot import recover_early_eot
            recovered = recover_early_eot(
                engine, features, mel, prompt_tokens, list(gen_ids),
                word_ts=word_ts_local, is_last=is_last,
                max_length=cfg.max_new_tokens, suppress_tokens=None,
                config=cfg.early_eot,
            )
            if len(recovered) != len(gen_ids):
                gen_ids = recovered
                if gen_ids and gen_ids[-1] == eot:
                    gen_ids = gen_ids[:-1]
                print(f"    early-EOT recovered gen_ids={len(gen_ids)}  re-attn...", flush=True)
                attention = decoder_only_cross_attn(engine, features, prompt_tokens, gen_ids, heads)
                word_ts_local = extract_word_timings(
                    engine, gen_ids, attention, mel,
                    audio_duration_s=chunk_dur,
                    keep_unplaceable=True,
                    language=builder.language,
                )
        words = [wt.word for wt in word_ts_local]
        if not words:
            print("    EMPTY", flush=True)
            chunk_results.append({
                "chunk_idx": i, "start_sec": round(start_sec, 2),
                "end_sec": round(start_sec + chunk_dur, 2),
                "text": "", "context": context_text, "is_last": is_last,
                "n_tokens": len(gen_ids),
            })
        else:
            keep = _overlap_drop_index(
                word_ts_local,
                stride_sec=cfg.stride,
                drop_words=cfg.drop_words,
                timestamp_aware_drop=cfg.timestamp_aware_drop,
                is_last=is_last,
            )
            safe_words = words[:keep]
            safe_word_ts = word_ts_local[:keep]
            confirmed_words.extend(safe_words)
            for wt in safe_word_ts:
                if wt.start is None or wt.end is None:
                    continue
                global_words.append({
                    "word": wt.word,
                    "start": round(float(wt.start) + start_sec, 3),
                    "end": round(float(wt.end) + start_sec, 3),
                })
            chunk_results.append({
                "chunk_idx": i,
                "start_sec": round(start_sec, 2),
                "end_sec": round(start_sec + chunk_dur, 2),
                "text": " ".join(safe_words),
                "context": context_text,
                "is_last": is_last,
                "n_tokens": len(gen_ids),
                "n_words": len(words),
                "n_confirmed": len(safe_words),
            })
            print(f"    words={len(words)} confirmed={len(safe_words)}", flush=True)
        del features, mel, attention, word_ts_local
        gc.collect()
        write_partial(
            partial_path,
            model=args.model,
            audio=args.audio,
            language=args.language,
            mode=args.mode,
            used=used,
            elapsed=elapsed_prev + (time.perf_counter() - t1),
            fallback=cfg.temperature_fallback,
            early_eot=cfg.early_eot.enabled,
            n_chunks=n_chunks,
            next_chunk=i + 1,
            confirmed_words=confirmed_words,
            chunk_results=chunk_results,
            global_words=global_words,
        )

    # monotonize on a tiny shim
    from crisperwhisper.result import WordTimestamp
    objs = [WordTimestamp(word=w["word"], start=w["start"], end=w["end"]) for w in global_words]
    monotonize_words(objs)
    global_words = [{"word": w.word, "start": w.start, "end": w.end} for w in objs]
    text = " ".join(confirmed_words)
    elapsed = elapsed_prev + (time.perf_counter() - t1)
    out = {
        "model": args.model,
        "audio": str(args.audio),
        "language": args.language,
        "mode": args.mode,
        "audio_seconds": used,
        "elapsed_seconds": elapsed,
        "rtfx": used / elapsed if elapsed else None,
        "temperature_fallback": bool(cfg.temperature_fallback),
        "early_eot": bool(cfg.early_eot.enabled),
        "attention": "teacher_forced_decoder_only",
        "text": text,
        "n_chunks": n_chunks,
        "chunks": chunk_results,
        "words": global_words,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    if partial_path.exists():
        partial_path.unlink()
    print(f"wrote {args.out}  chunks={n_chunks} words={len(global_words)}  {elapsed:.1f}s", flush=True)
    print("text[:240]=", repr(text[:240]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
