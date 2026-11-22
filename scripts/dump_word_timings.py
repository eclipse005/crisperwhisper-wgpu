#!/usr/bin/env python3
"""Memory-safe official word-timestamp gold.

Does NOT call generate(output_attentions=True) (that OOMs on 15 GiB by
materializing encoder self-attentions).  Instead:

1. Greedy-generate text/tokens as usual (no attentions).
2. Teacher-force prompt+tokens through the decoder only, with
   output_attentions=True, and average official alignment heads.
3. Run crisperwhisper.word_timing.extract_word_timings.

This is the same timing algorithm and the same attention tensors the
transformers backend would use for those tokens, without the encoder-attn blowup.
"""
from __future__ import annotations

import argparse
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


def load_clip(path: Path, seconds: float):
    from crisperwhisper.audio import SAMPLE_RATE, load_audio

    audio = load_audio(path)
    full = len(audio) / SAMPLE_RATE
    if seconds and seconds > 0:
        audio = audio[: int(seconds * SAMPLE_RATE)]
    return audio, full, len(audio) / SAMPLE_RATE


def teacher_forced_cross_attn(engine, features, prompt, gen_ids, heads):
    """Return [T, F_enc] head-averaged cross-attn, 1-to-1 with gen_ids.

    Decoder-only: a full-model `output_attentions=True` materializes encoder
    self-attentions and OOMs on a 15 GiB host (especially large).
    """
    import torch

    if not gen_ids:
        return np.zeros((0, 0), dtype=np.float32)
    with torch.no_grad():
        enc = engine.model.model.encoder(input_features=features)
        hidden = enc.last_hidden_state
        # include gen_ids so query at prompt_len-1+k predicts gen_ids[k]
        full = list(prompt) + list(gen_ids)
        dec = torch.tensor([full], device=engine.device, dtype=torch.long)
        out = engine.model.model.decoder(
            input_ids=dec,
            encoder_hidden_states=hidden,
            output_attentions=True,
            use_cache=False,
        )
        cross = out.cross_attentions
        T = len(gen_ids)
        start = len(prompt) - 1
        F_enc = cross[0].shape[-1]
        acc = torch.zeros((T, F_enc), dtype=torch.float32, device=engine.device)
        for layer, head in heads:
            acc += cross[layer][0, head, start:start + T].to(torch.float32)
        acc /= max(len(heads), 1)
        return acc.cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--mode", default="verbatim")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from crisperwhisper.features import FeatureExtractor
    from crisperwhisper.prompt import PromptBuilder, strip_prompt_artifacts
    from crisperwhisper.transformers_engine import TransformersEngine
    from crisperwhisper.word_timing import extract_word_timings

    model_path = resolve_model(args.model)
    print(f"loading {model_path}", flush=True)
    engine = TransformersEngine(model_path, device="cpu", compute_type="float32")
    heads = engine.default_alignment_heads
    if not heads:
        raise SystemExit("no alignment_heads in generation_config")

    audio, full, used = load_clip(Path(args.audio), args.seconds)
    builder = PromptBuilder(engine, language=args.language)
    prompt = builder.verbatim() if args.mode == "verbatim" else builder.intended()
    feats, mel = engine.extract_features_with_mel(audio)
    # Official longform/engines pad to 30s then features.py padding=0.
    # Transformers feature extractor may differ slightly; also dump features.py mel.
    fe = FeatureExtractor(feature_size=engine.n_mels, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400)
    wav = audio
    chunk = 30 * 16000
    if len(wav) > chunk:
        wav = wav[:chunk]
    elif len(wav) < chunk:
        wav = np.pad(wav, (0, chunk - len(wav)))
    mel_fw = fe(wav, padding=0).astype(np.float32)

    print(f"generate {used:.2f}s...", flush=True)
    t0 = time.perf_counter()
    gen_ids = engine.generate(feats, [prompt], max_length=256, beam_size=1)[0]
    # drop trailing eot if present
    eot = engine.eot_id
    if eot is not None and gen_ids and gen_ids[-1] == eot:
        gen_ids = gen_ids[:-1]
    text = strip_prompt_artifacts(engine.decode_tokens(gen_ids, skip_special=True))
    print(f"generated {len(gen_ids)} tokens in {time.perf_counter()-t0:.1f}s  {text[:120]!r}", flush=True)

    print("teacher-forced cross-attention...", flush=True)
    t1 = time.perf_counter()
    attn = teacher_forced_cross_attn(engine, feats, prompt, gen_ids, heads)
    print(f"attn {attn.shape} in {time.perf_counter()-t1:.1f}s", flush=True)

    words = extract_word_timings(
        engine, gen_ids, attn, mel_fw,
        audio_duration_s=used, language=args.language,
    )
    rec = {
        "gold": "transformers_fp32_teacher_forced_attention",
        "note": "Decoder-only teacher-forced cross-attention + official extract_word_timings. Avoids generate(output_attentions=True) OOM. Mel is features.py (padding=0 after 30s pad), same as the Rust extractor.",
        "model": args.model,
        "model_path": model_path,
        "audio": str(args.audio),
        "audio_seconds_full": full,
        "audio_seconds_used": used,
        "language": args.language,
        "mode": args.mode,
        "prompt_tokens": [int(t) for t in prompt],
        "tokens": [int(t) for t in gen_ids],
        "text": text,
        "alignment_heads": heads,
        "attention_shape": list(attn.shape),
        "words": [{"word": w.word, "start": w.start, "end": w.end} for w in words],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # also save attention for rust unit tests? can be large (26*1500*4=156KB, audio1 48*1500*4=288KB). Fine.
    attn_path = out.with_suffix(".attn.f32")
    attn.astype(np.float32).tofile(attn_path)
    rec["attention_file"] = attn_path.name
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(f"wrote {out}  {len(words)} words", flush=True)
    for w in words[:8]:
        print(f"  {w.start:.3f}-{w.end:.3f}  {w.word}", flush=True)
    if len(words) > 8:
        print("  ...", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
