# Test catalog

Public clips and transcripts for checking the Rust port. This folder is the accuracy catalog, not a speed suite.

The `wgpu` Cargo feature is still a placeholder and is **not implemented**. Fast CPU kernels (encoder GEMM, attention, softmax/GELU) are **AVX-512 only**. Hosts without AVX-512 (for example Intel Arrow Lake / Core Ultra 200) fall back to scalar paths and will look much slower; do not treat those RTFx numbers as a product baseline.

## Four result sets

Accuracy is always scored against **official transformers fp32**. CTranslate2 INT8 is a speed and memory reference only.

| Set | Directory | What it is | How to use it |
|---|---|---|---|
| 1. Official gold | `tests/gold/official_fp32/` | Official Python `transformers` fp32 (text + word timestamps). Teacher-forced decoder-only cross-attention + official `extract_word_timings` for the word files. | **This is the accuracy gold.** |
| 2. Rust fp32 (移植 fp32) | `tests/gold/rust_fp32/` | The port's default `--compute-type fp32` path. | Must match official gold: exact text, timestamp MAE 0.0 ms, 0 words over the 20 ms soft threshold. |
| 3. CT2 INT8 | `tests/gold/ct2_int8/` | Official CTranslate2 int8 (`backend=ct2`, `compute_type=int8`) with word timestamps. | Use only for speed and memory comparison. **Not an accuracy gold.** Do not chase matching CT2 INT8 text. |
| 4. Rust INT8 | `tests/gold/rust_int8/` | The port's optional `--compute-type int8` path (last encoder layer kept in official fp32). | Score accuracy against official fp32 gold (set 1), not against CT2 INT8. It should be no worse than CT2 when both are scored against that gold. |

`turbo_jfk_intended.json` in official gold and Rust fp32 is **text only** (no word timestamps). Pass = exact text.

Medium JFK official text punctuation differs slightly from turbo / small / large: medium has `you. Ask`, the others have `you, ask`. Compare each model to its own official gold file.

zh / ja / th language clips are short **segmentation-only** checks. They do not carry a word-timestamp quality claim.

## Public audio

| Clip | Path | Duration | Language | Source / license |
|---|---|---|---:|---|
| JFK inaugural (excerpt) | `tests/audio/jfk.flac` | 11.0 s | en | US public-domain speech (classic Whisper example). sha256 `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`, 1152693 bytes. |
| 10 official languages | `tests/audio/langs/{en,de,es,nl,fr,it,pl,pt,ru,sv}.wav` | 15.0 s | see code | LibriVox / Internet Archive, public domain or CC0. Details in `tests/audio/langs/SOURCES.md` and `manifest.json`. |
| zh / ja / th | `tests/audio/langs/{zh,ja,th}.wav` | 3.9 s / 1.4 s / 4.1 s | zh / ja / th | Wikimedia Commons / Lingua Libre (CC0 or CC BY 4.0). Segmentation only. See `SOURCES.md`. |

User evaluation clips (`tests/audio/user_langs/`, `tests/audio/user_zh_ja/`, audio1 / audio2 / audio3) are **not** in the public tree.

## Which JSON to use

Files share a stem `{model}_{clip}_{mode}.json` under each of the four directories. `jfk` uses `small`, `medium`, `turbo`, and `large`. Language clips use `turbo` only.

Examples:

- Official gold, turbo JFK verbatim: `tests/gold/official_fp32/turbo_jfk_verbatim.json`
- Rust fp32, same clip: `tests/gold/rust_fp32/turbo_jfk_verbatim.json`
- Official gold, turbo Spanish: `tests/gold/official_fp32/turbo_es_verbatim.json`
- CT2 INT8 / Rust INT8: `tests/gold/ct2_int8/…` and `tests/gold/rust_int8/…` with the same stems

Absolute lab paths (`/workspace/...`) have been rewritten to repo-relative paths such as `tests/audio/jfk.flac`. Model locations are stored as `nyralabs/CrisperWhisper2.0_{size}`. Published JSON does not include lab machine timings, PIDs, or attention dumps.

## How to transcribe and compare

Weights are not in this repo. Download them first (`scripts/download_models.py` or `--model-dir`).

```bash
# Rust default fp32 (must match official gold)
cargo run --release --bin transcribe -- \
  --model turbo \
  --audio tests/audio/jfk.flac \
  --language en \
  --mode verbatim \
  --compute-type fp32 \
  --json-out /tmp/rust_turbo_jfk_verbatim.json

python scripts/compare_transcripts.py \
  --gold tests/gold/official_fp32/turbo_jfk_verbatim.json \
  --rust /tmp/rust_turbo_jfk_verbatim.json

# Optional INT8: still compare against official fp32 gold, not against CT2 INT8
cargo run --release --bin transcribe -- \
  --model turbo \
  --audio tests/audio/jfk.flac \
  --language en \
  --mode verbatim \
  --compute-type int8 \
  --json-out /tmp/rust_turbo_jfk_int8.json

python scripts/compare_transcripts.py \
  --gold tests/gold/official_fp32/turbo_jfk_verbatim.json \
  --rust /tmp/rust_turbo_jfk_int8.json
```

### Pass criteria

- **Rust fp32 vs official gold:** `text_exact` is true, timestamp MAE is 0.0 ms, and `words_over_soft_ms` is 0 (20 ms soft threshold). `compare_transcripts.py` exits 0 only when both text and timestamps pass. For `turbo_jfk_intended` (no words), require exact text only.
- **Rust INT8 vs official gold:** score text and timestamps against official fp32. It is allowed to drift the way CT2 INT8 does; it should be no worse than CT2 when both are scored against that same gold. Do not treat a mismatch with `tests/gold/ct2_int8/` as a failure.
- **CT2 INT8:** keep for speed / memory comparisons. Do not use it as the accuracy target.

Unit tests (log-mel / Hann / mel-filter tensors) still load from `tests/fixtures/` (`hann400.f32`, `mel_filters_*.f32`, `jfk_*`, `jfk6s_*`, `silence_*`, `noise_*`, `features_meta.json`):

```bash
cargo test --release --lib --tests
```

MP3 bit-exact official decode still wants `libsndfile` on Unix. Without it, WAV goes through hound and other formats through Symphonia (same Xing/Info delay trim). The bundled public clips are WAV and FLAC, so a Windows or macOS build can load them without libsndfile.
