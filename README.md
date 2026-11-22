# crisperwhisper-wgpu

Rust port of [CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper). The first milestone is a **CPU** implementation that matches the official Python `transformers` fp32 path on text and word timestamps. After the CPU port is aligned, a **wgpu** GPU backend will be added and verified on a Kaggle T4 (Linux + CUDA). The `wgpu` Cargo feature is reserved for that stage and is **not implemented yet** — do not expect `--features wgpu` to do anything in this phase.

This repository contains **code only**. Official model weights are not included and must be downloaded yourself from Hugging Face (`nyralabs/CrisperWhisper2.0_{large,turbo,medium,small}`). Those checkpoints are released by Nyra under a **non-commercial research license** — read and follow the license on the model card before downloading or using them. This project's MIT license covers the Rust code in this repo, not the weights.

## Status

**Current focus: finish the Rust CPU port and align it.** wgpu comes after that.

Already working on CPU (fp32 math, official BF16 safetensors):

- Models: `large`, `turbo`, `medium`, `small` (not `*_pro`, not deprecated v1) — all four downloaded and checked on English JFK
- Greedy decode, `verbatim` / `intended` prompts, required language (no auto language detect)
- Log-mel features numerically matched to official `crisperwhisper.features.FeatureExtractor` (worst max abs error ~1.7e-5 on speech)
- Word timestamps (attention + mel-energy blank + Viterbi, 20 ms frames) and special tags such as `[UM]` `[UH]` `[scream]`
- Default-style continuation longform (30 s window, 26 s stride, 12 context words, timestamp-aware drop, hallucination repair, temperature fallback, early-EOT recovery)
- MP3/lossy decode aligned with official `load_audio` (libsndfile first, then soxr HQ; Xing/Info encoder-delay trim on the Symphonia fallback)
- Legal 15 s clips for the official 10 languages plus unspaced zh/ja/th (segmentation only) in `tests/audio/langs/`

English alignment against official **transformers fp32** gold (exact text, word-timestamp MAE 0.0 ms, 0 words over the 20 ms soft threshold):

- JFK `verbatim`: small, medium, turbo, and large
- turbo `intended` on JFK, and official-default longform on additional held-out English audio (those clips and their gold dumps are not bundled here)
- turbo short clips for all 10 official languages (and unspaced segmentation)

Gold timestamps use teacher-forced *decoder-only* cross-attention plus official `extract_word_timings` — `generate(output_attentions=True)` can OOM on a 15 GiB CPU host. CTranslate2 int8 is a speed baseline only.

**Default fp32** is the product path and matches official transformers fp32 gold on the bundled fixtures (JFK and the 10-language clips). On a quiet 8-core Xeon measurement it was also faster than official transformers fp32, and ahead of a same-machine CTranslate2 int8 baseline on JFK; that is not a promise the default path will stay ahead or beat CT2 overall. Keep `--compute-type fp32` when you need gold-aligned text and timestamps. The default softmax/GELU uses glibc libmvec. The pre-packed AVX-512 encoder GEMM is default and bit-identical to gemm 0.18 on encoder-sized shapes. The tq=1 attention kernel is close to gemm 0.18 (~1e-7 / ~1e-6) but not bit-identical; it shipped only after the timestamp gate passed.

**Optional INT8** (`--compute-type int8`) is not the product default. It is a full-model INT8 path (last encoder layer kept in official fp32; other linears stay int8; default fp32 is unchanged) that aims to be faster and lower-memory than same-machine CT2 int8, and no less accurate than CT2 when both are scored against official fp32 gold. On JFK, text and timestamps match gold (MAE 0.0 ms). That is not a promise INT8 will stay ahead of CT2 on speed or memory, or close remaining gold gaps on other audio.

### Roadmap

1. ~~CPU text + word-timestamp alignment on English short audio, all four non-Pro sizes~~
2. ~~Continuation longform + hallucination repair + official-default temperature fallback and early-EOT~~
3. Broader multilingual coverage beyond the 15 s legal clips
4. Keep pushing CPU RTFx vs CT2 int8 on the exact fp32 path (optional int8 exists but is not default; no guarantees)
5. **After CPU is done:** implement the wgpu backend and verify it on a Kaggle T4 (Linux + CUDA)

Not in the CPU milestone (and not started): English ForcedAligner, hotwords (Pro-only), speculative decoding, verbatimize, dual-mode, deprecated v1.

## Build

Rust 1.70+ (tested with 1.100), CMake, and a C compiler are required (vendored [libsoxr](https://github.com/chirlu/soxr) is built at compile time for HQ resampling). A C++ compiler is not required. The current build expects **Linux with glibc** (`libmvec`) and an **AVX-512-capable** C toolchain (`src/libmvec_wrap.c` is compiled with `-mavx512f`). macOS, Windows, and non-AVX-512 hosts are not supported yet. For official-identical MP3 decode, `libsndfile` (`libsndfile.so.1`) should be available at runtime; otherwise the Symphonia fallback applies the same Xing/Info delay trim.

This repository does not set `target-cpu=native` by default. For the best SIMD on your machine, build with `RUSTFLAGS="-C target-cpu=native"`.

```bash
cargo build --release
# or: RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Download weights

```bash
# uses $HF_HOME if set; otherwise the huggingface_hub default cache
python scripts/download_models.py              # large turbo medium small
python scripts/download_models.py turbo small  # subset
```

Or copy a snapshot directory that contains `config.json`, `preprocessor_config.json`, `generation_config.json`, `tokenizer.json`, and `model.safetensors`.

## Transcribe

```bash
cargo run --release --bin transcribe -- \
  --model turbo \
  --model-dir /path/to/CrisperWhisper2.0_turbo \
  --audio sample.wav \
  --language en \
  --mode verbatim
```

Temperature fallback and early-EOT recovery default to on (official longform). Pass `--temperature-fallback false --early-eot-recovery false` for the deterministic greedy+repair-only path. `--compute-type` defaults to `fp32` (exact official alignment). Optional `--compute-type int8` is a full-model INT8 path (not the default): it aims to be faster and lower-memory than same-machine CT2 int8, and no less accurate than CT2 when both are scored against official fp32 gold. Keep `fp32` when you need the gold-aligned result. `--model` also accepts `large` / `medium` / `small`, or a direct directory path. `--language` is required. Mel bins are read from each checkpoint's `preprocessor_config.json` (`feature_size`: 128 for turbo, 80 for the others) and are not hard-coded.

## Tests

```bash
cargo test --release --lib --tests

# Compare a rust JSON dump to frozen gold
python scripts/compare_transcripts.py --gold tests/fixtures/gold_turbo_jfk_verbatim_words.json \
  --rust tests/fixtures/rust_turbo_jfk_verbatim.json

# Refresh gold (official Python package + weights)
python scripts/dump_word_timings.py --model turbo --audio /path/to/jfk.flac --language en --out gold.json
python scripts/dump_longform_gold.py --model turbo --audio /path/to/long.wav --out gold_long.json
python scripts/dump_longform_gold.py --model turbo --audio /path/to/long.wav --full-defaults --resume --out gold_long_defaults.json
```

Language clips and licenses: `tests/audio/langs/SOURCES.md`.

## License

- Rust code in this repository: [MIT](LICENSE)
- Port of [CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper): MIT, Copyright (c) 2026 Nyra Health GmbH
- Feature extraction follows official `features.py`, derived from [faster-whisper](https://github.com/SYSTRAN/faster-whisper): MIT, Copyright (c) 2023 SYSTRAN
- Vendored libsoxr: LGPLv2.1+ (see `third_party/soxr/LICENCE`)
- CrisperWhisper model weights: Nyra non-commercial research license (see the official model card)
