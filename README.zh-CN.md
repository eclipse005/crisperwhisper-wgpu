# crisperwhisper-wgpu

[CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper) 的 Rust 移植。当前里程碑是先把 **CPU** 实现做完，并在文本和词级时间戳上对齐官方 Python `transformers` fp32 路径。CPU 对齐完成之后，再做 **wgpu** GPU 后端，并在 Kaggle T4（Linux + CUDA）上验证。仓库里已预留 `wgpu` feature，**现在还没有实现**；本阶段不要指望 `--features wgpu` 会跑起来。

本仓库**只包含代码**。官方模型权重不会入库，需要自行从 Hugging Face 下载（`nyralabs/CrisperWhisper2.0_{large,turbo,medium,small}`）。这些权重由 Nyra 以**非商用研究许可**发布，下载和使用前请阅读并遵守模型卡上的许可。本仓库的 MIT 许可只覆盖这里的 Rust 代码，不覆盖权重。

## 现状

**当前重点：先把 Rust CPU 移植对齐做完。** wgpu 放到 CPU 完成之后。

CPU 上已经能跑的部分（fp32 计算，官方 BF16 safetensors）：

- 模型：`large` / `turbo` / `medium` / `small`（不含 `*_pro`，不含已弃用的 v1）——四个都已下载，并在英语 JFK 上对照过金标
- 贪心解码，`verbatim` / `intended` 提示，必须指定语言（不做自动语种检测）
- 对数梅尔特征与官方 `crisperwhisper.features.FeatureExtractor` 数值对齐（语音上最坏最大绝对误差约 1.7e-5）
- 词级时间戳（cross-attention + 梅尔能量 blank + Viterbi，帧长 20 ms）以及 `[UM]` `[UH]` `[scream]` 等特殊标记
- 默认风格的 continuation longform（30 秒窗、26 秒步长、12 词上下文、按时间戳丢边、幻觉修复、temperature fallback、early-EOT）
- MP3/有损解码与官方 `load_audio` 对齐（优先 libsndfile，再 soxr HQ；无 libsndfile 时对 Xing/Info 编码器延迟做与官方相同的裁剪）
- 官方榜 10 语的合法 15 秒素材，外加无空格 zh/ja/th（只验切分），见 `tests/audio/langs/`

相对官方 **transformers fp32** 金标（文本完全一致，词级时间戳 MAE 0.0 ms，没有超过 20 ms 软门槛的词）：

- JFK `verbatim`：small / medium / turbo / large
- turbo 在 JFK 上的 `intended`，以及在额外英语长音频上的官方默认 longform（这些片段和对应金标不入库）
- turbo 对官方 10 语短片段（以及无空格语切分）

金标时间戳使用 decoder-only teacher-forced cross-attention + 官方 `extract_word_timings`——在 15 GiB 的 CPU 机器上 `generate(output_attentions=True)` 会 OOM。CTranslate2 int8 只用于速度对照。

**默认 fp32** 是产品路径，在已入库夹具（JFK 和 10 语短片段）上与官方 transformers fp32 金标对齐。在一台 8 核 Xeon 的安静测量中，它也快于官方 transformers fp32，并在 JFK 上快于同机 CTranslate2 int8 基线；这不是在打包票默认路径会稳定或全面超过 CT2。需要与金标完全一致时请保持 `--compute-type fp32`。默认 softmax/GELU 走 glibc libmvec。预打包 AVX-512 编码器 GEMM 仍为默认，且与 gemm 0.18 在编码器尺寸上逐比特一致。tq=1 注意力内核与 gemm 0.18 接近（约 1e-7 / 1e-6）但不是逐比特一致，过时间戳门槛后才进默认。

**可选 INT8**（`--compute-type int8`）不是产品默认。这是全模型 INT8 路径（最后一层编码器留在官方 fp32，其余线性层仍是 int8；默认 fp32 不变），目标是同机速度超过 CT2 int8、峰值内存低于 CT2，以及相对官方 fp32 金标的精度不低于 CT2。在 JFK 上文本与时间戳与金标一致（MAE 0.0 ms）。不打包票 INT8 会一直在速度或内存上超过 CT2，也不打包票能收完其他音频上剩余的金标差距。

### 路线图

1. ~~英语短音频文本 + 词级时间戳对齐，四个非 Pro 型号~~
2. ~~continuation longform + 幻觉修复 + 官方默认 temperature fallback 与 early-EOT~~
3. 比 15 秒合法片段更全面的多语言覆盖
4. 在精确 fp32 路径上继续把 CPU RTFx 往 CT2 int8 上推（已有可选 int8，但因对齐差异不能作为默认；不事先打包票）
5. **CPU 对齐之后：** 实现 wgpu 后端，并在 Kaggle T4（Linux + CUDA）上验证

CPU 里程碑不做（也尚未开始）：English ForcedAligner、热词（仅 Pro）、投机解码、verbatimize、双通路、已弃用的 v1。

## 编译

需要 Rust 1.70+（已在 1.100 上测试）、CMake 和 C 编译器（编译时会构建 vendored [libsoxr](https://github.com/chirlu/soxr) 做 HQ 重采样）。不需要 C++ 编译器。现在可以在 **Linux（glibc）、Windows（MSVC 或 GNU）和 macOS** 上编译并运行。快速 CPU 内核（编码器 GEMM、注意力、softmax/GELU）目前仍只走 **AVX-512**；没有 AVX-512 的机器（例如 Intel Arrow Lake / Core Ultra 200）会落到标量回退，测出的速度不能当作产品 RTFx 基线。glibc `libmvec` 的 C 包装（`src/libmvec_wrap.c`，以 `-mavx512f` 编译）只在 x86_64 Linux + glibc 上构建；其他目标用标量 `libm` 提供同名符号。要与官方逐样点一致地解码 MP3，Unix 运行时仍需能找到 `libsndfile`（`libsndfile.so.1`，macOS 上也可以是 `libsndfile.1.dylib`）。没有时，WAV 走 hound，其他格式走 Symphonia 回退（同样会裁剪 Xing/Info 延迟）。`wgpu` feature 仍是占位，尚未实现。

本仓库默认不会设置 `target-cpu=native`。若要在本机用上最好的 SIMD，编译时加上 `RUSTFLAGS="-C target-cpu=native"`。

```bash
cargo build --release
# 或：RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## 下载权重

```bash
# 若设置了 $HF_HOME 则用之，否则走 huggingface_hub 的默认缓存
python scripts/download_models.py              # large turbo medium small
python scripts/download_models.py turbo small  # 子集
```

也可以直接指向一个包含 `config.json`、`preprocessor_config.json`、`generation_config.json`、`tokenizer.json` 和 `model.safetensors` 的本地目录。

## 转录

```bash
cargo run --release --bin transcribe -- \
  --model turbo \
  --model-dir /path/to/CrisperWhisper2.0_turbo \
  --audio sample.wav \
  --language en \
  --mode verbatim
```

temperature fallback 与 early-EOT 默认开启（官方 longform）。只跑贪心+修复时加 `--temperature-fallback false --early-eot-recovery false`。`--compute-type` 默认为 `fp32`（与官方金标精确对齐）。可选 `--compute-type int8` 是全模型 INT8 路径（不是默认）：目标是同机速度超过 CT2 int8、峰值内存低于 CT2，以及相对官方 fp32 金标的精度不低于 CT2。需要与金标完全一致时请保持 `fp32`。`--language` 为必填。梅尔维度从各 checkpoint 的 `preprocessor_config.json` 读取（turbo 为 128，其余为 80），不能写死。

## 测试

公开音频、四套结果（官方 fp32 金标、Rust fp32、CT2 INT8、Rust INT8）和通过标准见 [`tests/README.zh-CN.md`](tests/README.zh-CN.md)。准确度以官方 transformers fp32 为金标；CT2 INT8 只用于速度和内存对比。

```bash
cargo test --release --lib --tests
python scripts/compare_transcripts.py \
  --gold tests/gold/official_fp32/turbo_jfk_verbatim.json \
  --rust tests/gold/rust_fp32/turbo_jfk_verbatim.json
```

语种素材与许可见 `tests/audio/langs/SOURCES.md`。

## 许可

- 本仓库 Rust 代码：[MIT](LICENSE)
- 移植自 [CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper)：MIT，Copyright (c) 2026 Nyra Health GmbH
- 特征提取对齐官方 `features.py`，后者源自 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)：MIT，Copyright (c) 2023 SYSTRAN
- Vendored libsoxr：LGPLv2.1+（见 `third_party/soxr/LICENCE`）
- CrisperWhisper 模型权重：Nyra 非商用研究许可（见官方模型卡）
