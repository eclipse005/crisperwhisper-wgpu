# 测试目录

这里放的是公开测试音频和转写结果，用来核对 Rust 移植的**准确度**，不是速度榜。

`wgpu` feature 仍是占位，**尚未实现**。快速 CPU 内核（编码器 GEMM、注意力、softmax/GELU）目前仍只走 **AVX-512**。没有 AVX-512 的机器（例如 Intel Arrow Lake / Core Ultra 200）会落到标量回退，看起来会慢很多；那些 RTFx 数字不能当作产品基线。

## 四套结果

准确度一律对 **官方 transformers fp32** 评分。CTranslate2 INT8 只用来对比速度和内存。

| 套别 | 目录 | 是什么 | 怎么用 |
|---|---|---|---|
| 1. 官方金标 | `tests/gold/official_fp32/` | 官方 Python `transformers` fp32（文本 + 词级时间戳）。带词的文件用 teacher-forced 只跑 decoder 的交叉注意力，再走官方 `extract_word_timings`。 | **这是准确度金标。** |
| 2. Rust fp32（移植 fp32） | `tests/gold/rust_fp32/` | 移植默认的 `--compute-type fp32` 路径。 | 必须与官方金标一致：文本完全相同，时间戳 MAE 0.0 ms，没有超过 20 ms 软门槛的词。 |
| 3. CT2 INT8 | `tests/gold/ct2_int8/` | 官方 CTranslate2 int8（`backend=ct2`，`compute_type=int8`），带词级时间戳。 | 只用于速度和内存对比。**不是准确度金标。** 不要去追着对齐 CT2 INT8 的文本。 |
| 4. Rust INT8 | `tests/gold/rust_int8/` | 移植可选的 `--compute-type int8` 路径（最后一层编码器保持官方 fp32）。 | 准确度对官方 fp32 金标（第 1 套）评分，不要对 CT2 INT8 对齐。两边都对官方金标评分时，不应差于 CT2。 |

官方金标和 Rust fp32 里的 `turbo_jfk_intended.json` **只有文本**，没有词级时间戳。通过标准是文本完全一致。

medium 的 JFK 官方文本标点与 turbo / small / large 略有不同：medium 是 `you. Ask`，其余是 `you, ask`。每个模型请对它自己的官方金标文件比较。

zh / ja / th 语种片段只做短音频的**切分**检查，不做词级时间戳质量声明。

## 公开音频

| 片段 | 路径 | 时长 | 语言 | 来源 / 许可 |
|---|---|---|---:|---|
| JFK 就职演说（节选） | `tests/audio/jfk.flac` | 11.0 秒 | en | 美国公有领域演讲（经典 Whisper 示例）。sha256 `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`，1152693 字节。 |
| 官方榜 10 语 | `tests/audio/langs/{en,de,es,nl,fr,it,pl,pt,ru,sv}.wav` | 15.0 秒 | 见文件名 | LibriVox / Internet Archive，公有领域或 CC0。详见 `tests/audio/langs/SOURCES.md` 和 `manifest.json`。 |
| zh / ja / th | `tests/audio/langs/{zh,ja,th}.wav` | 3.9 秒 / 1.4 秒 / 4.1 秒 | zh / ja / th | Wikimedia Commons / Lingua Libre（CC0 或 CC BY 4.0）。只验切分。见 `SOURCES.md`。 |

用户评测音频（`tests/audio/user_langs/`、`tests/audio/user_zh_ja/`，以及 audio1 / audio2 / audio3）**不**在公开树里。

## 用哪份 JSON

四个目录里的文件名都是 `{模型}_{片段}_{模式}.json`。JFK 有 small、medium、turbo、large；语种片段只有 turbo。

例如：

- 官方金标，turbo JFK verbatim：`tests/gold/official_fp32/turbo_jfk_verbatim.json`
- 同片段的 Rust fp32：`tests/gold/rust_fp32/turbo_jfk_verbatim.json`
- 官方金标，turbo 西班牙语：`tests/gold/official_fp32/turbo_es_verbatim.json`
- CT2 INT8 / Rust INT8：`tests/gold/ct2_int8/…` 和 `tests/gold/rust_int8/…`，文件名相同

实验室绝对路径（`/workspace/...`）已改成仓库内相对路径，例如 `tests/audio/jfk.flac`。模型位置写成 `nyralabs/CrisperWhisper2.0_{size}`。公开 JSON 不含实验室机器耗时、PID 或注意力 dump。

## 如何转写和比较

权重不在本仓库，需先自行下载（`scripts/download_models.py` 或 `--model-dir`）。

```bash
# Rust 默认 fp32（必须与官方金标一致）
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

# 可选 INT8：仍然对官方 fp32 金标评分，不要对 CT2 INT8 对齐
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

### 通过标准

- **Rust fp32 对官方金标：** `text_exact` 为真，时间戳 MAE 为 0.0 ms，且 `words_over_soft_ms` 为 0（20 ms 软门槛）。文本和时间戳都通过时，`compare_transcripts.py` 才返回 0。`turbo_jfk_intended`（没有词）只要求文本完全一致。
- **Rust INT8 对官方金标：** 文本和时间戳都对官方 fp32 评分。允许出现与 CT2 INT8 同类的偏差；两边都对同一份官方金标评分时，不应差于 CT2。与 `tests/gold/ct2_int8/` 不一致本身不算失败。
- **CT2 INT8：** 只留作速度 / 内存对比，不要当成准确度目标。

单元测试（log-mel / Hann / 梅尔滤波器张量）仍从 `tests/fixtures/` 读取（`hann400.f32`、`mel_filters_*.f32`、`jfk_*`、`jfk6s_*`、`silence_*`、`noise_*`、`features_meta.json`）：

```bash
cargo test --release --lib --tests
```

要与官方逐样点一致地解码 MP3，Unix 上仍需 `libsndfile`。没有时，WAV 走 hound，其他格式走 Symphonia（同样会裁剪 Xing/Info 延迟）。本目录里的公开音频是 WAV 和 FLAC，Windows / macOS 没有 libsndfile 也能加载。
