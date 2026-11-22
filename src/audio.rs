//! Load audio as 16 kHz mono f32, resampled with soxr HQ (same as official
//! `crisperwhisper.audio._resample`).

use std::ffi::CStr;
use std::os::raw::{c_char, c_uint, c_ulong, c_void};
use std::path::Path;

use crate::error::{Error, Result};

pub const SAMPLE_RATE: u32 = 16_000;
const SOXR_BLOCK: usize = 8192;

pub fn load_audio(path: &Path) -> Result<Vec<f32>> {
    let (samples, sr, channels) = decode_file(path)?;
    let mono = if channels == 1 {
        samples
    } else {
        downmix_mono(&samples, channels)
    };
    if sr == SAMPLE_RATE {
        return Ok(mono);
    }
    resample_soxr(&mono, sr, SAMPLE_RATE)
}

fn downmix_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    let n = samples.len() / channels;
    let inv = 1.0 / channels as f32;
    let mut mono = vec![0.0f32; n];
    for i in 0..n {
        let mut s = 0.0f32;
        for c in 0..channels {
            s += samples[i * channels + c];
        }
        mono[i] = s * inv;
    }
    mono
}

fn decode_file(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "wav" {
        if let Ok(decoded) = decode_wav_hound(path) {
            return Ok(decoded);
        }
    }
    // Official `load_audio` uses libsndfile (Python soundfile) first. This
    // path is bit-identical to that decoder, including MP3 encoder delay
    // and padding. It is Unix-only (`dlopen`). On Windows, and on Unix when
    // libsndfile is not installed, Symphonia is the portable fallback; WAV
    // already went through hound above.
    #[cfg(unix)]
    if let Ok(decoded) = decode_libsndfile(path) {
        return Ok(decoded);
    }
    decode_symphonia(path)
}

#[cfg(unix)]
#[repr(C)]
struct SfInfo {
    frames: i64,
    samplerate: i32,
    channels: i32,
    format: i32,
    sections: i32,
    seekable: i32,
}

#[cfg(unix)]
fn decode_libsndfile(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_void};

    const RTLD_LAZY: c_int = 1;
    const SFM_READ: c_int = 0x10;

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    type SfOpen = unsafe extern "C" fn(*const c_char, c_int, *mut SfInfo) -> *mut c_void;
    type SfReadfFloat = unsafe extern "C" fn(*mut c_void, *mut f32, i64) -> i64;
    type SfClose = unsafe extern "C" fn(*mut c_void) -> c_int;
    type SfStrerror = unsafe extern "C" fn(*mut c_void) -> *const c_char;

    // Linux ships .so; macOS Homebrew / system installs use .dylib.
    // Trying both on every Unix is harmless if a name is absent.
    let names = [
        c"libsndfile.so.1",
        c"libsndfile.so",
        c"libsndfile.1.dylib",
        c"libsndfile.dylib",
    ];
    let mut handle = std::ptr::null_mut();
    for name in names {
        handle = unsafe { dlopen(name.as_ptr(), RTLD_LAZY) };
        if !handle.is_null() {
            break;
        }
    }
    if handle.is_null() {
        return Err(Error::Audio("libsndfile not found".into()));
    }

    let load = |sym: &CStr| -> Result<*mut c_void> {
        let p = unsafe { dlsym(handle, sym.as_ptr()) };
        if p.is_null() {
            unsafe { dlclose(handle) };
            Err(Error::Audio(format!("libsndfile missing {}", sym.to_string_lossy())))
        } else {
            Ok(p)
        }
    };

    let sf_open = unsafe { std::mem::transmute::<*mut c_void, SfOpen>(load(c"sf_open")?) };
    let sf_readf_float =
        unsafe { std::mem::transmute::<*mut c_void, SfReadfFloat>(load(c"sf_readf_float")?) };
    let sf_close = unsafe { std::mem::transmute::<*mut c_void, SfClose>(load(c"sf_close")?) };
    let sf_strerror =
        unsafe { std::mem::transmute::<*mut c_void, SfStrerror>(load(c"sf_strerror")?) };

    let c_path = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| Error::Audio(format!("path contains NUL: {}", path.display())))?;
    let mut info = SfInfo {
        frames: 0,
        samplerate: 0,
        channels: 0,
        format: 0,
        sections: 0,
        seekable: 0,
    };
    let snd = unsafe { sf_open(c_path.as_ptr(), SFM_READ, &mut info) };
    if snd.is_null() {
        let msg = unsafe { CStr::from_ptr(sf_strerror(snd)) }.to_string_lossy();
        unsafe { dlclose(handle) };
        return Err(Error::Audio(format!("libsndfile open {}: {msg}", path.display())));
    }
    let channels = info.channels.max(1) as usize;
    let sr = info.samplerate as u32;
    let frames = if info.frames > 0 { info.frames as usize } else { 0 };
    let mut samples = if frames > 0 {
        vec![0.0f32; frames.saturating_mul(channels)]
    } else {
        Vec::new()
    };
    let mut read_frames = 0i64;
    if frames > 0 {
        read_frames = unsafe { sf_readf_float(snd, samples.as_mut_ptr(), info.frames) };
    } else {
        // Unknown length: read in 1s chunks.
        let chunk_frames = sr.max(1) as usize;
        let mut tmp = vec![0.0f32; chunk_frames * channels];
        loop {
            let n = unsafe { sf_readf_float(snd, tmp.as_mut_ptr(), chunk_frames as i64) };
            if n <= 0 {
                break;
            }
            samples.extend_from_slice(&tmp[..n as usize * channels]);
            read_frames += n;
        }
    }
    unsafe {
        sf_close(snd);
        dlclose(handle);
    }
    if read_frames < 0 {
        return Err(Error::Audio(format!("libsndfile read failed for {}", path.display())));
    }
    samples.truncate(read_frames as usize * channels);
    if sr == 0 || samples.is_empty() {
        return Err(Error::Audio(format!("libsndfile produced no audio for {}", path.display())));
    }
    Ok((samples, sr, channels))
}

/// LAME-compatible encoder delay/padding from an MP3 Xing/Info tag.
/// Returns `(leading, trailing)` samples per channel, including the 529-sample
/// decoder delay that libsndfile / FFmpeg apply. `None` if the tag is absent
/// or the delay fields are zero.
fn mp3_xing_encoder_trim(path: &Path) -> Option<(usize, usize)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; 16_384];
    let n = f.read(&mut buf).ok()?;
    buf.truncate(n);
    let tag_at = buf.windows(4).position(|w| w == b"Xing" || w == b"Info")?;
    if tag_at + 8 > buf.len() {
        return None;
    }
    let flags = u32::from_be_bytes(buf[tag_at + 4..tag_at + 8].try_into().ok()?);
    let mut off = tag_at + 8;
    if flags & 0x1 != 0 {
        off += 4;
    }
    if flags & 0x2 != 0 {
        off += 4;
    }
    if flags & 0x4 != 0 {
        off += 100;
    }
    if flags & 0x8 != 0 {
        off += 4;
    }
    // Encoder string (9 bytes) then the 24-bit delay/padding at offset +21.
    let field = off + 21;
    if field + 3 > buf.len() {
        return None;
    }
    let v = u32::from_be_bytes([0, buf[field], buf[field + 1], buf[field + 2]]);
    let raw_delay = (v >> 12) & 0xfff;
    let raw_pad = v & 0xfff;
    if raw_delay == 0 && raw_pad == 0 {
        return None;
    }
    const DECODER_DELAY: u32 = 528 + 1;
    let delay = (DECODER_DELAY + raw_delay) as usize;
    let padding = raw_pad.saturating_sub(DECODER_DELAY) as usize;
    Some((delay, padding))
}

fn trim_encoder_padding(samples: &mut Vec<f32>, channels: usize, delay: usize, padding: usize) {
    if channels == 0 || samples.is_empty() {
        return;
    }
    let frames = samples.len() / channels;
    let start = delay.min(frames);
    let end = frames.saturating_sub(padding).max(start);
    if start == 0 && end == frames {
        return;
    }
    let lo = start * channels;
    let hi = end * channels;
    samples.copy_within(lo..hi, 0);
    samples.truncate(hi - lo);
}

fn decode_wav_hound(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    let mut reader = hound::WavReader::open(path).map_err(|e| Error::Audio(e.to_string()))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(e.to_string()))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            // Match libsndfile / soundfile float conversion: divide by 2^(bits-1).
            let scale = (1u32 << (bits.saturating_sub(1).min(31))) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::Audio(e.to_string()))?
        }
    };
    Ok((samples, sr, channels))
}

fn decode_symphonia(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    use std::fs::File;
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SyError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let fmt_opts = FormatOptions { enable_gapless: true, ..FormatOptions::default() };
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &MetadataOptions::default())
        .map_err(|e| Error::Audio(format!("probe {}: {e}", path.display())))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| Error::Audio(format!("no audio track in {}", path.display())))?
        .clone();
    let sr = track
        .codec_params
        .sample_rate
        .ok_or_else(|| Error::Audio("missing sample rate".into()))?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::Audio(e.to_string()))?;

    let mut samples = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SyError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SyError::ResetRequired) => break,
            Err(e) => return Err(Error::Audio(e.to_string())),
        };
        if packet.track_id() != track.id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                if sample_buf.is_none() {
                    sample_buf = Some(SampleBuffer::<f32>::new(
                        decoded.capacity() as u64,
                        *decoded.spec(),
                    ));
                }
                if let Some(buf) = sample_buf.as_mut() {
                    buf.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(buf.samples());
                }
            }
            Err(SyError::DecodeError(_)) => continue,
            Err(e) => return Err(Error::Audio(e.to_string())),
        }
    }
    // Symphonia only auto-trims when a LAME tag is present. FFmpeg/Lavc
    // writes the same delay/padding fields under a "Lavc..." encoder id;
    // libsndfile still honours them. Apply that trim when the demuxer
    // did not already advertise a delay.
    let already = track.codec_params.delay.unwrap_or(0) > 0;
    if !already {
        if let Some((delay, padding)) = mp3_xing_encoder_trim(path) {
            trim_encoder_padding(&mut samples, channels, delay, padding);
        }
    }
    Ok((samples, sr, channels))
}

fn resample_soxr(mono: &[f32], sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    #[repr(C)]
    struct SoxrQualitySpec {
        precision: f64,
        phase_response: f64,
        passband_end: f64,
        stopband_begin: f64,
        e: *mut c_void,
        flags: c_ulong,
    }
    #[repr(C)]
    struct SoxrRuntimeSpec {
        log2_min_dft_size: c_uint,
        log2_large_dft_size: c_uint,
        coef_size_kbytes: c_uint,
        num_threads: c_uint,
        e: *mut c_void,
        flags: c_ulong,
    }
    type SoxrT = *mut c_void;
    type SoxrErrorT = *const c_char;
    const SOXR_HQ: c_ulong = 4;

    unsafe extern "C" {
        fn soxr_quality_spec(recipe: c_ulong, flags: c_ulong) -> SoxrQualitySpec;
        fn soxr_runtime_spec(num_threads: c_uint) -> SoxrRuntimeSpec;
        fn soxr_create(
            input_rate: f64,
            output_rate: f64,
            num_channels: c_uint,
            err: *mut SoxrErrorT,
            io_spec: *const c_void,
            quality_spec: *const SoxrQualitySpec,
            runtime_spec: *const SoxrRuntimeSpec,
        ) -> SoxrT;
        fn soxr_process(
            resampler: SoxrT,
            in_: *const c_void,
            ilen: usize,
            idone: *mut usize,
            out: *mut c_void,
            olen: usize,
            odone: *mut usize,
        ) -> SoxrErrorT;
        fn soxr_delete(resampler: SoxrT);
    }

    fn soxr_err(err: SoxrErrorT) -> Result<()> {
        if err.is_null() {
            return Ok(());
        }
        let msg = unsafe { CStr::from_ptr(err) }.to_string_lossy();
        Err(Error::Audio(format!("soxr: {msg}")))
    }

    let expected = (mono.len() as f64 * target_sr as f64 / sr as f64).ceil() as usize;
    let q = unsafe { soxr_quality_spec(SOXR_HQ, 0) };
    let rt = unsafe { soxr_runtime_spec(1) };
    let mut err: SoxrErrorT = std::ptr::null();
    let soxr = unsafe {
        soxr_create(
            sr as f64,
            target_sr as f64,
            1,
            &mut err,
            std::ptr::null(),
            &q,
            &rt,
        )
    };
    soxr_err(err)?;
    if soxr.is_null() {
        return Err(Error::Audio("soxr_create returned null".into()));
    }

    // Bounded blocks keep soxr's DFT working set in cache (same approach as
    // well-known soxr streaming usage); flush with a null input at the end.
    let block = SOXR_BLOCK;
    let mut out = vec![0.0f32; expected + 8192 + block];
    let mut in_off = 0usize;
    let mut written = 0usize;
    let result = (|| -> Result<Vec<f32>> {
        while in_off < mono.len() {
            let mut idone = 0usize;
            let mut odone = 0usize;
            let remain_out = out.len() - written;
            if remain_out == 0 {
                return Err(Error::Audio("soxr output overflow".into()));
            }
            let want_in = (mono.len() - in_off).min(block);
            let err = unsafe {
                soxr_process(
                    soxr,
                    mono[in_off..].as_ptr().cast(),
                    want_in,
                    &mut idone,
                    out[written..].as_mut_ptr().cast(),
                    remain_out,
                    &mut odone,
                )
            };
            soxr_err(err)?;
            if idone == 0 && odone == 0 {
                return Err(Error::Audio("soxr made no progress".into()));
            }
            in_off += idone;
            written += odone;
        }
        loop {
            let remain_out = out.len() - written;
            if remain_out == 0 {
                break;
            }
            let mut idone = 0usize;
            let mut odone = 0usize;
            let err = unsafe {
                soxr_process(
                    soxr,
                    std::ptr::null(),
                    0,
                    &mut idone,
                    out[written..].as_mut_ptr().cast(),
                    remain_out,
                    &mut odone,
                )
            };
            soxr_err(err)?;
            if odone == 0 {
                break;
            }
            written += odone;
        }
        // Official `soxr.resample(..., quality="HQ")` returns the library's
        // natural output length. Do not pad/truncate to ceil(n * out/in):
        // that added a trailing zero and made 48 kHz -> 16 kHz one sample
        // longer than Python soxr (34865494 vs 34865493 on audio3).
        out.truncate(written);
        Ok(out)
    })();
    unsafe { soxr_delete(soxr) };
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_channels() {
        let stereo = vec![1.0, 3.0, 5.0, 7.0];
        assert_eq!(downmix_mono(&stereo, 2), vec![2.0, 6.0]);
    }

    #[test]
    fn xing_trim_on_official_audio3_mp3() {
        let p = Path::new("/workspace/crisperwhisper-run/input/audio3_213421c.mp3");
        if !p.exists() {
            return;
        }
        let (delay, padding) = mp3_xing_encoder_trim(p).expect("Lavc Info tag trim");
        assert_eq!((delay, padding), (1105, 559));
    }

    #[test]
    #[ignore]
    fn audio3_mp3_matches_official_16k_wav() {
        let mp3 = Path::new("/workspace/crisperwhisper-run/input/audio3_213421c.mp3");
        let wav = Path::new("/workspace/crisperwhisper-run/clips/audio3_official_16k.wav");
        if !mp3.exists() || !wav.exists() {
            return;
        }
        let rust = load_audio(mp3).expect("load mp3");
        let gold = load_audio(wav).expect("load official wav");
        assert_eq!(rust.len(), gold.len(), "sample count mp3={} official={}", rust.len(), gold.len());
        let mut max = 0.0f32;
        let mut arg = 0usize;
        for (i, (&a, &b)) in rust.iter().zip(gold.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max {
                max = d;
                arg = i;
            }
        }
        assert!(
            max < 1e-6,
            "mp3 vs official wav max_abs={max} at sample {arg}"
        );
    }

    /// Force the portable decoder (skip libsndfile) and compare to official WAV.
    #[test]
    #[ignore]
    fn audio3_mp3_symphonia_fallback_matches_official_16k_wav() {
        let mp3 = Path::new("/workspace/crisperwhisper-run/input/audio3_213421c.mp3");
        let wav = Path::new("/workspace/crisperwhisper-run/clips/audio3_official_16k.wav");
        if !mp3.exists() || !wav.exists() {
            return;
        }
        let (samples, sr, channels) = decode_symphonia(mp3).expect("symphonia mp3");
        let mono = if channels == 1 {
            samples
        } else {
            downmix_mono(&samples, channels)
        };
        let rust = if sr == SAMPLE_RATE {
            mono
        } else {
            resample_soxr(&mono, sr, SAMPLE_RATE).expect("soxr")
        };
        let gold = load_audio(wav).expect("load official wav");
        assert_eq!(
            rust.len(),
            gold.len(),
            "symphonia sample count mp3={} official={}",
            rust.len(),
            gold.len()
        );
        let mut max = 0.0f32;
        let mut arg = 0usize;
        for (i, (&a, &b)) in rust.iter().zip(gold.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max {
                max = d;
                arg = i;
            }
        }
        let mut sum = 0.0f64;
        let mut over_1e6 = 0usize;
        for (&a, &b) in rust.iter().zip(gold.iter()) {
            let d = (a - b).abs();
            sum += d as f64;
            if d >= 1e-6 {
                over_1e6 += 1;
            }
        }
        let mae = sum / rust.len() as f64;
        eprintln!(
            "symphonia vs official: n={} max_abs={max:.6e} at {arg} mae={mae:.6e} n>=1e-6={over_1e6}",
            rust.len()
        );
        // Length already matches (Xing trim). Residual is decoder arithmetic,
        // far below log-mel / ASR sensitivity; libsndfile remains the
        // official-equivalent path (max_abs < 1e-6).
        assert!(
            max < 1e-4,
            "symphonia vs official wav max_abs={max} at sample {arg} mae={mae}"
        );
    }

    #[test]
    fn trim_encoder_padding_drops_leading_and_trailing() {
        let mut s: Vec<f32> = (0..20).map(|i| i as f32).collect();
        trim_encoder_padding(&mut s, 2, 2, 1);
        assert_eq!(s, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    }
}
