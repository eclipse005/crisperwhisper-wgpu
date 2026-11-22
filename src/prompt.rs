//! Prompt construction matching official `crisperwhisper.prompt.PromptBuilder`.
//!
//! Verbatim / intended only change the `[verbatim_N]` / `[intended_N]` prefix.
//! Language must be provided; there is no auto language detect. Hotwords and
//! verbatimize are out of scope for phase 1.

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use crate::tokenizer::WhisperTokenizer;

pub const PROMPT_MARKER_TOKENS: &[&str] = &["<vtx>", "<evtx>", "<htx>", "<ehtx>", "<ctx>", "<ectx>"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptMode {
    Verbatim,
    Intended,
}

impl TranscriptMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "verbatim" => Ok(Self::Verbatim),
            "intended" => Ok(Self::Intended),
            other => Err(Error::msg(format!(
                "unknown mode {other:?}; expected 'verbatim' or 'intended'"
            ))),
        }
    }

    pub fn tags(self, count: usize) -> String {
        let stem = match self {
            Self::Verbatim => "verbatim",
            Self::Intended => "intended",
        };
        (1..=count).map(|i| format!("[{stem}_{i}]")).collect()
    }
}

pub fn decoder_prefix(cfg: &ModelConfig, language: &str) -> Result<Vec<u32>> {
    let mut prefix = vec![cfg.decoder_start_token_id];
    prefix.push(cfg.language_id(language)?);
    if let Some(id) = cfg.transcribe_token_id {
        prefix.push(id);
    }
    if let Some(id) = cfg.no_timestamps_token_id {
        prefix.push(id);
    }
    Ok(prefix)
}

/// Mode tags (optionally plus continuation context) followed by the Whisper
/// decoder prefix `<|sot|><|lang|><|transcribe|><|notimestamps|>`.
pub fn build_prompt(
    tok: &WhisperTokenizer,
    cfg: &ModelConfig,
    mode: TranscriptMode,
    language: &str,
    context: Option<&str>,
) -> Result<Vec<u32>> {
    let mut text = mode.tags(5);
    if let Some(ctx) = context {
        if !ctx.is_empty() {
            text.push_str(&format!(" <ctx> {ctx} <ectx>"));
        }
    }
    let mut ids = tok.encode_text(&text)?;
    ids.extend(decoder_prefix(cfg, language)?);
    Ok(ids)
}

/// Remove CrisperWhisper prompt markers from decoded output text.
/// Non-verbal tags such as `[UM]`, `[UH]`, `[breath]`, `[laughter]` are kept.
pub fn strip_prompt_artifacts(text: &str) -> String {
    let mut s = strip_numbered_tags(text, "verbatim_");
    s = strip_numbered_tags(&s, "intended_");
    s = strip_wrapped(&s, "<htx>", "<ehtx>");
    s = strip_wrapped(&s, "<vtx>", "<evtx>");
    s = strip_wrapped(&s, "<ctx>", "<ectx>");
    s = strip_wrapped(&s, "<sot>", "<eot>");
    if let Some(i) = s.find("<sot>") {
        s.truncate(i);
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_numbered_tags(text: &str, stem: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let needle = format!("[{stem}");
    while let Some(i) = rest.find(&needle) {
        out.push_str(&rest[..i]);
        let after = &rest[i + needle.len()..];
        if let Some(end) = after.find(']') {
            let digits = &after[..end];
            if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                rest = &after[end + 1..];
                continue;
            }
        }
        // Not a numbered tag; copy the '[' and continue after it.
        out.push('[');
        rest = &rest[i + 1..];
    }
    out.push_str(rest);
    out
}

fn strip_wrapped(text: &str, start: &str, end: &str) -> String {
    let mut s = text.to_string();
    while let Some(a) = s.find(start) {
        if let Some(rel) = s[a..].find(end) {
            let b = a + rel + end.len();
            s.replace_range(a..b, "");
        } else {
            break;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbatim_tags() {
        assert_eq!(
            TranscriptMode::Verbatim.tags(5),
            "[verbatim_1][verbatim_2][verbatim_3][verbatim_4][verbatim_5]"
        );
    }

    #[test]
    fn strip_keeps_nonverbal() {
        let t = "[verbatim_1][intended_2] hello [UM] [breath] world <ctx> x <ectx>";
        assert_eq!(strip_prompt_artifacts(t), "hello [UM] [breath] world");
    }
}
