use std::collections::HashSet;
use std::path::Path;

use tokenizers::Tokenizer;

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use crate::prompt::PROMPT_MARKER_TOKENS;

const PROMPT_TAG_PREFIXES: &[&str] = &["[verbatim_", "[intended_"];

pub struct WhisperTokenizer {
    inner: Tokenizer,
    pub all_special_ids: HashSet<u32>,
    pub eot_id: u32,
    pub sot_id: u32,
    pub no_timestamps_id: Option<u32>,
    pub transcribe_id: Option<u32>,
}

impl WhisperTokenizer {
    pub fn load(model_dir: &Path, cfg: &ModelConfig) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path).map_err(|e| Error::Tokenizer(e.to_string()))?;

        let mut all_special_ids = HashSet::new();
        for (token, id) in inner.get_vocab(true) {
            if (token.starts_with("<|") && token.ends_with("|>"))
                || PROMPT_TAG_PREFIXES.iter().any(|p| token.starts_with(p))
                || PROMPT_MARKER_TOKENS.iter().any(|m| token == *m)
            {
                all_special_ids.insert(id);
            }
        }

        let eot_id = cfg.eos_token_id;
        let sot_id = cfg.decoder_start_token_id;
        all_special_ids.insert(eot_id);
        all_special_ids.insert(sot_id);
        if let Some(id) = cfg.no_timestamps_token_id {
            all_special_ids.insert(id);
        }
        if let Some(id) = cfg.transcribe_token_id {
            all_special_ids.insert(id);
        }
        for id in cfg.lang_to_id.values() {
            all_special_ids.insert(*id);
        }

        Ok(Self {
            inner,
            all_special_ids,
            eot_id,
            sot_id,
            no_timestamps_id: cfg.no_timestamps_token_id,
            transcribe_id: cfg.transcribe_token_id,
        })
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        let filtered: Vec<u32> = if skip_special {
            ids.iter()
                .copied()
                .filter(|t| !self.all_special_ids.contains(t))
                .collect()
        } else {
            ids.to_vec()
        };
        self.inner
            .decode(&filtered, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Decode each id separately. Used for word-boundary detection (leading space).
    pub fn decode_pieces(&self, ids: &[u32]) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            out.push(self.decode(&[id], false)?);
        }
        Ok(out)
    }
}
