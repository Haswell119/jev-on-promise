//! Internal dataset record format (JSONL), shared by eval / train /
//! calibrate / leakage.
//!
//! ```json
//! {"id": "...", "source": "...", "license": "...", "split": "train", "tier": "internal",
//!  "family": "intent", "synthetic": true, "transformation": "paraphrase", "group": "g1",
//!  "request": {"model": "sextant-1", "state": ..., "questions": {"q": {...}}},
//!  "gold": {"q": "option_key" | 2 | true | false},
//!  "gold_probs": {"q": {"a": 0.7, "b": 0.3}}   // optional soft labels
//! }
//! ```

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sextant_core::SystemOneRequest;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub split: String,
    #[serde(default)]
    pub tier: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub synthetic: bool,
    #[serde(default)]
    pub transformation: String,
    #[serde(default)]
    pub group: Option<String>,
    pub request: SystemOneRequest,
    pub gold: IndexMap<String, Value>,
    #[serde(default)]
    pub gold_probs: Option<IndexMap<String, Value>>,
    #[serde(flatten)]
    pub extra: IndexMap<String, Value>,
}

impl Record {
    /// Value of a metadata field for grouping (top-level string fields).
    pub fn field(&self, name: &str) -> String {
        match name {
            "tier" => self.tier.clone(),
            "family" => self.family.clone(),
            "source" => self.source.clone(),
            "split" => self.split.clone(),
            "synthetic" => self.synthetic.to_string(),
            "transformation" => self.transformation.clone(),
            other => self
                .extra
                .get(other)
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                })
                .unwrap_or_default(),
        }
    }
}

pub fn load_records(paths: &[PathBuf]) -> Result<Vec<Record>, String> {
    let mut out = Vec::new();
    for p in super::common::expand_jsonl(paths) {
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let rec: Record = serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", p.display(), i + 1))?;
            out.push(rec);
        }
    }
    Ok(out)
}

pub fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut s = String::new();
    for r in rows {
        s.push_str(&serde_json::to_string(r).map_err(|e| e.to_string())?);
        s.push('\n');
    }
    std::fs::write(path, s).map_err(|e| format!("{}: {e}", path.display()))
}
