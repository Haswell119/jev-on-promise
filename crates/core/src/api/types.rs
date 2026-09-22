//! Serde types for the `/v1/systemone` contract.
//!
//! The shape intentionally mirrors the publicly documented System One request
//! contract (state + typed questions → typed answers) so that existing clients
//! can be pointed at a self-hosted engine. Names of models are our own.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Any JSON entry: string, object, array, number, bool or null.
pub type Entry = Value;

/// A request to evaluate one `state` against a map of typed questions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemOneRequest {
    /// Model identifier (see `GET /v1/models`).
    pub model: String,
    /// The evidence: a string, an object, or an array.
    pub state: Value,
    /// Questions keyed by caller-chosen ids. Ids never influence inference.
    pub questions: IndexMap<String, Question>,
    /// When true, every answer carries an `explain` block. Does not alter inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain: Option<bool>,
}

/// A typed question. `type` selects the primitive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul(NoulQuestion),
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
}

impl Question {
    pub fn kind(&self) -> QuestionKind {
        match self {
            Question::Noul(_) => QuestionKind::Noul,
            Question::Choice(_) => QuestionKind::Choice,
            Question::Score(_) => QuestionKind::Score,
        }
    }
    pub fn instructions(&self) -> &Value {
        match self {
            Question::Noul(q) => &q.instructions,
            Question::Choice(q) => &q.instructions,
            Question::Score(q) => &q.instructions,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    Noul,
    Choice,
    Score,
}

impl QuestionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionKind::Noul => "noul",
            QuestionKind::Choice => "choice",
            QuestionKind::Score => "score",
        }
    }
}

/// Binary judgment: P(proposition is true).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoulQuestion {
    #[serde(default)]
    pub instructions: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

/// Optional descriptions of what `true` and `false` mean.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<Value>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub no: Option<Value>,
}

/// Pick exactly one option from a closed set (2..=255 options).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChoiceQuestion {
    #[serde(default)]
    pub instructions: Value,
    /// option key → description (string | object | array | null).
    pub criteria: IndexMap<String, Value>,
}

/// Ordinal rating over 2..=10 ordered, described levels.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreQuestion {
    #[serde(default)]
    pub instructions: Value,
    /// Ordered level descriptions (index 0 = lowest level).
    pub criteria: Vec<Value>,
}

/// Response envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// A typed answer. `type` matches the question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

impl Answer {
    pub fn explain(&self) -> Option<&Explanation> {
        match self {
            Answer::Noul(a) => a.explain.as_ref(),
            Answer::Choice(a) => a.explain.as_ref(),
            Answer::Score(a) => a.explain.as_ref(),
        }
    }
    pub fn strip_explain(&mut self) {
        match self {
            Answer::Noul(a) => a.explain = None,
            Answer::Choice(a) => a.explain = None,
            Answer::Score(a) => a.explain = None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoulAnswer {
    /// Calibrated probability that the proposition is true.
    pub noul: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain: Option<Explanation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChoiceAnswer {
    /// The argmax option (ties broken by lexicographic key order).
    pub choice: String,
    /// Every option → calibrated probability; sums to 1.
    pub probabilities: IndexMap<String, f64>,
    /// Calibrated scalar confidence in [0, 1].
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain: Option<Explanation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreAnswer {
    /// Σ level_index × P(level).
    pub score: f64,
    /// Level index (as string) → level description (stringified).
    pub legend: IndexMap<String, String>,
    /// Level index (as string) → probability; sums to 1.
    pub probabilities: IndexMap<String, f64>,
    /// Calibrated scalar confidence in [0, 1].
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain: Option<Explanation>,
}

/// Optional inspection block. Never alters inference.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Explanation {
    /// Detected question family (heuristic archetype).
    pub family: String,
    /// Which resolver produced the answer: `symbolic:<name>` or `semantic`.
    pub path: String,
    /// The most relevant evidence fragments (path + text + retrieval score).
    pub top_evidence: Vec<EvidenceItem>,
    /// Per-option (or per-level / per-hypothesis) raw scores before calibration.
    pub raw_scores: IndexMap<String, f64>,
    /// Per-option feature vectors (name → value).
    pub features: IndexMap<String, IndexMap<String, f64>>,
    /// Calibration details (temperature, method, family bucket…).
    pub calibration: IndexMap<String, Value>,
    /// Free-form notes emitted by resolvers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceItem {
    pub path: String,
    pub text: String,
    pub score: f64,
}

/// Model card returned by `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub description: String,
    pub aliases: Vec<String>,
    pub max_choice_options: usize,
    pub max_score_levels: usize,
    pub neural: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}
