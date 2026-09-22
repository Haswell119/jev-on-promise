//! Cheap deterministic question-family detector. Families only select a
//! weight profile (mixture of linear experts) and never restrict what a
//! question may ask; unknown questions use the generic profile.

use crate::api::QuestionKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Generic,
    EnumExtraction,
    Intent,
    Routing,
    Topic,
    Policy,
    FactualYesNo,
    Adequacy,
    Sentiment,
    Severity,
    RequestDetection,
    Relevance,
    Compatibility,
    Similarity,
    Numeric,
    Temporal,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Generic => "generic",
            Family::EnumExtraction => "enum_extraction",
            Family::Intent => "intent",
            Family::Routing => "routing",
            Family::Topic => "topic",
            Family::Policy => "policy",
            Family::FactualYesNo => "factual_yes_no",
            Family::Adequacy => "adequacy",
            Family::Sentiment => "sentiment",
            Family::Severity => "severity",
            Family::RequestDetection => "request_detection",
            Family::Relevance => "relevance",
            Family::Compatibility => "compatibility",
            Family::Similarity => "similarity",
            Family::Numeric => "numeric",
            Family::Temporal => "temporal",
        }
    }

    pub fn all() -> &'static [Family] {
        &[
            Family::Generic,
            Family::EnumExtraction,
            Family::Intent,
            Family::Routing,
            Family::Topic,
            Family::Policy,
            Family::FactualYesNo,
            Family::Adequacy,
            Family::Sentiment,
            Family::Severity,
            Family::RequestDetection,
            Family::Relevance,
            Family::Compatibility,
            Family::Similarity,
            Family::Numeric,
            Family::Temporal,
        ]
    }

    pub fn from_str_opt(s: &str) -> Option<Family> {
        Family::all().iter().copied().find(|f| f.as_str() == s)
    }
}

fn any(text: &str, words: &[&str]) -> bool {
    words.iter().any(|w| text.contains(w))
}

/// Detect the family from the lowercase instruction text, the question kind
/// and simple criteria statistics.
pub fn detect(kind: QuestionKind, instr: &str, n_options: usize, options_are_literal: bool, level_words: &str) -> Family {
    let t = instr;
    let sentiment = any(t, &["sentiment", "tone", "emotion", "feel", "mood", "attitude", "frustrat", "angry", "anger", "happy", "satisf", "polite", "rude", "toxic", "hostil", "friendly", "positive", "negative", "upset", "pleased", "complain"]);
    let severity = any(t, &["severity", "severe", "urgen", "priority", "how bad", "risk", "impact", "critical", "how serious", "escalat", "how strongly", "how much", "how likely", "degree", "extent", "how well", "quality", "rate ", "rating", "how good", "how helpful", "how verbose", "how long"]);
    let numeric = any(t, &["how many", "how much", "amount", "total", "count", "number of", "greater than", "less than", "more than", "at least", "at most", "exceed", "percent", "price", "cost", "sum", "average", "larger", "smaller", "compare"]);
    let temporal = any(t, &["date", "deadline", "before", "after", "expire", "due ", "overdue", "within", "days", "weeks", "months", "year", "when ", "time ", "recent", "old", "late", "on time", "schedule"]);
    let extraction = any(t, &["extract", "value of", "which option is the value", "which of these is the", "correct answer", "the answer to", "name of", "appears in", "which value", "verbatim", "select the span", "which option matches", "exact value", "literal"]);
    let intent = any(t, &["intent", "what does the user want", "trying to", "purpose", "goal of", "wants to", "asking for", "request type", "what is the user"]);
    let routing = any(t, &["route", "routing", "team", "department", "handle", "assign", "who should", "queue", "escalate to", "which agent", "which handler", "specialist", "tier"]);
    let topic = any(t, &["topic", "category", "categor", "classif", "subject", "about", "domain", "genre", "type of", "kind of", "what is this", "which section", "label"]);
    let policy = any(t, &["policy", "rule", "comply", "complian", "violat", "allowed", "permitted", "eligib", "applies", "applicable", "guideline", "terms", "regulation", "legal", "law", "required by", "prohibit", "must ", "may the", "is it permissible", "qualif", "entitled", "approve", "approved", "cover"]);
    let adequacy = any(t, &["adequate", "sufficient", "answer the question", "answers the question", "address", "resolve", "resolved", "complete answer", "fully answer", "correct answer", "acceptable", "helpful", "respond", "response", "reply", "solution", "solves"]);
    let relevance = any(t, &["relevant", "relevance", "related to", "on topic", "off topic", "pertain", "about the", "support the claim", "supports", "cites", "evidence for", "mention"]);
    let compat = any(t, &["consistent", "contradict", "entail", "match", "matches", "same person", "same product", "duplicate", "agree", "compatible", "conflict", "relationship of", "follows from", "implied", "true given", "same entity", "refer to the same"]);
    let similarity = any(t, &["similar", "paraphrase", "equivalent", "same meaning", "mean the same", "how close", "resemble"]);
    let request = any(t, &["request", "asks for", "asking", "demand", "wants a", "want a", "requesting", "is the user asking", "does the customer ask", "does the user ask", "ask to"]);

    match kind {
        QuestionKind::Score => {
            if numeric && !sentiment {
                return Family::Numeric;
            }
            if sentiment {
                return Family::Sentiment;
            }
            if similarity {
                return Family::Similarity;
            }
            if adequacy {
                return Family::Adequacy;
            }
            if severity || any(level_words, &["severe", "critical", "minor", "major", "urgent", "low", "high", "medium"]) {
                return Family::Severity;
            }
            if policy {
                return Family::Policy;
            }
            Family::Severity
        }
        QuestionKind::Noul => {
            if compat {
                return Family::Compatibility;
            }
            if request {
                return Family::RequestDetection;
            }
            if sentiment {
                return Family::Sentiment;
            }
            if policy {
                return Family::Policy;
            }
            if adequacy {
                return Family::Adequacy;
            }
            if relevance {
                return Family::Relevance;
            }
            if numeric {
                return Family::Numeric;
            }
            if temporal {
                return Family::Temporal;
            }
            if similarity {
                return Family::Similarity;
            }
            Family::FactualYesNo
        }
        QuestionKind::Choice => {
            if options_are_literal {
                return Family::EnumExtraction;
            }
            if compat {
                return Family::Compatibility;
            }
            if extraction {
                return Family::EnumExtraction;
            }
            if sentiment {
                return Family::Sentiment;
            }
            if routing {
                return Family::Routing;
            }
            if intent {
                return Family::Intent;
            }
            if policy {
                return Family::Policy;
            }
            if adequacy {
                return Family::Adequacy;
            }
            if similarity {
                return Family::Similarity;
            }
            if numeric && !topic {
                return Family::Numeric;
            }
            if temporal && !topic {
                return Family::Temporal;
            }
            if topic || n_options >= 8 {
                return Family::Topic;
            }
            Family::Generic
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_families() {
        assert_eq!(detect(QuestionKind::Choice, "which team should handle this request?", 3, false, ""), Family::Routing);
        assert_eq!(detect(QuestionKind::Choice, "which intent does the user's message express?", 5, false, ""), Family::Intent);
        assert_eq!(detect(QuestionKind::Choice, "which option is the value of `field` in `source_text`?", 5, true, ""), Family::EnumExtraction);
        assert_eq!(detect(QuestionKind::Score, "how frustrated is the customer?", 3, false, "calm frustrated angry"), Family::Sentiment);
        assert_eq!(detect(QuestionKind::Noul, "does this convey urgency?", 2, false, ""), Family::FactualYesNo);
        assert_eq!(detect(QuestionKind::Noul, "did the customer request a refund?", 2, false, ""), Family::RequestDetection);
        assert_eq!(detect(QuestionKind::Choice, "what is the relationship of `hypothesis` to `premise`?", 3, false, ""), Family::Compatibility);
    }
}
