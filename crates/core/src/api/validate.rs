//! Schema validation. Runs before any semantic processing so malformed
//! requests are rejected cheaply with a precise field path.

use super::{ApiError, Question, SystemOneRequest};
use serde_json::Value;

/// Hard limits enforced on every request. Configurable at server start.
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_questions: usize,
    pub max_choice_options: usize,
    pub min_choice_options: usize,
    pub max_score_levels: usize,
    pub min_score_levels: usize,
    /// Maximum serialized size of the `state` in bytes.
    pub max_state_bytes: usize,
    /// Maximum serialized size of a single question in bytes.
    pub max_question_bytes: usize,
    /// Maximum nesting depth of any JSON value.
    pub max_depth: usize,
    /// Maximum length of an option key.
    pub max_key_len: usize,
    /// Maximum total number of JSON leaves in the state.
    pub max_state_leaves: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_questions: 1024,
            max_choice_options: 255,
            min_choice_options: 2,
            max_score_levels: 10,
            min_score_levels: 2,
            max_state_bytes: 2 * 1024 * 1024,
            max_question_bytes: 256 * 1024,
            max_depth: 64,
            max_key_len: 512,
            max_state_leaves: 50_000,
        }
    }
}

pub const MODEL_ID: &str = "sextant-1";
pub const MODEL_ALIASES: &[&str] = &["sextant-latest", "sextant"];

pub fn is_known_model(name: &str) -> bool {
    name == MODEL_ID || MODEL_ALIASES.contains(&name)
}

fn depth(v: &Value) -> usize {
    match v {
        Value::Array(a) => 1 + a.iter().map(depth).max().unwrap_or(0),
        Value::Object(o) => 1 + o.values().map(depth).max().unwrap_or(0),
        _ => 0,
    }
}

fn leaves(v: &Value) -> usize {
    match v {
        Value::Array(a) => a.iter().map(leaves).sum::<usize>().max(1),
        Value::Object(o) => o.values().map(leaves).sum::<usize>().max(1),
        _ => 1,
    }
}

fn approx_bytes(v: &Value) -> usize {
    // Cheap size estimate without re-serializing: sum of string lengths + structure.
    match v {
        Value::Null => 4,
        Value::Bool(_) => 5,
        Value::Number(_) => 12,
        Value::String(s) => s.len() + 2,
        Value::Array(a) => 2 + a.iter().map(|x| approx_bytes(x) + 1).sum::<usize>(),
        Value::Object(o) => {
            2 + o
                .iter()
                .map(|(k, x)| k.len() + 4 + approx_bytes(x))
                .sum::<usize>()
        }
    }
}

fn validate_entry(v: &Value, field: &str, limits: &Limits) -> Result<(), ApiError> {
    if depth(v) > limits.max_depth {
        return Err(ApiError::invalid(field, format!("nesting deeper than {} levels", limits.max_depth)));
    }
    if approx_bytes(v) > limits.max_question_bytes {
        return Err(ApiError::invalid(field, format!("larger than {} bytes", limits.max_question_bytes)));
    }
    Ok(())
}

/// Validate the full request. Returns the first problem found.
pub fn validate_request(req: &SystemOneRequest, limits: &Limits) -> Result<(), ApiError> {
    if req.model.is_empty() {
        return Err(ApiError::invalid("model", "model must be a non-empty string"));
    }
    if !is_known_model(&req.model) {
        return Err(ApiError::unknown_model(&req.model));
    }
    match &req.state {
        Value::String(_) | Value::Object(_) | Value::Array(_) => {}
        Value::Null => return Err(ApiError::invalid("state", "state must be a string, object or array (got null)")),
        _ => return Err(ApiError::invalid("state", "state must be a string, object or array")),
    }
    if depth(&req.state) > limits.max_depth {
        return Err(ApiError::invalid("state", format!("state nested deeper than {} levels", limits.max_depth)));
    }
    let bytes = approx_bytes(&req.state);
    if bytes > limits.max_state_bytes {
        return Err(ApiError::too_large(format!(
            "state is ~{} bytes; the limit is {} bytes",
            bytes, limits.max_state_bytes
        )));
    }
    if leaves(&req.state) > limits.max_state_leaves {
        return Err(ApiError::too_large(format!(
            "state has more than {} leaves",
            limits.max_state_leaves
        )));
    }
    if req.questions.is_empty() {
        return Err(ApiError::invalid("questions", "questions must contain at least one question"));
    }
    if req.questions.len() > limits.max_questions {
        return Err(ApiError::invalid(
            "questions",
            format!("at most {} questions per request", limits.max_questions),
        ));
    }
    for (id, q) in &req.questions {
        if id.is_empty() {
            return Err(ApiError::invalid("questions", "question ids must be non-empty strings"));
        }
        if id.len() > limits.max_key_len {
            return Err(ApiError::invalid(
                format!("questions.{id}"),
                format!("question id longer than {} bytes", limits.max_key_len),
            ));
        }
        let base = format!("questions.{id}");
        validate_entry(q.instructions(), &format!("{base}.instructions"), limits)?;
        match q {
            Question::Noul(n) => {
                if let Some(c) = &n.criteria {
                    if let Some(v) = &c.yes {
                        validate_entry(v, &format!("{base}.criteria.true"), limits)?;
                    }
                    if let Some(v) = &c.no {
                        validate_entry(v, &format!("{base}.criteria.false"), limits)?;
                    }
                }
                if n.instructions.is_null() && n.criteria.as_ref().map(|c| c.yes.is_none() && c.no.is_none()).unwrap_or(true) {
                    return Err(ApiError::invalid(
                        format!("{base}.instructions"),
                        "noul questions need instructions (or true/false criteria)",
                    ));
                }
            }
            Question::Choice(c) => {
                let n = c.criteria.len();
                if n < limits.min_choice_options {
                    return Err(ApiError::invalid(
                        format!("{base}.criteria"),
                        format!("choice needs at least {} options (got {n})", limits.min_choice_options),
                    ));
                }
                if n > limits.max_choice_options {
                    return Err(ApiError::invalid(
                        format!("{base}.criteria"),
                        format!("choice allows at most {} options (got {n})", limits.max_choice_options),
                    ));
                }
                for (k, v) in &c.criteria {
                    if k.is_empty() {
                        return Err(ApiError::invalid(format!("{base}.criteria"), "option keys must be non-empty"));
                    }
                    if k.len() > limits.max_key_len {
                        return Err(ApiError::invalid(
                            format!("{base}.criteria"),
                            format!("option key longer than {} bytes", limits.max_key_len),
                        ));
                    }
                    validate_entry(v, &format!("{base}.criteria.{k}"), limits)?;
                }
            }
            Question::Score(s) => {
                let n = s.criteria.len();
                if n < limits.min_score_levels {
                    return Err(ApiError::invalid(
                        format!("{base}.criteria"),
                        format!("score needs at least {} levels (got {n})", limits.min_score_levels),
                    ));
                }
                if n > limits.max_score_levels {
                    return Err(ApiError::invalid(
                        format!("{base}.criteria"),
                        format!("score allows at most {} levels (got {n})", limits.max_score_levels),
                    ));
                }
                for (i, v) in s.criteria.iter().enumerate() {
                    validate_entry(v, &format!("{base}.criteria[{i}]"), limits)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(v: serde_json::Value) -> SystemOneRequest {
        serde_json::from_value(v).expect("parse")
    }

    #[test]
    fn accepts_minimal_valid_request() {
        let r = req(json!({
            "model": "sextant-1",
            "state": "hello",
            "questions": {"q": {"type": "noul", "instructions": "Is it a greeting?"}}
        }));
        assert!(validate_request(&r, &Limits::default()).is_ok());
    }

    #[test]
    fn rejects_unknown_model_and_bad_cardinalities() {
        let r = req(json!({"model": "gpt-x", "state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}));
        assert_eq!(validate_request(&r, &Limits::default()).unwrap_err().code, "unknown_model");
        let r = req(json!({"model": "sextant-1", "state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": null}}}}));
        assert_eq!(validate_request(&r, &Limits::default()).unwrap_err().status, 422);
        let levels: Vec<String> = (0..11).map(|i| format!("l{i}")).collect();
        let r = req(json!({"model": "sextant-1", "state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": levels}}}));
        assert_eq!(validate_request(&r, &Limits::default()).unwrap_err().status, 422);
    }

    #[test]
    fn rejects_bad_state_and_depth() {
        let r = req(json!({"model": "sextant-1", "state": 12, "questions": {"q": {"type": "noul", "instructions": "?"}}}));
        assert!(validate_request(&r, &Limits::default()).is_err());
        let mut v = json!("leaf");
        for _ in 0..70 {
            v = json!([v]);
        }
        let r = req(json!({"model": "sextant-1", "state": v, "questions": {"q": {"type": "noul", "instructions": "?"}}}));
        assert!(validate_request(&r, &Limits::default()).is_err());
    }

    #[test]
    fn unknown_question_type_fails_to_parse() {
        let v = json!({"model": "sextant-1", "state": "x", "questions": {"q": {"type": "regress", "instructions": "?"}}});
        assert!(serde_json::from_value::<SystemOneRequest>(v).is_err());
    }
}
