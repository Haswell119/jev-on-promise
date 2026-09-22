//! JSON → flat fields with dotted paths (`ticket.messages[0].text`).
//! Structure is preserved: every leaf keeps its path, its key tokens and
//! its scalar value, so questions can reference paths and symbolic
//! resolvers can compare values.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    Text,
    Number,
    Bool,
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlatField {
    /// Dotted path; empty for a top-level string state.
    pub path: String,
    /// Last key of the path (without array indices), e.g. `text`.
    pub key: String,
    /// The stringified value.
    pub text: String,
    pub kind: FieldKind,
    pub number: Option<f64>,
    pub boolean: Option<bool>,
    pub depth: u8,
    /// Index in the parent array when the parent is an array.
    pub array_index: Option<u32>,
}

fn push_key(path: &mut String, key: &str) {
    if !path.is_empty() {
        path.push('.');
    }
    path.push_str(key);
}

/// An array in the state: its path, its last key and its length.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayInfo {
    pub path: String,
    pub key: String,
    pub len: usize,
}

fn walk(v: &Value, path: &mut String, key: &str, depth: u8, array_index: Option<u32>, out: &mut Vec<FlatField>, arrays: &mut Vec<ArrayInfo>) {
    match v {
        Value::Object(map) => {
            for (k, child) in map {
                let saved = path.len();
                push_key(path, k);
                walk(child, path, k, depth.saturating_add(1), None, out, arrays);
                path.truncate(saved);
            }
        }
        Value::Array(items) => {
            arrays.push(ArrayInfo { path: path.clone(), key: key.to_string(), len: items.len() });
            for (i, child) in items.iter().enumerate() {
                let saved = path.len();
                path.push_str(&format!("[{i}]"));
                walk(child, path, key, depth.saturating_add(1), Some(i as u32), out, arrays);
                path.truncate(saved);
            }
        }
        Value::String(s) => out.push(FlatField {
            path: path.clone(),
            key: key.to_string(),
            text: s.clone(),
            kind: FieldKind::Text,
            number: None,
            boolean: None,
            depth,
            array_index,
        }),
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(0.0);
            out.push(FlatField {
                path: path.clone(),
                key: key.to_string(),
                text: n.to_string(),
                kind: FieldKind::Number,
                number: Some(f),
                boolean: None,
                depth,
                array_index,
            })
        }
        Value::Bool(b) => out.push(FlatField {
            path: path.clone(),
            key: key.to_string(),
            text: if *b { "true".into() } else { "false".into() },
            kind: FieldKind::Bool,
            number: None,
            boolean: Some(*b),
            depth,
            array_index,
        }),
        Value::Null => out.push(FlatField {
            path: path.clone(),
            key: key.to_string(),
            text: "null".into(),
            kind: FieldKind::Null,
            number: None,
            boolean: None,
            depth,
            array_index,
        }),
    }
}

/// Flatten a state value. A plain string becomes a single field with an
/// empty path.
pub fn flatten_state(state: &Value) -> Vec<FlatField> {
    flatten_state_with_arrays(state).0
}

/// Flatten and also report every array with its length.
pub fn flatten_state_with_arrays(state: &Value) -> (Vec<FlatField>, Vec<ArrayInfo>) {
    let mut out = Vec::new();
    let mut arrays = Vec::new();
    let mut path = String::new();
    walk(state, &mut path, "", 0, None, &mut out, &mut arrays);
    (out, arrays)
}

/// Split a JSON key into words: snake_case, kebab-case, camelCase, dotted.
pub fn split_key_words(key: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = key.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '_' || c == '-' || c == '.' || c == ' ' || c == '/' || c == ':' {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if c.is_uppercase() && i > 0 && chars[i - 1].is_lowercase() {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
        }
        if c.is_ascii_digit() && i > 0 && chars[i - 1].is_alphabetic() {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
        }
        cur.push(c.to_ascii_lowercase());
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flattens_with_paths() {
        let v = json!({"ticket": {"messages": [{"text": "hi"}, {"text": "bye"}], "amount": 12.5, "paid": false, "note": null}});
        let f = flatten_state(&v);
        let paths: Vec<&str> = f.iter().map(|x| x.path.as_str()).collect();
        assert_eq!(paths, vec!["ticket.messages[0].text", "ticket.messages[1].text", "ticket.amount", "ticket.paid", "ticket.note"]);
        assert_eq!(f[2].number, Some(12.5));
        assert_eq!(f[3].boolean, Some(false));
        assert_eq!(f[4].kind, FieldKind::Null);
        assert_eq!(f[0].key, "text");
        let direct = flatten_state(&json!({"tags": ["a", "b"]}));
        assert_eq!(direct[1].array_index, Some(1));
        assert_eq!(direct[1].key, "tags");
    }

    #[test]
    fn string_state() {
        let f = flatten_state(&json!("hello"));
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].path, "");
    }

    #[test]
    fn key_splitting() {
        assert_eq!(split_key_words("refund_requested"), vec!["refund", "requested"]);
        assert_eq!(split_key_words("orderTotalUSD"), vec!["order", "total", "usd"]);
        assert_eq!(split_key_words("item-count2"), vec!["item", "count", "2"]);
    }
}
