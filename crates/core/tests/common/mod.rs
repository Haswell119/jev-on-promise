//! Shared helpers for the integration test suites: one engine per test
//! binary (building one parses ~8 MB of embedded lexicon) and proptest
//! strategies that generate realistic *and* unusual requests.

#![allow(dead_code)]

use proptest::prelude::*;
use serde_json::{json, Map, Value};
use sextant_core::api::{ChoiceAnswer, NoulAnswer, ScoreAnswer};
use sextant_core::{Answer, Engine, SystemOneRequest, SystemOneResponse};
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

/// The shared engine for this test binary (embedded lexicon + embedded model).
pub fn engine() -> Arc<Engine> {
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE.get_or_init(|| Arc::new(Engine::default_embedded())).clone()
}

/// Deserialize a request from JSON (panics on malformed JSON: that is a test bug).
pub fn request(v: Value) -> SystemOneRequest {
    serde_json::from_value(v).expect("request JSON must deserialize")
}

/// Evaluate a request that must be accepted.
pub fn evaluate(v: Value) -> SystemOneResponse {
    let req = request(v);
    engine().evaluate(&req).unwrap_or_else(|e| panic!("evaluate failed: {e:?}"))
}

pub fn as_choice(a: &Answer) -> &ChoiceAnswer {
    match a {
        Answer::Choice(c) => c,
        other => panic!("expected a choice answer, got {other:?}"),
    }
}

pub fn as_score(a: &Answer) -> &ScoreAnswer {
    match a {
        Answer::Score(s) => s,
        other => panic!("expected a score answer, got {other:?}"),
    }
}

pub fn as_noul(a: &Answer) -> &NoulAnswer {
    match a {
        Answer::Noul(n) => n,
        other => panic!("expected a noul answer, got {other:?}"),
    }
}

/// Canonical JSON text of a response (used for bit-identity comparisons).
pub fn to_json(resp: &SystemOneResponse) -> String {
    serde_json::to_string(resp).expect("response serializes")
}

/// Every number in a response must be finite (no NaN / inf ever leaks out).
pub fn all_finite(resp: &SystemOneResponse) -> bool {
    resp.answers.values().all(|a| match a {
        Answer::Noul(n) => n.noul.is_finite(),
        Answer::Choice(c) => {
            c.confidence.is_finite() && c.probabilities.values().all(|p| p.is_finite())
        }
        Answer::Score(s) => {
            s.score.is_finite()
                && s.confidence.is_finite()
                && s.probabilities.values().all(|p| p.is_finite())
        }
    })
}

/// The documented argmax rule: highest probability, ties broken by the
/// lexicographically smallest key.
pub fn expected_argmax(p: &indexmap::IndexMap<String, f64>) -> String {
    let mut best: Option<(&str, f64)> = None;
    for (k, &v) in p {
        match best {
            None => best = Some((k, v)),
            Some((bk, bv)) => {
                if v > bv || (v == bv && k.as_str() < bk) {
                    best = Some((k, v));
                }
            }
        }
    }
    best.map(|(k, _)| k.to_string()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

pub const WORDS: &[&str] = &[
    "refund",
    "order",
    "package",
    "late",
    "invoice",
    "charged",
    "twice",
    "cancel",
    "shipping",
    "delivery",
    "angry",
    "urgent",
    "bug",
    "outage",
    "login",
    "password",
    "upgrade",
    "pricing",
    "return",
    "damaged",
    "tracking",
    "payment",
    "subscription",
    "account",
    "email",
    "support",
    "thanks",
    "please",
    "help",
    "not",
    "never",
    "yes",
    "no",
    "customer",
    "agent",
    "meeting",
    "budget",
    "weather",
    "recipe",
    "guitar",
    "volcano",
    "the",
    "a",
    "is",
    "was",
    "and",
    "but",
    "I",
    "we",
    "they",
    "3",
    "42",
    "$500",
    "12%",
    "2026-03-03",
    "March",
    "asap",
    "!!!",
    "ok",
    "fine",
    "broken",
    "excellent",
    "terrible",
];

/// Unicode-heavy samples: accents, CJK, Arabic and Hebrew (right-to-left),
/// emoji (including ZWJ sequences and flags), ligatures, smart punctuation,
/// zero-width characters and explicit bidi marks (all written as escapes).
pub const UNICODE_SAMPLES: &[&str] = &[
    "Caf\u{e9} na\u{ef}ve r\u{e9}sum\u{e9}",
    "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8}\u{3002}",
    "\u{645}\u{631}\u{62d}\u{628}\u{627} \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645}",
    "\u{5e9}\u{5dc}\u{5d5}\u{5dd} \u{5e2}\u{5d5}\u{5dc}\u{5dd}",
    "\u{1f680} launch \u{1f389} party \u{1f600} refund please \u{1f4b8}",
    "\u{dc}n\u{ef}c\u{f6}d\u{e9} \u{fb01} ligature\u{2026} \u{201c}smart quotes\u{201d} \u{2013} dashes \u{2014}",
    "\u{200b}zero\u{200d}width\u{feff}joiners\u{ad}soft-hyphen",
    "\u{395}\u{3bb}\u{3bb}\u{3b7}\u{3bd}\u{3b9}\u{3ba}\u{3ac} \u{3ba}\u{3b5}\u{3af}\u{3bc}\u{3b5}\u{3bd}\u{3bf}",
    "\u{440}\u{443}\u{441}\u{441}\u{43a}\u{438}\u{439} \u{442}\u{435}\u{43a}\u{441}\u{442}",
    "\u{939}\u{93f}\u{928}\u{94d}\u{926}\u{940} \u{92a}\u{93e}\u{920}",
    "\u{1d518}\u{1d52b}\u{1d526}\u{1d520}\u{1d52c}\u{1d521}\u{1d522} math",
    "flags \u{1f1fa}\u{1f1f8}\u{1f1ef}\u{1f1f5} and tones \u{1f44d}\u{1f3fd}\u{1f44d}\u{1f3ff}",
    "tab\tseparated\nnew\r\nlines",
    "bidi marks \u{200f}rtl\u{200e} ltr \u{202b}embedded\u{202c}",
];

pub const QUESTIONS: &[&str] = &[
    "Which team should handle this?",
    "What is the user's intent?",
    "Which option is the value of `carrier`?",
    "Which category best describes the text?",
    "Is the customer angry?",
    "Did the customer request a refund?",
    "Has the order shipped?",
    "How severe is the issue?",
    "How frustrated is the customer?",
    "Rate the quality of the response.",
    "Is the invoice total greater than $500?",
    "Was the order placed after March 3, 2026?",
    "Does `extracted_value` match `field` in `source_text`?",
    "Does this convey urgency?",
    "Is the reply consistent with the policy?",
    "",
    "?",
];

pub const KEYS: &[&str] = &[
    "subject",
    "body",
    "text",
    "message",
    "order",
    "items",
    "refund_requested",
    "customerName",
    "order.total",
    "amount",
    "status",
    "paid",
    "notes",
    "source_text",
    "extracted_value",
    "field",
    "priority",
    "tags",
    "messages",
    "date",
    "",
];

const PUNCT: &[&str] = &[" ", "\n", "\t\t", "...", "?!", "\"", "'", "`", "- ", ":"];

pub fn word() -> impl Strategy<Value = String> {
    prop::sample::select(WORDS).prop_map(String::from)
}

pub fn sentence() -> impl Strategy<Value = String> {
    prop::collection::vec(word(), 0..12).prop_map(|ws| ws.join(" "))
}

/// Text that is sometimes realistic, sometimes unicode-heavy, sometimes empty.
pub fn text() -> impl Strategy<Value = String> {
    prop_oneof![
        5 => sentence(),
        2 => prop::sample::select(UNICODE_SAMPLES).prop_map(String::from),
        1 => "[ -~]{0,40}",
        1 => "\\PC{0,24}",
        1 => Just(String::new()),
        1 => prop::sample::select(PUNCT).prop_map(String::from),
    ]
}

pub fn key() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => prop::sample::select(KEYS).prop_map(String::from),
        1 => "[a-z_]{1,12}",
        1 => "\\PC{0,8}",
    ]
}

fn leaf() -> impl Strategy<Value = Value> {
    prop_oneof![
        5 => text().prop_map(Value::String),
        1 => any::<i32>().prop_map(|n| json!(n)),
        1 => (-1.0e6f64..1.0e6f64).prop_map(|x| json!(x)),
        1 => any::<bool>().prop_map(Value::Bool),
        1 => Just(Value::Null),
    ]
}

fn object_of(inner: impl Strategy<Value = Value>) -> impl Strategy<Value = Value> {
    prop::collection::vec((key(), inner), 0..6).prop_map(|kv| {
        let mut m = Map::new();
        for (k, v) in kv {
            m.insert(k, v);
        }
        Value::Object(m)
    })
}

/// A valid state: string, object or array, nested up to 4 levels.
pub fn state() -> impl Strategy<Value = Value> {
    let nested = leaf().prop_recursive(3, 32, 5, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..5).prop_map(Value::Array),
            object_of(inner),
        ]
    });
    prop_oneof![
        3 => text().prop_map(Value::String),
        2 => prop::collection::vec(nested.clone(), 0..5).prop_map(Value::Array),
        3 => object_of(nested),
    ]
}

/// A criterion description: string, null, object (with example lists and
/// exclusion fields), array or a nested object.
pub fn description() -> impl Strategy<Value = Value> {
    prop_oneof![
        4 => sentence().prop_map(Value::String),
        1 => Just(Value::Null),
        2 => (sentence(), prop::collection::vec(sentence(), 0..3), prop::option::of(sentence()))
            .prop_map(|(what, examples, not_for)| {
                let mut m = json!({ "what": what, "examples": examples });
                if let Some(n) = not_for {
                    m["not_for"] = json!(n);
                }
                m
            }),
        1 => prop::collection::vec(sentence(), 0..4)
            .prop_map(|v| Value::Array(v.into_iter().map(Value::String).collect())),
        1 => (sentence(), any::<bool>(), any::<i32>()).prop_map(|(d, b, n)| json!({
            "description": d,
            "refund": b,
            "count": n,
            "nested": { "detail": d, "unicode": "\u{65e5}\u{672c}\u{8a9e} \u{1f680}" }
        })),
    ]
}

/// Instructions: a known question, a generated sentence, a structured
/// instruction object or arbitrary text. Never null (Noul requires it).
pub fn instructions() -> impl Strategy<Value = Value> {
    prop_oneof![
        4 => prop::sample::select(QUESTIONS).prop_map(|s| Value::String(s.to_string())),
        2 => sentence().prop_map(Value::String),
        1 => (prop::sample::select(QUESTIONS), sentence())
            .prop_map(|(q, c)| json!({ "question": q, "context": c })),
        1 => text().prop_map(Value::String),
    ]
}

fn make_unique(keys: Vec<String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    keys.into_iter()
        .enumerate()
        .map(|(i, k)| {
            let mut k = if k.trim().is_empty() { format!("opt{i}") } else { k };
            while !seen.insert(k.clone()) {
                k = format!("{k}_{i}");
            }
            k
        })
        .collect()
}

/// `n` distinct, non-empty option keys (words, ASCII identifiers, unicode).
pub fn option_keys(n: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(
        prop_oneof![
            3 => word(),
            1 => "[a-zA-Z0-9_ -]{1,16}",
            1 => "\\PC{1,10}",
        ],
        n,
    )
    .prop_map(make_unique)
}

/// A Choice question with an option count drawn from `n`.
pub fn choice_question(n: impl Strategy<Value = usize>) -> impl Strategy<Value = Value> {
    n.prop_flat_map(|n| (instructions(), option_keys(n), prop::collection::vec(description(), n)))
        .prop_map(|(ins, keys, descs)| {
            let mut criteria = Map::new();
            for (k, d) in keys.into_iter().zip(descs) {
                criteria.insert(k, d);
            }
            json!({ "type": "choice", "instructions": ins, "criteria": criteria })
        })
}

/// A Score question with a level count drawn from `k`.
pub fn score_question(k: impl Strategy<Value = usize>) -> impl Strategy<Value = Value> {
    k.prop_flat_map(|k| (instructions(), prop::collection::vec(description(), k)))
        .prop_map(|(ins, levels)| json!({ "type": "score", "instructions": ins, "criteria": levels }))
}

/// A Noul question, optionally with true/false criteria.
pub fn noul_question() -> impl Strategy<Value = Value> {
    (
        instructions(),
        prop::option::of((prop::option::of(description()), prop::option::of(description()))),
    )
        .prop_map(|(ins, criteria)| {
            let mut q = json!({ "type": "noul", "instructions": ins });
            if let Some((yes, no)) = criteria {
                let mut c = Map::new();
                if let Some(y) = yes {
                    c.insert("true".into(), y);
                }
                if let Some(n) = no {
                    c.insert("false".into(), n);
                }
                q["criteria"] = Value::Object(c);
            }
            q
        })
}

/// A question of any kind.
pub fn any_question() -> impl Strategy<Value = Value> {
    prop_oneof![choice_question(2usize..=12), score_question(2usize..=10), noul_question()]
}

/// Wrap a single question into a request (id `q`).
pub fn single(state: Value, question: Value) -> Value {
    json!({ "model": "sextant-1", "state": state, "questions": { "q": question } })
}
