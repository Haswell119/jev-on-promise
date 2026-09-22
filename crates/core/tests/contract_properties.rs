//! Contract properties of the three primitives over generated inputs.
//!
//! For ANY well-formed request (unicode / RTL / emoji / empty-ish strings,
//! nested objects and arrays, structured descriptions, 2..=60 options and a
//! few explicit 255-option cases, 2..=10 score levels) the engine must
//! produce answers that satisfy the documented contract:
//!
//! * Choice: probability keys == option keys, every p in [0, 1], sum p = 1,
//!   `choice` is the argmax with lexicographic tie-break, 0 <= confidence <= 1;
//! * Score: one probability per level keyed "0".."K-1", score = sum k*p,
//!   0 <= score <= K-1, legend keys match;
//! * Noul: 0 <= noul <= 1;
//! * never panics; every response serializes to JSON and round-trips.

mod common;

use common::*;
use proptest::prelude::*;
use serde_json::{json, Value};
use sextant_core::{Answer, QuestionKind, SystemOneResponse};
use std::collections::BTreeSet;

const EPS: f64 = 1e-9;

/// Serialize -> parse -> compare (numbers within one ulp of serde_json's
/// float parser, everything else identical).
fn check_roundtrip(resp: &SystemOneResponse) -> Result<(), TestCaseError> {
    if let Some(diff) = roundtrip_error(resp) {
        return Err(TestCaseError::fail(format!("JSON round trip: {diff}")));
    }
    Ok(())
}

fn check_choice(state: &Value, question: &Value) -> Result<(), TestCaseError> {
    let criteria = question["criteria"].as_object().expect("choice criteria");
    let n = criteria.len();
    let req = request(single(state.clone(), question.clone()));
    let resp = engine().evaluate(&req);
    prop_assert!(resp.is_ok(), "valid request rejected: {:?}", resp.err());
    let resp = resp.unwrap();
    prop_assert!(all_finite(&resp));
    let a = as_choice(&resp.answers["q"]);

    prop_assert_eq!(a.probabilities.len(), n, "one probability per option");
    let expected_keys: BTreeSet<&str> = criteria.keys().map(|k| k.as_str()).collect();
    let got_keys: BTreeSet<&str> = a.probabilities.keys().map(|k| k.as_str()).collect();
    prop_assert_eq!(got_keys, expected_keys, "probability keys == option keys");
    // Insertion order is preserved too (nice for clients, not strictly required).
    prop_assert!(a.probabilities.keys().eq(criteria.keys()));

    for (k, p) in &a.probabilities {
        prop_assert!((0.0..=1.0).contains(p), "p[{}] = {} out of [0, 1]", k, p);
    }
    let sum: f64 = a.probabilities.values().sum();
    prop_assert!((sum - 1.0).abs() < EPS, "sum p = {}", sum);
    prop_assert_eq!(&a.choice, &expected_argmax(&a.probabilities));
    prop_assert!(criteria.contains_key(&a.choice));
    prop_assert!((0.0..=1.0).contains(&a.confidence), "confidence {}", a.confidence);
    check_roundtrip(&resp)
}

fn check_score(state: &Value, question: &Value) -> Result<(), TestCaseError> {
    let levels = question["criteria"].as_array().expect("score levels");
    let k = levels.len();
    let req = request(single(state.clone(), question.clone()));
    let resp = engine().evaluate(&req);
    prop_assert!(resp.is_ok(), "valid request rejected: {:?}", resp.err());
    let resp = resp.unwrap();
    prop_assert!(all_finite(&resp));
    let a = as_score(&resp.answers["q"]);

    prop_assert_eq!(a.probabilities.len(), k);
    let expected: Vec<String> = (0..k).map(|i| i.to_string()).collect();
    let got: Vec<&String> = a.probabilities.keys().collect();
    prop_assert_eq!(got, expected.iter().collect::<Vec<_>>(), "keys are 0..K-1 in order");
    let legend: Vec<&String> = a.legend.keys().collect();
    prop_assert_eq!(legend, expected.iter().collect::<Vec<_>>(), "legend keys match");
    for (i, level) in levels.iter().enumerate() {
        if let Value::String(s) = level {
            prop_assert_eq!(&a.legend[&i.to_string()], s, "string levels are echoed verbatim");
        }
    }
    for (key, p) in &a.probabilities {
        prop_assert!((0.0..=1.0).contains(p), "p[{}] = {}", key, p);
    }
    let sum: f64 = a.probabilities.values().sum();
    prop_assert!((sum - 1.0).abs() < EPS, "sum p = {}", sum);
    let expected_score: f64 = a.probabilities.iter().map(|(key, p)| key.parse::<f64>().unwrap() * p).sum();
    prop_assert!((a.score - expected_score).abs() < EPS, "score {} != sum k*p {}", a.score, expected_score);
    prop_assert!(a.score >= 0.0 && a.score <= (k - 1) as f64, "score {} outside [0, {}]", a.score, k - 1);
    prop_assert!((0.0..=1.0).contains(&a.confidence));
    check_roundtrip(&resp)
}

fn check_noul(state: &Value, question: &Value) -> Result<(), TestCaseError> {
    let req = request(single(state.clone(), question.clone()));
    let resp = engine().evaluate(&req);
    prop_assert!(resp.is_ok(), "valid request rejected: {:?}", resp.err());
    let resp = resp.unwrap();
    prop_assert!(all_finite(&resp));
    let a = as_noul(&resp.answers["q"]);
    prop_assert!((0.0..=1.0).contains(&a.noul), "noul = {}", a.noul);
    check_roundtrip(&resp)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn choice_contract_holds(state in state(), q in choice_question(2usize..=60)) {
        check_choice(&state, &q)?;
    }

    #[test]
    fn choice_contract_holds_for_small_option_sets(state in state(), q in choice_question(2usize..=4)) {
        check_choice(&state, &q)?;
    }

    #[test]
    fn score_contract_holds(state in state(), q in score_question(2usize..=10)) {
        check_score(&state, &q)?;
    }

    #[test]
    fn noul_contract_holds(state in state(), q in noul_question()) {
        check_noul(&state, &q)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    /// Several questions of mixed kinds in one request, with explain on or
    /// off: never panics, every answer keeps its contract, JSON round-trips.
    #[test]
    fn mixed_requests_never_panic_and_roundtrip(
        state in state(),
        qs in prop::collection::vec(any_question(), 1..6),
        explain in any::<bool>(),
    ) {
        let mut questions = serde_json::Map::new();
        for (i, q) in qs.iter().enumerate() {
            questions.insert(format!("q{i}"), q.clone());
        }
        let req = request(json!({
            "model": "sextant-1", "state": state, "questions": questions, "explain": explain
        }));
        let resp = engine().evaluate(&req);
        prop_assert!(resp.is_ok(), "{:?}", resp.err());
        let resp = resp.unwrap();
        prop_assert_eq!(resp.answers.len(), qs.len());
        prop_assert!(all_finite(&resp));
        for (id, a) in &resp.answers {
            let kind = req.questions[id].kind();
            prop_assert_eq!(a.explain().is_some(), explain, "explain block presence follows the flag");
            match a {
                Answer::Choice(c) => {
                    prop_assert_eq!(kind, QuestionKind::Choice);
                    let sum: f64 = c.probabilities.values().sum();
                    prop_assert!((sum - 1.0).abs() < EPS);
                    prop_assert_eq!(&c.choice, &expected_argmax(&c.probabilities));
                }
                Answer::Score(s) => {
                    prop_assert_eq!(kind, QuestionKind::Score);
                    let sum: f64 = s.probabilities.values().sum();
                    prop_assert!((sum - 1.0).abs() < EPS);
                }
                Answer::Noul(n) => {
                    prop_assert_eq!(kind, QuestionKind::Noul);
                    prop_assert!((0.0..=1.0).contains(&n.noul));
                }
            }
        }
        check_roundtrip(&resp)?;
    }
}

/// Explicit cases at the maximum option count (255).
#[test]
fn choice_with_255_options_satisfies_contract() {
    let states = [
        json!("Help! My payouts have been failing for 3 days. I need this fixed today."),
        json!({
            "subject": "opt_042",
            "body": "The value is opt_200 and nothing else.",
            "tags": ["\u{65e5}\u{672c}\u{8a9e}", "\u{1f680}"]
        }),
        json!(""),
    ];
    let descriptions = [
        json!("Payments, invoicing, refunds, payouts"),
        json!(null),
        json!({
            "what": "Bugs and outages",
            "examples": ["the app crashes", "login fails"],
            "not_for": "pricing"
        }),
        json!(["a list", "of phrases"]),
    ];
    for (si, state) in states.iter().enumerate() {
        let mut criteria = serde_json::Map::new();
        for i in 0..255 {
            criteria.insert(format!("opt_{i:03}"), descriptions[(i + si) % descriptions.len()].clone());
        }
        let q = json!({
            "type": "choice",
            "instructions": "Which option is the value?",
            "criteria": criteria
        });
        check_choice(state, &q).unwrap_or_else(|e| panic!("255-option contract violated for state #{si}: {e}"));
    }
}

/// Every level count from 2 to 10 with plain, structured and null levels.
#[test]
fn score_levels_two_to_ten_satisfy_contract() {
    let state = json!({
        "ticket": "The app crashes on every login since the update, I lost a day of work!",
        "priority": "high"
    });
    for k in 2..=10usize {
        let levels: Vec<Value> = (0..k)
            .map(|i| match i % 3 {
                0 => json!(format!("level {i}: severity {}", i * 10)),
                1 => json!({ "label": format!("L{i}"), "description": format!("impact level {i}") }),
                _ => json!(null),
            })
            .collect();
        let q = json!({ "type": "score", "instructions": "How severe is the issue?", "criteria": levels });
        check_score(&state, &q).unwrap_or_else(|e| panic!("score contract violated for K={k}: {e}"));
    }
}

/// Unicode-heavy states and questions are handled like any other text.
#[test]
fn unicode_states_and_criteria_satisfy_contract() {
    for sample in UNICODE_SAMPLES {
        let state = json!({ "text": sample, "note": "\u{1f4e6} package \u{1f6a8}" });
        let choice = json!({
            "type": "choice",
            "instructions": format!("Which option matches {sample}?"),
            "criteria": {
                "\u{1f4b8}": "money",
                "\u{5e9}\u{5dc}\u{5d5}\u{5dd}": null,
                "caf\u{e9}": ["coffee", sample],
                "plain": { "what": sample }
            }
        });
        check_choice(&state, &choice).unwrap_or_else(|e| panic!("{sample:?}: {e}"));
        let score = json!({
            "type": "score",
            "instructions": format!("How much does the text resemble {sample}?"),
            "criteria": ["not at all", sample, "\u{1f680}\u{1f680}\u{1f680}"]
        });
        check_score(&state, &score).unwrap_or_else(|e| panic!("{sample:?}: {e}"));
        let noul = json!({ "type": "noul", "instructions": format!("Does the text mention {sample}?") });
        check_noul(&state, &noul).unwrap_or_else(|e| panic!("{sample:?}: {e}"));
    }
}
