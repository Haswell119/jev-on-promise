//! The learned retrieval ranker must refuse an artifact it cannot honour.
//! A silently mismatched ranker would reorder evidence with meaningless
//! weights, which looks like a model regression rather than a bad file.

use sextant_core::retrieval::{RetrievalRanker, FEATURE_NAMES, N_RETRIEVAL_FEATURES};

fn weights(n: usize) -> String {
    let w: Vec<String> = (0..n).map(|i| format!("{}", i as f32 * 0.01)).collect();
    format!("[{}]", w.join(","))
}

#[test]
fn accepts_a_matching_artifact() {
    let names: Vec<String> = FEATURE_NAMES.iter().map(|n| format!("\"{n}\"")).collect();
    let json = format!(
        r#"{{"weights":{},"bias":0.25,"version":"test","feature_names":[{}]}}"#,
        weights(N_RETRIEVAL_FEATURES),
        names.join(",")
    );
    let r = RetrievalRanker::from_json(&json).expect("should load");
    assert_eq!(r.weights.len(), N_RETRIEVAL_FEATURES);
    let f = [1.0f32; N_RETRIEVAL_FEATURES];
    let expected: f32 = 0.25 + (0..N_RETRIEVAL_FEATURES).map(|i| i as f32 * 0.01).sum::<f32>();
    assert!((r.score(&f) - expected).abs() < 1e-5, "got {} want {expected}", r.score(&f));
}

#[test]
fn accepts_an_artifact_without_feature_names() {
    let json = format!(r#"{{"weights":{},"bias":0.0}}"#, weights(N_RETRIEVAL_FEATURES));
    assert!(RetrievalRanker::from_json(&json).is_ok());
}

#[test]
fn rejects_the_wrong_number_of_weights() {
    let json = format!(r#"{{"weights":{},"bias":0.0}}"#, weights(N_RETRIEVAL_FEATURES - 1));
    let err = RetrievalRanker::from_json(&json).expect_err("should reject");
    assert!(err.contains("weights"), "unhelpful error: {err}");
}

#[test]
fn rejects_a_different_feature_set() {
    let mut names: Vec<String> = FEATURE_NAMES.iter().map(|n| format!("\"{n}\"")).collect();
    names[0] = "\"something_else\"".into();
    let json = format!(
        r#"{{"weights":{},"bias":0.0,"feature_names":[{}]}}"#,
        weights(N_RETRIEVAL_FEATURES),
        names.join(",")
    );
    let err = RetrievalRanker::from_json(&json).expect_err("should reject");
    assert!(err.contains("feature set"), "unhelpful error: {err}");
}

#[test]
fn rejects_non_finite_weights() {
    let mut w: Vec<String> = (0..N_RETRIEVAL_FEATURES).map(|_| "0.1".to_string()).collect();
    w[3] = "1e400".into(); // parses to +inf
    let json = format!(r#"{{"weights":[{}],"bias":0.0}}"#, w.join(","));
    let err = RetrievalRanker::from_json(&json).expect_err("should reject");
    assert!(err.contains("non-finite"), "unhelpful error: {err}");
}
