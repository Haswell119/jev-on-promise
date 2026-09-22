//! Neural scorer smoke test. Runs only when SEXTANT_NEURAL_DIR points at an
//! exported artifact directory (CI sets it when a model is published).
#![cfg(feature = "neural")]

use sextant_core::neural::NeuralScorer;
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("SEXTANT_NEURAL_DIR").ok()?);
    p.join("scorer.json").exists().then_some(p)
}

#[test]
fn loads_and_scores_candidates() {
    let Some(d) = dir() else {
        eprintln!("skipped: set SEXTANT_NEURAL_DIR to an exported scorer");
        return;
    };
    let s = NeuralScorer::load(&d).expect("load scorer");
    let cands: Vec<String> = ["billing: payments, invoices and refunds", "shipping: delivery status and tracking", "returns: sending items back"].iter().map(|x| x.to_string()).collect();
    let logits = s.score("Which team should handle this message?", "My package has not arrived and tracking has not updated for five days.", &cands, None, None).expect("score");
    assert_eq!(logits.len(), 3);
    assert!(logits.iter().all(|v| v.is_finite()), "logits must be finite: {logits:?}");
    // Deterministic: the same call twice gives identical logits.
    let again = s.score("Which team should handle this message?", "My package has not arrived and tracking has not updated for five days.", &cands, None, None).unwrap();
    assert_eq!(logits, again);
}
