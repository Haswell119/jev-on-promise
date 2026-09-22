//! Scalar confidence for Choice / Score answers. Built from the SHAPE of the
//! distribution (normalized entropy concentration and winner margin) plus
//! evidence signals, passed through a logistic map whose coefficients are
//! fitted against empirical correctness on the calibration split.
//!
//! confidence = σ(a·concentration + b·margin + c·evidence + d·ood + e)
//!
//! where concentration = 1 − H(p)/log K and margin = p₁ − p₂.

use super::softmax::entropy_normalized;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfidenceParams {
    pub a_concentration: f64,
    pub b_margin: f64,
    pub c_evidence: f64,
    pub d_ood: f64,
    pub bias: f64,
}

impl Default for ConfidenceParams {
    fn default() -> Self {
        // Uncalibrated default: mostly concentration, some margin.
        ConfidenceParams { a_concentration: 3.0, b_margin: 2.0, c_evidence: 0.0, d_ood: 0.0, bias: -2.0 }
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Shape statistics used by the confidence model.
pub fn shape(p: &[f64]) -> (f64, f64) {
    let concentration = 1.0 - entropy_normalized(p);
    let mut sorted = p.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let margin = if sorted.len() >= 2 { sorted[0] - sorted[1] } else { 1.0 };
    (concentration, margin)
}

pub fn confidence(p: &[f64], evidence: f64, ood: f64, params: &ConfidenceParams) -> f64 {
    let (conc, margin) = shape(p);
    let x = params.a_concentration * conc + params.b_margin * margin + params.c_evidence * evidence + params.d_ood * ood + params.bias;
    sigmoid(x).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_is_monotone_in_peakedness() {
        let p = ConfidenceParams::default();
        let flat = confidence(&[0.34, 0.33, 0.33], 0.0, 0.0, &p);
        let peaked = confidence(&[0.9, 0.05, 0.05], 0.0, 0.0, &p);
        assert!(peaked > flat);
        assert!((0.0..=1.0).contains(&flat) && (0.0..=1.0).contains(&peaked));
    }
}
