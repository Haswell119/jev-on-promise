//! Calibration artifacts (`model/calibration.json`), kept separate from the
//! semantic weights. Temperature scaling per (primitive, family,
//! cardinality bucket) with fallbacks, Platt scaling for Noul, ordinal
//! smoothing strength and confidence-map coefficients.

use crate::api::QuestionKind;
use crate::question::Family;
use crate::scoring::ConfidenceParams;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Cardinality buckets used for temperature lookup.
pub fn cardinality_bucket(k: usize) -> &'static str {
    match k {
        0..=2 => "2",
        3..=4 => "3-4",
        5..=8 => "5-8",
        9..=16 => "9-16",
        17..=64 => "17-64",
        _ => "65+",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlattParams {
    pub a: f64,
    pub b: f64,
}

impl Default for PlattParams {
    fn default() -> Self {
        PlattParams { a: 1.0, b: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Calibration {
    pub version: String,
    pub method: String,
    /// Global temperature per primitive ("choice" | "score").
    pub temperature: IndexMap<String, f64>,
    /// primitive → family → temperature.
    #[serde(default)]
    pub family_temperature: IndexMap<String, IndexMap<String, f64>>,
    /// primitive → bucket → temperature multiplier (applied on top of family/global).
    #[serde(default)]
    pub bucket_temperature: IndexMap<String, IndexMap<String, f64>>,
    /// Temperature for symbolic (resolver) logits per primitive.
    #[serde(default)]
    pub symbolic_temperature: IndexMap<String, f64>,
    /// Evidence-sensitive temperature per primitive: T_eff = T · (1 + γ · (1 − best_evidence)),
    /// where best_evidence = max_k cov_w(k). Flattens distributions when no option is well supported.
    #[serde(default)]
    pub evidence_gamma: IndexMap<String, f64>,
    /// Laplace-smoothed error rate of symbolic resolvers per primitive
    /// ("choice" | "score" | "noul"): the resolver distribution is mixed
    /// with the uniform distribution by this amount, so a resolver that was
    /// always right on N calibration examples still leaves 1/(N+2) mass.
    #[serde(default)]
    pub symbolic_epsilon: IndexMap<String, f64>,
    /// Noul Platt scaling (global + per family): p = σ(a·logit + b).
    pub noul_platt: PlattParams,
    #[serde(default)]
    pub noul_family_platt: IndexMap<String, PlattParams>,
    /// Platt scaling for symbolic Noul logits.
    #[serde(default)]
    pub noul_symbolic_platt: Option<PlattParams>,
    /// Ordinal adjacency smoothing λ for Score.
    pub ordinal_lambda: f64,
    /// Confidence map coefficients per primitive.
    pub confidence: IndexMap<String, ConfidenceParams>,
    /// Neural scorer fusion (absent = symbolic-only engine).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neural: Option<NeuralFusion>,
}

/// How the neural logits and the symbolic logits are combined.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuralFusion {
    /// Weight on the neural compatibility logit.
    pub weight_neural: f64,
    /// Weight on the symbolic fusion logit.
    pub weight_symbolic: f64,
    /// Temperature applied to the fused logits per primitive.
    #[serde(default)]
    pub temperature: IndexMap<String, f64>,
    /// Platt scaling of the fused Noul logit (z_yes - z_no).
    #[serde(default)]
    pub noul_platt: Option<PlattParams>,
    /// Let symbolic resolvers keep priority when they fire.
    #[serde(default = "default_true")]
    pub resolver_priority: bool,
    /// Confidence coefficients per primitive for the fused path.
    #[serde(default)]
    pub confidence: IndexMap<String, ConfidenceParams>,
}

fn default_true() -> bool {
    true
}

impl Default for NeuralFusion {
    fn default() -> Self {
        NeuralFusion {
            weight_neural: 1.0,
            weight_symbolic: 0.0,
            temperature: IndexMap::new(),
            noul_platt: None,
            resolver_priority: true,
            confidence: IndexMap::new(),
        }
    }
}

impl NeuralFusion {
    pub fn temperature_for(&self, kind: QuestionKind) -> f64 {
        self.temperature.get(kind.as_str()).copied().unwrap_or(1.0).max(1e-3)
    }
    pub fn confidence_for(&self, kind: QuestionKind) -> Option<ConfidenceParams> {
        self.confidence.get(kind.as_str()).cloned()
    }
}

impl Default for Calibration {
    fn default() -> Self {
        let mut temperature = IndexMap::new();
        temperature.insert("choice".to_string(), 1.0);
        temperature.insert("score".to_string(), 1.0);
        let mut symbolic_temperature = IndexMap::new();
        symbolic_temperature.insert("choice".to_string(), 1.0);
        symbolic_temperature.insert("score".to_string(), 1.0);
        let mut symbolic_epsilon = IndexMap::new();
        symbolic_epsilon.insert("choice".to_string(), 0.02);
        symbolic_epsilon.insert("score".to_string(), 0.02);
        symbolic_epsilon.insert("noul".to_string(), 0.02);
        let mut confidence = IndexMap::new();
        confidence.insert("choice".to_string(), ConfidenceParams::default());
        confidence.insert("score".to_string(), ConfidenceParams::default());
        Calibration {
            version: "uncalibrated".into(),
            method: "temperature".into(),
            temperature,
            family_temperature: IndexMap::new(),
            bucket_temperature: IndexMap::new(),
            symbolic_temperature,
            symbolic_epsilon,
            evidence_gamma: IndexMap::new(),
            noul_platt: PlattParams::default(),
            noul_family_platt: IndexMap::new(),
            noul_symbolic_platt: None,
            ordinal_lambda: 0.2,
            confidence,
            neural: None,
        }
    }
}

impl Calibration {
    /// Evidence-sensitive multiplier: 1 + γ · (1 − best_evidence).
    pub fn evidence_multiplier(&self, kind: QuestionKind, best_evidence: f64) -> f64 {
        let g = self.evidence_gamma.get(kind.as_str()).copied().unwrap_or(0.0).max(0.0);
        1.0 + g * (1.0 - best_evidence.clamp(0.0, 1.0))
    }

    /// Effective temperature for a semantic answer.
    pub fn temperature_for(&self, kind: QuestionKind, family: Family, k: usize) -> f64 {
        let key = kind.as_str();
        let base = self
            .family_temperature
            .get(key)
            .and_then(|m| m.get(family.as_str()))
            .copied()
            .or_else(|| self.temperature.get(key).copied())
            .unwrap_or(1.0);
        let mult = self.bucket_temperature.get(key).and_then(|m| m.get(cardinality_bucket(k))).copied().unwrap_or(1.0);
        (base * mult).max(1e-3)
    }

    pub fn symbolic_temperature_for(&self, kind: QuestionKind) -> f64 {
        self.symbolic_temperature.get(kind.as_str()).copied().unwrap_or(1.0).max(1e-3)
    }

    /// Uniform-mixing weight for symbolic answers (default 0.02).
    pub fn symbolic_epsilon_for(&self, kind: QuestionKind) -> f64 {
        self.symbolic_epsilon.get(kind.as_str()).copied().unwrap_or(0.02).clamp(0.0, 0.5)
    }

    pub fn platt_for(&self, family: Family) -> &PlattParams {
        self.noul_family_platt.get(family.as_str()).unwrap_or(&self.noul_platt)
    }

    pub fn confidence_for(&self, kind: QuestionKind) -> ConfidenceParams {
        self.confidence.get(kind.as_str()).cloned().unwrap_or_default()
    }
}

#[inline]
pub fn platt(logit: f64, p: &PlattParams) -> f64 {
    let x = p.a * logit + p.b;
    (1.0 / (1.0 + (-x).exp())).clamp(0.0, 1.0)
}
