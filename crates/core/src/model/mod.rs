//! Model artifact = weights + calibration. Both are small JSON files that
//! are embedded at compile time (so the binary is self-contained) and can be
//! overridden from a directory at runtime.

use crate::calibration::Calibration;
use crate::scoring::{DenseWeights, Weights};
use std::path::Path;

pub const EMBEDDED_WEIGHTS: &str = include_str!("../../../../model/weights.json");
pub const EMBEDDED_CALIBRATION: &str = include_str!("../../../../model/calibration.json");

#[derive(Debug, Clone)]
pub struct Model {
    pub weights: Weights,
    pub dense: DenseWeights,
    pub calibration: Calibration,
    pub source: String,
    /// Locally hosted neural decision scorer (loaded from `<dir>/neural`).
    #[cfg(feature = "neural")]
    pub neural: Option<std::sync::Arc<crate::neural::NeuralScorer>>,
    /// Learned evidence ranker (loaded from `<dir>/retrieval.json`). Absent
    /// means the heuristic pooled-BM25 ordering is used.
    pub retrieval: Option<std::sync::Arc<crate::retrieval::RetrievalRanker>>,
}

impl Model {
    pub fn from_parts(weights: Weights, calibration: Calibration, source: &str) -> Model {
        let dense = weights.dense();
        Model {
            weights,
            dense,
            calibration,
            source: source.to_string(),
            #[cfg(feature = "neural")]
            neural: None,
            retrieval: None,
        }
    }

    /// Attach a neural scorer directory (`config.json`, `model.safetensors`,
    /// `tokenizer.json`, `head.safetensors`, `scorer.json`).
    #[cfg(feature = "neural")]
    pub fn with_neural_dir(mut self, dir: &Path) -> Result<Model, String> {
        let scorer = crate::neural::NeuralScorer::load(dir)?;
        self.neural = Some(std::sync::Arc::new(scorer));
        Ok(self)
    }

    /// Attach a learned retrieval ranker from `<dir>/retrieval.json`.
    pub fn with_retrieval_file(mut self, path: &Path) -> Result<Model, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ranker = crate::retrieval::RetrievalRanker::from_json(&text)?;
        self.retrieval = Some(std::sync::Arc::new(ranker));
        Ok(self)
    }

    /// Candidate index the loaded scorer treats as the "true" hypothesis of
    /// a Noul question. 0 with no scorer, which is the correct convention.
    pub fn noul_true_index(&self) -> usize {
        #[cfg(feature = "neural")]
        {
            return self.neural.as_ref().map(|n| n.config.noul_true_index.min(1)).unwrap_or(0);
        }
        #[cfg(not(feature = "neural"))]
        {
            0
        }
    }

    /// True when a learned retrieval ranker is loaded.
    pub fn has_retrieval(&self) -> bool {
        self.retrieval.is_some()
    }

    /// True when a neural scorer is loaded.
    pub fn has_neural(&self) -> bool {
        #[cfg(feature = "neural")]
        {
            self.neural.is_some()
        }
        #[cfg(not(feature = "neural"))]
        {
            false
        }
    }

    /// The compiled-in artifact.
    pub fn embedded() -> Model {
        let weights: Weights = serde_json::from_str(EMBEDDED_WEIGHTS).unwrap_or_else(|_| Weights::bootstrap());
        let calibration: Calibration = serde_json::from_str(EMBEDDED_CALIBRATION).unwrap_or_default();
        Model::from_parts(weights, calibration, "embedded")
    }

    /// Bootstrap weights + default calibration (no training at all).
    pub fn bootstrap() -> Model {
        Model::from_parts(Weights::bootstrap(), Calibration::default(), "bootstrap")
    }

    /// Load `weights.json` and `calibration.json` from a directory.
    pub fn load_dir(dir: &Path) -> Result<Model, String> {
        let w = std::fs::read_to_string(dir.join("weights.json")).map_err(|e| format!("weights.json: {e}"))?;
        let weights: Weights = serde_json::from_str(&w).map_err(|e| format!("weights.json: {e}"))?;
        let c = std::fs::read_to_string(dir.join("calibration.json")).map_err(|e| format!("calibration.json: {e}"))?;
        let calibration: Calibration = serde_json::from_str(&c).map_err(|e| format!("calibration.json: {e}"))?;
        validate_calibration(&calibration)?;
        let mut model = Model::from_parts(weights, calibration, &dir.display().to_string());
        let rf = dir.join("retrieval.json");
        if rf.exists() {
            model = model.with_retrieval_file(&rf)?;
        }
        #[cfg(feature = "neural")]
        {
            let nd = dir.join("neural");
            if nd.join("scorer.json").exists() {
                return model.with_neural_dir(&nd);
            }
        }
        Ok(model)
    }
}

/// Reject artifacts with invalid numbers (NaN, non-positive temperatures).
pub fn validate_calibration(c: &Calibration) -> Result<(), String> {
    for (k, v) in c.temperature.iter().chain(c.symbolic_temperature.iter()) {
        if !v.is_finite() || *v <= 0.0 {
            return Err(format!("temperature[{k}] must be a positive finite number"));
        }
    }
    for (_, m) in c.family_temperature.iter().chain(c.bucket_temperature.iter()) {
        for (k, v) in m {
            if !v.is_finite() || *v <= 0.0 {
                return Err(format!("temperature[{k}] must be a positive finite number"));
            }
        }
    }
    if !c.ordinal_lambda.is_finite() || !(0.0..1.0).contains(&c.ordinal_lambda) {
        return Err("ordinal_lambda must be in [0, 1)".into());
    }
    if !c.noul_platt.a.is_finite() || !c.noul_platt.b.is_finite() {
        return Err("noul_platt must be finite".into());
    }
    Ok(())
}
