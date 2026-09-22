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
}

impl Model {
    pub fn from_parts(weights: Weights, calibration: Calibration, source: &str) -> Model {
        let dense = weights.dense();
        Model { weights, dense, calibration, source: source.to_string() }
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
        Ok(Model::from_parts(weights, calibration, &dir.display().to_string()))
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
