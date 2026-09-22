//! Weight artifacts. A `Head` is a linear scorer over the feature vector
//! with a base weight vector plus optional per-family deltas (a mixture of
//! linear experts). The Noul head scores the difference between the two
//! hypotheses plus the raw yes-hypothesis features.

use crate::features::{FeatureVec, F, FEATURE_NAMES, N_FEATURES};
use crate::question::Family;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Head {
    /// feature name → weight (base expert).
    pub base: IndexMap<String, f32>,
    /// family → (feature name → delta added to base).
    #[serde(default)]
    pub families: IndexMap<String, IndexMap<String, f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoulHead {
    /// weights over (f_yes - f_no)
    pub diff: IndexMap<String, f32>,
    /// weights over f_yes (presence of evidence)
    pub yes: IndexMap<String, f32>,
    pub bias: f32,
    #[serde(default)]
    pub families: IndexMap<String, NoulFamilyDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct NoulFamilyDelta {
    #[serde(default)]
    pub diff: IndexMap<String, f32>,
    #[serde(default)]
    pub yes: IndexMap<String, f32>,
    #[serde(default)]
    pub bias: f32,
}

/// Complete weight artifact (`model/weights.json`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Weights {
    pub version: String,
    pub description: String,
    pub features: Vec<String>,
    pub choice: Head,
    pub score: Head,
    pub noul: NoulHead,
}

/// Dense, resolved weights for fast scoring.
#[derive(Debug, Clone)]
pub struct DenseHead {
    pub base: Vec<f32>,
    pub families: Vec<Option<Vec<f32>>>,
}

#[derive(Debug, Clone)]
pub struct DenseNoul {
    pub diff: Vec<f32>,
    pub yes: Vec<f32>,
    pub bias: f32,
    pub families: Vec<Option<(Vec<f32>, Vec<f32>, f32)>>,
}

#[derive(Debug, Clone)]
pub struct DenseWeights {
    pub choice: DenseHead,
    pub score: DenseHead,
    pub noul: DenseNoul,
}

fn to_dense(map: &IndexMap<String, f32>) -> Vec<f32> {
    let mut v = vec![0.0f32; N_FEATURES];
    for (name, w) in map {
        if let Some(i) = FEATURE_NAMES.iter().position(|n| n == name) {
            v[i] = *w;
        }
    }
    v
}

fn family_index(name: &str) -> Option<usize> {
    Family::all().iter().position(|f| f.as_str() == name)
}

impl Head {
    pub fn dense(&self) -> DenseHead {
        let base = to_dense(&self.base);
        let mut families: Vec<Option<Vec<f32>>> = vec![None; Family::all().len()];
        for (fam, delta) in &self.families {
            if let Some(i) = family_index(fam) {
                let d = to_dense(delta);
                let mut w = base.clone();
                for j in 0..N_FEATURES {
                    w[j] += d[j];
                }
                families[i] = Some(w);
            }
        }
        DenseHead { base, families }
    }
}

impl DenseHead {
    #[inline]
    pub fn weights_for(&self, family: Family) -> &[f32] {
        let i = Family::all().iter().position(|f| *f == family).unwrap_or(0);
        self.families[i].as_deref().unwrap_or(&self.base)
    }
    #[inline]
    pub fn score(&self, f: &FeatureVec, family: Family) -> f32 {
        f.dot(self.weights_for(family))
    }
}

impl NoulHead {
    pub fn dense(&self) -> DenseNoul {
        let diff = to_dense(&self.diff);
        let yes = to_dense(&self.yes);
        let mut families: Vec<Option<(Vec<f32>, Vec<f32>, f32)>> = vec![None; Family::all().len()];
        for (fam, delta) in &self.families {
            if let Some(i) = family_index(fam) {
                let dd = to_dense(&delta.diff);
                let dy = to_dense(&delta.yes);
                let mut d = diff.clone();
                let mut y = yes.clone();
                for j in 0..N_FEATURES {
                    d[j] += dd[j];
                    y[j] += dy[j];
                }
                families[i] = Some((d, y, self.bias + delta.bias));
            }
        }
        DenseNoul { diff, yes, bias: self.bias, families }
    }
}

impl DenseNoul {
    /// Logit of P(true) from the yes/no hypothesis feature vectors.
    pub fn logit(&self, f_yes: &FeatureVec, f_no: &FeatureVec, family: Family) -> f32 {
        let i = Family::all().iter().position(|f| *f == family).unwrap_or(0);
        let (d, y, b) = match &self.families[i] {
            Some((d, y, b)) => (d.as_slice(), y.as_slice(), *b),
            None => (self.diff.as_slice(), self.yes.as_slice(), self.bias),
        };
        let mut s = b;
        for j in 0..N_FEATURES {
            s += d[j] * (f_yes.0[j] - f_no.0[j]) + y[j] * f_yes.0[j];
        }
        s
    }
}

impl Weights {
    pub fn dense(&self) -> DenseWeights {
        DenseWeights { choice: self.choice.dense(), score: self.score.dense(), noul: self.noul.dense() }
    }

    /// Hand-set bootstrap weights used before any training has run. They
    /// encode the obvious signs (coverage up, contradiction down) so that the
    /// engine is usable out of the box; `sextant train` replaces them.
    pub fn bootstrap() -> Weights {
        let mut base: IndexMap<String, f32> = IndexMap::new();
        let mut set = |f: F, w: f32| {
            base.insert(FEATURE_NAMES[f.idx()].to_string(), w);
        };
        set(F::bm25_best, 1.0);
        set(F::bm25_global, 0.5);
        set(F::cov_w, 2.0);
        set(F::cov_best, 1.0);
        set(F::cov_rare, 1.5);
        set(F::cov_name, 1.0);
        set(F::cos_global, 1.0);
        set(F::cos_best, 2.0);
        set(F::jaccard_best, 1.0);
        set(F::bigram_hits, 1.5);
        set(F::gram_dice_best, 1.0);
        set(F::gram_cov, 0.5);
        set(F::literal_hit, 2.0);
        set(F::literal_len, 2.0);
        set(F::literal_count, 0.3);
        set(F::syn_cov, 1.0);
        set(F::antonym_hits, -1.5);
        set(F::neg_agree, 1.0);
        set(F::neg_conflict, -3.0);
        set(F::hyp_conflict, -1.5);
        set(F::req_agree, 1.0);
        set(F::neg_field_cov, -2.0);
        set(F::neg_field_best, -0.5);
        set(F::example_max, 1.5);
        set(F::example_mean, 0.5);
        set(F::focus_cov, 1.0);
        set(F::focus_literal, 1.0);
        set(F::valence_agree, 2.0);
        set(F::intensity_dist, 2.0);
        set(F::range_hit, 4.0);
        set(F::range_dist, 2.0);
        set(F::evidence_density, 0.3);
        set(F::ood, -0.5);
        set(F::q_cov_best, 0.5);
        set(F::key_match, 2.0);
        set(F::key_negated, -3.0);
        set(F::hyp_state, -0.5);
        set(F::hyper_match, 1.0);
        set(F::domain_match, 0.5);
        set(F::antonym_negated, 1.0);
        set(F::directive_frac, -2.0);
        set(F::x_sim_pos, 2.0);
        set(F::x_conflict_neg, 2.0);
        set(F::x_low_neutral, 1.5);
        set(F::ord_pos_int, 2.0);
        set(F::ord_pos_val, 2.0);
        set(F::ord_hit, 1.0);
        let choice = Head { base: base.clone(), families: IndexMap::new() };
        let score = Head { base: base.clone(), families: IndexMap::new() };
        let mut yes: IndexMap<String, f32> = IndexMap::new();
        yes.insert(FEATURE_NAMES[F::cov_w.idx()].to_string(), 1.0);
        yes.insert(FEATURE_NAMES[F::cov_rare.idx()].to_string(), 1.0);
        yes.insert(FEATURE_NAMES[F::ood.idx()].to_string(), -0.5);
        yes.insert(FEATURE_NAMES[F::xfield_cov_ba.idx()].to_string(), 2.0);
        yes.insert(FEATURE_NAMES[F::xfield_neg_conflict.idx()].to_string(), -3.0);
        yes.insert(FEATURE_NAMES[F::xfield_antonym.idx()].to_string(), -2.0);
        yes.insert(FEATURE_NAMES[F::window_ok.idx()].to_string(), 2.0);
        let noul = NoulHead { diff: base, yes, bias: 0.0, families: IndexMap::new() };
        Weights {
            version: "bootstrap".into(),
            description: "Hand-set bootstrap weights (not trained). Run `sextant train` to fit.".into(),
            features: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            choice,
            score,
            noul,
        }
    }
}
