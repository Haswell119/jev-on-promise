//! Compact, named feature vectors per criterion. The names are part of the
//! model artifact so weights stay human-inspectable.

pub mod extract;

pub use extract::{extract_features, CriterionEvidence, FeatureMatrix};

macro_rules! features {
    ($( $name:ident ),* $(,)?) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(usize)]
        pub enum F { $( $name ),* }
        pub const FEATURE_NAMES: &[&str] = &[ $( stringify!($name) ),* ];
        pub const N_FEATURES: usize = FEATURE_NAMES.len();
    };
}

features! {
    bm25_best,
    bm25_global,
    cov_w,
    cov_best,
    cov_rare,
    cov_name,
    cos_global,
    cos_best,
    jaccard_best,
    bigram_hits,
    gram_dice_best,
    gram_cov,
    literal_hit,
    literal_len,
    literal_count,
    syn_cov,
    antonym_hits,
    neg_agree,
    neg_conflict,
    hyp_conflict,
    req_agree,
    neg_field_cov,
    neg_field_best,
    example_max,
    example_mean,
    focus_cov,
    focus_literal,
    valence_agree,
    valence_abs,
    intensity_dist,
    intensity_cov,
    range_hit,
    range_dist,
    evidence_density,
    term_count_log,
    ood,
    q_cov_best,
    key_match,
    key_negated,
    hyp_state,
    null_desc,
    hyper_match,
    domain_match,
    antonym_negated,
    directive_frac,
    bias,
}

impl F {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
}

/// Fixed-size feature vector.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVec(pub [f32; N_FEATURES]);

impl Default for FeatureVec {
    fn default() -> Self {
        FeatureVec([0.0; N_FEATURES])
    }
}

impl FeatureVec {
    #[inline]
    pub fn get(&self, f: F) -> f32 {
        self.0[f as usize]
    }
    #[inline]
    pub fn set(&mut self, f: F, v: f32) {
        self.0[f as usize] = v;
    }
    #[inline]
    pub fn dot(&self, w: &[f32]) -> f32 {
        debug_assert_eq!(w.len(), N_FEATURES);
        let mut s = 0.0f32;
        for i in 0..N_FEATURES {
            s += self.0[i] * w[i];
        }
        s
    }
    pub fn named(&self) -> indexmap::IndexMap<String, f64> {
        FEATURE_NAMES.iter().zip(self.0.iter()).map(|(n, v)| (n.to_string(), *v as f64)).collect()
    }
}
