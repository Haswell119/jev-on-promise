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
    xfield_available,
    xfield_cov_ab,
    xfield_cov_ba,
    xfield_jaccard,
    xfield_cos,
    xfield_gram,
    xfield_neg_conflict,
    xfield_antonym,
    xfield_num_conflict,
    x_sim_pos,
    x_conflict_neg,
    x_low_neutral,
    opt_neg_share,
    opt_hyp_share,
    ord_pos_int,
    ord_pos_val,
    ord_hit,
    window_ok,
    bias,
}

pub mod cross;

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
        self.0.iter().zip(w.iter()).map(|(a, b)| a * b).sum()
    }
    pub fn named(&self) -> indexmap::IndexMap<String, f64> {
        FEATURE_NAMES.iter().zip(self.0.iter()).map(|(n, v)| (n.to_string(), *v as f64)).collect()
    }
}
