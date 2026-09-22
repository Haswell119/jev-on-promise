//! Learned retrieval ranker.
//!
//! BM25 alone retrieves the decisive fragment in roughly a third of long
//! states, while the fragment itself is a median 38 words against a
//! 140-word budget: the budget is not the binding constraint, the ranking
//! is. This module scores each segment with a small linear model over
//! features that are already computed during retrieval, so the whole
//! ranker costs one dot product per segment and stays inside the CPU
//! latency budget that a second encoder pass would blow.
//!
//! The weights are fitted offline on gold evidence spans
//! (`scripts/neural/train_retrieval.py`) and loaded from
//! `<model_dir>/retrieval.json`. With no artifact present the engine keeps
//! the pooled BM25 ordering, so the ranker is strictly additive.

use crate::question::QuestionView;
use crate::state::index::StateIndex;
use serde::{Deserialize, Serialize};

/// Feature vector length. Keep in sync with `FEATURE_NAMES`.
pub const N_RETRIEVAL_FEATURES: usize = 16;

pub const FEATURE_NAMES: [&str; N_RETRIEVAL_FEATURES] = [
    "q_z",
    "q_rank",
    "pooled_z",
    "pooled_rank",
    "crit_max_z",
    "crit_max_rank",
    "q_term_coverage",
    "crit_term_coverage",
    "crit_gram_containment",
    "len_ratio",
    "doc_position",
    "focus_field",
    "has_number",
    "has_date",
    "has_negation",
    "log_n_segments",
];

/// Linear ranker over `FEATURE_NAMES`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRanker {
    pub weights: Vec<f32>,
    #[serde(default)]
    pub bias: f32,
    /// Provenance: which experiment fitted these weights.
    #[serde(default)]
    pub version: String,
    /// Feature names as fitted, checked against `FEATURE_NAMES` on load.
    #[serde(default)]
    pub feature_names: Vec<String>,
}

impl RetrievalRanker {
    pub fn from_json(s: &str) -> Result<RetrievalRanker, String> {
        let r: RetrievalRanker = serde_json::from_str(s).map_err(|e| format!("retrieval.json: {e}"))?;
        r.validate()?;
        Ok(r)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.weights.len() != N_RETRIEVAL_FEATURES {
            return Err(format!(
                "retrieval ranker has {} weights, this build computes {N_RETRIEVAL_FEATURES} features",
                self.weights.len()
            ));
        }
        if !self.feature_names.is_empty() && self.feature_names.iter().map(|s| s.as_str()).ne(FEATURE_NAMES) {
            return Err("retrieval ranker was fitted on a different feature set than this build computes".into());
        }
        if let Some(bad) = self.weights.iter().find(|w| !w.is_finite()) {
            return Err(format!("retrieval ranker has a non-finite weight: {bad}"));
        }
        Ok(())
    }

    #[inline]
    pub fn score(&self, f: &[f32; N_RETRIEVAL_FEATURES]) -> f32 {
        let mut s = self.bias;
        for (w, x) in self.weights.iter().zip(f.iter()) {
            s += w * x;
        }
        s
    }
}

/// Per-document statistics used to normalise a raw score into a z-score and
/// a rank percentile. Raw BM25 magnitudes are not comparable across states,
/// so a ranker trained on raw values would learn the document, not the
/// segment.
struct Norm {
    mean: f32,
    inv_sd: f32,
    /// `ranks[i]` is the percentile of segment `i` (1.0 = highest score).
    ranks: Vec<f32>,
}

impl Norm {
    fn new(scores: &[f32]) -> Norm {
        let n = scores.len().max(1) as f32;
        let mean = scores.iter().sum::<f32>() / n;
        let var = scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
        let sd = var.sqrt();
        let inv_sd = if sd > 1e-6 { 1.0 / sd } else { 0.0 };
        let mut order: Vec<u32> = (0..scores.len() as u32).collect();
        order.sort_by(|a, b| {
            scores[*a as usize].partial_cmp(&scores[*b as usize]).unwrap_or(std::cmp::Ordering::Equal).then(b.cmp(a))
        });
        let mut ranks = vec![0.0f32; scores.len()];
        let denom = (scores.len().saturating_sub(1)).max(1) as f32;
        for (rank, &seg) in order.iter().enumerate() {
            ranks[seg as usize] = rank as f32 / denom;
        }
        Norm { mean, inv_sd, ranks }
    }

    #[inline]
    fn z(&self, i: usize, scores: &[f32]) -> f32 {
        ((scores[i] - self.mean) * self.inv_sd).clamp(-6.0, 6.0)
    }
}

/// Fraction of `needles` present in the sorted, de-duplicated `hay`.
fn sorted_containment(needles: &[u32], hay: &[u32]) -> f32 {
    if needles.is_empty() {
        return 0.0;
    }
    let (mut i, mut j, mut hit) = (0usize, 0usize, 0usize);
    while i < needles.len() && j < hay.len() {
        match needles[i].cmp(&hay[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                hit += 1;
                i += 1;
                j += 1;
            }
        }
    }
    hit as f32 / needles.len() as f32
}

/// Fraction of `terms` that the segment's term-frequency list contains.
fn term_coverage(terms: &[crate::state::vocab::TermId], tf: &[(crate::state::vocab::TermId, u16)]) -> f32 {
    if terms.is_empty() {
        return 0.0;
    }
    let hit = terms.iter().filter(|t| tf.binary_search_by_key(*t, |(x, _)| *x).is_ok()).count();
    hit as f32 / terms.len() as f32
}

/// Compute the ranker's feature matrix for every segment of a state.
///
/// `q_scores`, `pooled` and `crit_scores` are the BM25 scores the evidence
/// selector has already computed, so the features cost no extra retrieval.
pub fn segment_features(
    view: &QuestionView<'_>,
    state: &StateIndex,
    q_scores: &[f32],
    pooled: &[f32],
    crit_scores: &[Vec<f32>],
) -> Vec<[f32; N_RETRIEVAL_FEATURES]> {
    let n_seg = state.segments.len();
    let crit_max: Vec<f32> = (0..n_seg)
        .map(|i| crit_scores.iter().map(|s| s[i]).fold(f32::NEG_INFINITY, f32::max))
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect();
    let nq = Norm::new(q_scores);
    let np = Norm::new(pooled);
    let nc = Norm::new(&crit_max);

    let q_terms: Vec<crate::state::vocab::TermId> = {
        let mut v: Vec<_> = view.terms.iter().filter(|t| !t.negated).map(|t| t.term).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let crit_terms: Vec<Vec<crate::state::vocab::TermId>> = view
        .criteria
        .iter()
        .map(|c| {
            let mut v: Vec<_> = c.terms.iter().filter(|t| t.polarity > 0).map(|t| t.term).collect();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();

    let mut has_number = vec![0.0f32; n_seg];
    for nspan in &state.numbers {
        if let Some(slot) = has_number.get_mut(nspan.seg as usize) {
            *slot = 1.0;
        }
    }
    let mut has_date = vec![0.0f32; n_seg];
    for d in &state.dates {
        if let Some(slot) = has_date.get_mut(d.seg as usize) {
            *slot = 1.0;
        }
    }

    let log_n = ((n_seg as f32) + 1.0).ln() / 8.0;
    let pos_denom = (n_seg.saturating_sub(1)).max(1) as f32;

    (0..n_seg)
        .map(|i| {
            let seg = &state.segments[i];
            let crit_cov = crit_terms.iter().map(|t| term_coverage(t, &seg.tf)).fold(0.0f32, f32::max);
            let gram_cov = view
                .criteria
                .iter()
                .map(|c| sorted_containment(&c.grams, &seg.grams))
                .fold(0.0f32, f32::max);
            [
                nq.z(i, q_scores),
                nq.ranks[i],
                np.z(i, pooled),
                np.ranks[i],
                nc.z(i, &crit_max),
                nc.ranks[i],
                term_coverage(&q_terms, &seg.tf),
                crit_cov,
                gram_cov,
                (seg.content_len as f32 / state.avg_seg_len.max(1.0)).min(8.0),
                i as f32 / pos_denom,
                if view.focus_fields.binary_search(&seg.field).is_ok() { 1.0 } else { 0.0 },
                has_number[i],
                has_date[i],
                if seg.flags_any & crate::text::negation::NEGATED != 0 { 1.0 } else { 0.0 },
                log_n,
            ]
        })
        .collect()
}
