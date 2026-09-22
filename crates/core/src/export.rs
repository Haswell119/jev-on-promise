//! Export of neural training / inference inputs.
//!
//! The neural scorer consumes the SAME question analysis and evidence
//! retrieval as the symbolic path, so training and inference cannot drift.
//! For one question we emit:
//!
//! * `question`: the flattened instruction text (plus context fields),
//! * `evidence`: the highest-scoring state fragments for the pooled query
//!   (question terms + every criterion's terms), in document order, capped
//!   by a word budget — this is what makes long states tractable,
//! * one entry per candidate criterion with its rendered description and its
//!   symbolic feature vector (so hybrid fusion can be trained offline).

use crate::api::{Question, QuestionKind};
use crate::features::{extract_features, FeatureVec, FEATURE_NAMES};
use crate::lexicon::Resources;
use crate::question::QuestionView;
use crate::resolvers;
use crate::state::index::StateIndex;
use serde::{Deserialize, Serialize};

/// Word budget for the evidence block handed to the encoder.
pub const DEFAULT_EVIDENCE_WORDS: usize = 140;

/// Segments taken either side of each chosen segment. Measured on the
/// internal long-context suite: full recall of the annotated span rises
/// from 0.306 at 0 to 0.462 at 3 and is flat beyond, so 3 is the knee.
pub const DEFAULT_NEIGHBOUR_GLUE: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateExport {
    pub key: String,
    /// Rendered candidate description (key words + description text).
    pub text: String,
    /// Symbolic feature vector in `FEATURE_NAMES` order.
    pub features: Vec<f32>,
    /// Logit produced by the symbolic fusion head (semantic path).
    pub symbolic_logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionExport {
    pub kind: String,
    pub family: String,
    /// Instruction text as analysed (question sentences first).
    pub question: String,
    /// Retrieved evidence fragments joined with " … " in document order.
    pub evidence: String,
    /// Whole state words (before retrieval).
    pub state_words: usize,
    /// Words actually handed over after retrieval.
    pub evidence_words: usize,
    /// Fraction of the state kept.
    pub coverage: f32,
    pub candidates: Vec<CandidateExport>,
    /// `symbolic:<resolver>` when a resolver fires, else `semantic`.
    pub path: String,
    /// Resolver logits when a resolver fired (same order as `candidates`,
    /// or a single element for Noul).
    pub symbolic_resolver_logits: Option<Vec<f32>>,
    pub feature_names: Vec<String>,
}

/// Render one criterion as the text the encoder sees (public: the engine
/// must build exactly the same strings at inference time).
pub fn candidate_text_for(view: &QuestionView<'_>, idx: usize) -> String {
    candidate_text(view, idx)
}

/// Render one criterion as the text the encoder sees.
fn candidate_text(view: &QuestionView<'_>, idx: usize) -> String {
    let c = &view.criteria[idx];
    let key_words: Vec<String> = crate::state::flatten::split_key_words(&c.key);
    let name = key_words.join(" ");
    let desc = c.full_text.trim();
    match (view.kind, name.is_empty(), desc.is_empty()) {
        (QuestionKind::Score, _, false) => desc.to_string(),
        (QuestionKind::Score, _, true) => format!("level {}", c.index),
        (QuestionKind::Noul, _, false) => format!("{name}: {desc}"),
        (QuestionKind::Noul, _, true) => name,
        (_, false, false) => format!("{name}: {desc}"),
        (_, false, true) => name,
        (_, true, _) => desc.to_string(),
    }
}

/// Evidence selection parameters. `per_criterion` reserves slots so that
/// every candidate gets a chance to contribute its supporting fragment
/// instead of the pooled query being dominated by one option's vocabulary.
#[derive(Debug, Clone, Copy)]
pub struct EvidenceParams {
    pub budget_words: usize,
    /// Segments reserved for the question-only query.
    pub question_slots: usize,
    /// Segments reserved per candidate criterion.
    pub per_criterion: usize,
    /// `pooled` (v1) or `quota` (v2, per-candidate reservations).
    pub strategy: Strategy,
    /// When > 0, the budget grows with the state length up to this cap:
    /// long documents get proportionally more evidence, short ones stay
    /// cheap. `budget = min(cap, base + (state_words - base) / 6)`.
    pub adaptive_cap: usize,
    /// Weight retrieval terms by their document frequency *inside this
    /// state* (see `local_idf`). Off reproduces the v1/v2 scoring.
    pub local_idf: bool,
    /// Expand the question query with the criteria synonym sets, so a
    /// needle that paraphrases the question is still reachable.
    pub q_expand: bool,
    /// Fill any budget left after the scored ranking with unscored
    /// segments in document order. A needle with no lexical overlap is
    /// otherwise unreachable, but the fill always starts at segment 0, so
    /// it is also a standing bet that evidence sits early in the state.
    /// Turn it off to measure how much of the heuristic's score is the
    /// ranking and how much is that bet.
    pub doc_order_fallback: bool,
    /// Also take this many segments either side of each chosen segment
    /// when the budget allows. The recall metric, and
    /// a reader, both need a whole sentence: an annotated span often runs
    /// across several segments, and a selection that takes the best ones
    /// individually can return the span in pieces.
    pub neighbour_glue: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Pooled,
    Quota,
}

impl Default for EvidenceParams {
    fn default() -> Self {
        EvidenceParams {
            budget_words: DEFAULT_EVIDENCE_WORDS,
            question_slots: 3,
            per_criterion: 2,
            strategy: Strategy::Quota,
            adaptive_cap: 0,
            local_idf: false,
            q_expand: false,
            doc_order_fallback: true,
            neighbour_glue: DEFAULT_NEIGHBOUR_GLUE,
        }
    }
}

impl EvidenceParams {
    pub fn pooled(budget_words: usize) -> Self {
        EvidenceParams { budget_words, question_slots: 0, per_criterion: 0, strategy: Strategy::Pooled, ..Default::default() }
    }
    pub fn quota(budget_words: usize) -> Self {
        EvidenceParams { budget_words, ..Default::default() }
    }
    pub fn from_name(name: &str, budget_words: usize) -> Self {
        match name {
            "pooled" => EvidenceParams::pooled(budget_words),
            _ => EvidenceParams::quota(budget_words),
        }
    }

    pub fn with_adaptive_cap(mut self, cap: usize) -> Self {
        self.adaptive_cap = cap;
        self
    }

    pub fn with_local_idf(mut self, on: bool) -> Self {
        self.local_idf = on;
        self
    }

    pub fn with_q_expand(mut self, on: bool) -> Self {
        self.q_expand = on;
        self
    }

    pub fn with_doc_order_fallback(mut self, on: bool) -> Self {
        self.doc_order_fallback = on;
        self
    }

    pub fn with_neighbour_glue(mut self, n: usize) -> Self {
        self.neighbour_glue = n;
        self
    }

    /// Effective word budget for a state of `state_words` words.
    pub fn effective_budget(&self, state_words: usize) -> usize {
        if self.adaptive_cap <= self.budget_words {
            return self.budget_words;
        }
        let grown = self.budget_words + state_words.saturating_sub(self.budget_words) / 6;
        grown.clamp(self.budget_words, self.adaptive_cap)
    }
}

/// Document frequency factor of a term *within this state*.
///
/// The query-side weight already carries a corpus-level idf, but inside one
/// long document the discriminating signal is how many of ITS segments
/// contain the term: a state about refunds mentions "refund" everywhere, so
/// that term localises nothing. Without this factor long states are scored
/// almost uniformly and retrieval degenerates to document order.
fn local_idf(n_segments: usize, df: usize) -> f32 {
    let n = n_segments as f32;
    let df = df.max(1) as f32;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

fn bm25_scores(
    state: &StateIndex,
    terms: impl Iterator<Item = (crate::state::vocab::TermId, f32)>,
    scores: &mut [f32],
    idf: bool,
) {
    let n_seg = state.segments.len();
    for (term, weight) in terms {
        if let Some(post) = state.postings.get(&term) {
            let boost = if idf { local_idf(n_seg, post.len()) } else { 1.0 };
            if boost <= 0.0 {
                continue;
            }
            for &(seg, tf) in post {
                let s = &state.segments[seg as usize];
                let tf = tf as f32;
                let norm = tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * s.content_len as f32 / state.avg_seg_len));
                scores[seg as usize] += weight * boost * norm;
            }
        }
    }
}

/// The three BM25 score vectors the selector and the learned ranker share.
pub struct BaseScores {
    /// Question terms (plus the expansion and the focus-field prior).
    pub q_scores: Vec<f32>,
    /// One vector per candidate criterion.
    pub crit_scores: Vec<Vec<f32>>,
    /// Question plus every criterion.
    pub pooled: Vec<f32>,
}

/// Score every segment for the question and for each candidate. Shared by
/// the heuristic selector and by the learned ranker's feature extraction so
/// the two can never disagree about what the base scores are.
pub fn base_scores(view: &QuestionView<'_>, state: &StateIndex, local_idf: bool, q_expand: bool) -> BaseScores {
    let n_seg = state.segments.len();
    let mut q_scores = vec![0.0f32; n_seg];
    bm25_scores(state, view.terms.iter().map(|t| (t.term, t.weight)), &mut q_scores, local_idf);
    if q_expand {
        // The question itself carries no synonym set, so borrow the union of
        // the criteria expansions: a needle that paraphrases the question is
        // otherwise invisible to the question-only ranking.
        for c in &view.criteria {
            bm25_scores(
                state,
                c.expanded.iter().take(24).map(|&(syn, weight, _)| (syn, weight * 0.35)),
                &mut q_scores,
                local_idf,
            );
        }
    }
    // A strong prior on fields the question references explicitly.
    if !view.focus_fields.is_empty() {
        for (i, seg) in state.segments.iter().enumerate() {
            if view.focus_fields.binary_search(&seg.field).is_ok() {
                q_scores[i] += 2.0;
            }
        }
    }
    let mut crit_scores: Vec<Vec<f32>> = Vec::with_capacity(view.criteria.len());
    let mut pooled = q_scores.clone();
    for c in &view.criteria {
        let mut s = vec![0.0f32; n_seg];
        let w = 1.0 / (c.terms.len().max(1) as f32).sqrt();
        bm25_scores(state, c.terms.iter().filter(|t| t.polarity > 0).map(|t| (t.term, t.weight * w)), &mut s, local_idf);
        bm25_scores(state, c.expanded.iter().take(24).map(|&(syn, weight, _)| (syn, weight * 0.5 * w)), &mut s, local_idf);
        for i in 0..n_seg {
            pooled[i] += s[i];
        }
        crit_scores.push(s);
    }
    BaseScores { q_scores, crit_scores, pooled }
}

/// Feature matrix for the learned retrieval ranker, one row per segment.
pub fn retrieval_features(
    view: &QuestionView<'_>,
    state: &StateIndex,
    local_idf: bool,
    q_expand: bool,
) -> Vec<[f32; crate::retrieval::N_RETRIEVAL_FEATURES]> {
    let b = base_scores(view, state, local_idf, q_expand);
    crate::retrieval::segment_features(view, state, &b.q_scores, &b.pooled, &b.crit_scores)
}

/// Select evidence fragments for a question, in document order, within a
/// word budget. Short states are returned whole.
pub fn select_evidence(view: &QuestionView<'_>, state: &StateIndex, budget_words: usize) -> (String, usize, usize) {
    select_evidence_with(view, state, EvidenceParams::quota(budget_words))
}

pub fn select_evidence_with(view: &QuestionView<'_>, state: &StateIndex, p: EvidenceParams) -> (String, usize, usize) {
    select_evidence_ranked(view, state, p, None)
}

/// Evidence selection with an optional learned ranker. With a ranker every
/// segment is a candidate and the budget is filled greedily by learned
/// score, which also removes the document-order fallback that biased the
/// heuristic selector toward the start of long states.
pub fn select_evidence_ranked(
    view: &QuestionView<'_>,
    state: &StateIndex,
    p: EvidenceParams,
    ranker: Option<&crate::retrieval::RetrievalRanker>,
) -> (String, usize, usize) {
    let all_words: usize = state.fields.iter().map(|f| f.text.split_whitespace().count()).sum();
    let mut p = p;
    p.budget_words = p.effective_budget(all_words);
    if all_words <= p.budget_words {
        let text = render_segments(state, &(0..state.segments.len() as u32).collect::<Vec<_>>());
        let w = text.split_whitespace().count();
        return (text, all_words, w);
    }
    let n_seg = state.segments.len();
    let BaseScores { q_scores, crit_scores, pooled } = base_scores(view, state, p.local_idf, p.q_expand);
    if let Some(r) = ranker {
        let feats = crate::retrieval::segment_features(view, state, &q_scores, &pooled, &crit_scores);
        let scores: Vec<f32> = feats.iter().map(|f| r.score(f)).collect();
        let mut order: Vec<u32> = (0..n_seg as u32).collect();
        order.sort_by(|a, b| {
            scores[*b as usize].partial_cmp(&scores[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
        });
        let mut taken = vec![false; n_seg];
        let mut chosen: Vec<u32> = Vec::new();
        let mut used = 0usize;
        let mut take = |seg: u32, taken: &mut Vec<bool>, chosen: &mut Vec<u32>, used: &mut usize| {
            if taken[seg as usize] {
                return;
            }
            let w = state.segment_text(seg).split_whitespace().count();
            if w == 0 || *used + w > p.budget_words {
                return;
            }
            taken[seg as usize] = true;
            chosen.push(seg);
            *used += w;
        };
        for seg in order {
            take(seg, &mut taken, &mut chosen, &mut used);
            // Keep the neighbourhood of a chosen segment intact: an
            // annotated span usually runs across several segments, and
            // taking only the individually best ones returns it in pieces.
            for d in 1..=p.neighbour_glue as u32 {
                if seg >= d {
                    take(seg - d, &mut taken, &mut chosen, &mut used);
                }
                if seg + d < n_seg as u32 {
                    take(seg + d, &mut taken, &mut chosen, &mut used);
                }
            }
            if used >= p.budget_words {
                break;
            }
        }
        chosen.sort_unstable();
        let text = render_segments(state, &chosen);
        return (text, all_words, used);
    }
    let words_of = |seg: u32| state.segment_text(seg).split_whitespace().count();
    let mut chosen: Vec<u32> = Vec::new();
    let mut used = 0usize;
    let take = |seg: u32, chosen: &mut Vec<u32>, used: &mut usize| -> bool {
        if chosen.contains(&seg) {
            return false;
        }
        let w = words_of(seg);
        if w == 0 || *used + w > p.budget_words {
            return false;
        }
        chosen.push(seg);
        *used += w;
        true
    };
    let ranked = |scores: &[f32]| -> Vec<u32> {
        let mut order: Vec<u32> = (0..n_seg as u32).filter(|i| scores[*i as usize] > 0.0).collect();
        order.sort_by(|a, b| scores[*b as usize].partial_cmp(&scores[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
        order
    };
    if p.strategy == Strategy::Quota {
        // 1. question-only evidence
        for seg in ranked(&q_scores).into_iter().take(p.question_slots * 3) {
            if chosen.len() >= p.question_slots {
                break;
            }
            take(seg, &mut chosen, &mut used);
        }
        // 2. round-robin reservations so no candidate is starved
        let per_crit: Vec<Vec<u32>> = crit_scores.iter().map(|s| ranked(s)).collect();
        for round in 0..p.per_criterion {
            for lists in per_crit.iter() {
                let mut placed = 0usize;
                for &seg in lists.iter() {
                    if placed > round {
                        break;
                    }
                    if !chosen.contains(&seg) {
                        take(seg, &mut chosen, &mut used);
                        break;
                    }
                    placed += 1;
                }
            }
        }
    }
    // 3. fill the remaining budget with the pooled ranking, then with
    //    unscored segments in document order (a needle with no lexical
    //    overlap is otherwise unreachable).
    for seg in ranked(&pooled) {
        if used >= p.budget_words {
            break;
        }
        take(seg, &mut chosen, &mut used);
        // Keep the neighbourhood of a chosen segment intact: an annotated
        // span usually runs across several segments, and taking only the
        // individually best ones returns it in pieces.
        for d in 1..=p.neighbour_glue as u32 {
            if seg >= d {
                take(seg - d, &mut chosen, &mut used);
            }
            if seg + d < n_seg as u32 {
                take(seg + d, &mut chosen, &mut used);
            }
        }
    }
    if p.doc_order_fallback && used < p.budget_words {
        for seg in 0..n_seg as u32 {
            if used >= p.budget_words {
                break;
            }
            take(seg, &mut chosen, &mut used);
        }
    }
    chosen.sort_unstable();
    let text = render_segments(state, &chosen);
    (text, all_words, used)
}

fn render_segments(state: &StateIndex, segs: &[u32]) -> String {
    let mut out = String::new();
    let mut last_field: Option<u32> = None;
    let mut last_seg: Option<u32> = None;
    for &seg in segs {
        let s = &state.segments[seg as usize];
        let text = state.segment_text(seg).trim();
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() {
            let contiguous = last_seg.map(|l| l + 1 == seg).unwrap_or(false);
            out.push_str(if contiguous { " " } else { " … " });
        }
        if last_field != Some(s.field) {
            let path = state.segment_path(seg);
            if !path.is_empty() {
                let label = crate::state::flatten::split_key_words(path.rsplit('.').next().unwrap_or(path)).join(" ");
                if !label.is_empty() {
                    out.push_str(&label);
                    out.push_str(": ");
                }
            }
            last_field = Some(s.field);
        }
        out.push_str(text);
        last_seg = Some(seg);
    }
    out
}

/// Build the export for one question against a prepared state index.
pub fn export_question(q: &Question, state: &StateIndex, res: &Resources, model: &crate::model::Model, budget_words: usize) -> QuestionExport {
    export_question_with(q, state, res, model, EvidenceParams::quota(budget_words))
}

pub fn export_question_with(q: &Question, state: &StateIndex, res: &Resources, model: &crate::model::Model, params: EvidenceParams) -> QuestionExport {
    let view = QuestionView::build(q, state, res);
    let feats = extract_features(&view, state, res);
    let resolved = resolvers::resolve(&view, state, &feats);
    // Training pairs must be retrieved exactly as inference retrieves them,
    // so the exporter uses the model's ranker when it has one.
    let (evidence, state_words, evidence_words) =
        select_evidence_ranked(&view, state, params, model.retrieval.as_deref());
    let head = match view.kind {
        QuestionKind::Choice => Some(&model.dense.choice),
        QuestionKind::Score => Some(&model.dense.score),
        QuestionKind::Noul => None,
    };
    let candidates: Vec<CandidateExport> = view
        .criteria
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let f: &FeatureVec = &feats.rows[i];
            let symbolic_logit = match head {
                Some(h) => h.score(f, view.family),
                None => {
                    if i == 0 {
                        model.dense.noul.logit(&feats.rows[0], &feats.rows[1], view.family)
                    } else {
                        0.0
                    }
                }
            };
            CandidateExport { key: c.key.clone(), text: candidate_text(&view, i), features: f.0.to_vec(), symbolic_logit }
        })
        .collect();
    let mut question = view.display_text.clone();
    if !view.context_text.is_empty() {
        question.push_str(" | ");
        question.push_str(&view.context_text);
    }
    QuestionExport {
        kind: view.kind.as_str().to_string(),
        family: view.family.as_str().to_string(),
        question,
        evidence,
        state_words,
        evidence_words,
        coverage: if state_words > 0 { evidence_words as f32 / state_words as f32 } else { 1.0 },
        candidates,
        path: match &resolved {
            Some(r) => format!("symbolic:{}", r.name),
            None => "semantic".to_string(),
        },
        symbolic_resolver_logits: resolved.map(|r| r.logits),
        feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
    }
}
