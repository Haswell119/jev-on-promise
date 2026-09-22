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

/// Select evidence fragments for a question: BM25 over the pooled query
/// (question terms + all criterion terms), then the best segments in
/// document order up to a word budget. Short states are returned whole.
pub fn select_evidence(view: &QuestionView<'_>, state: &StateIndex, budget_words: usize) -> (String, usize, usize) {
    let all_words: usize = state.fields.iter().map(|f| f.text.split_whitespace().count()).sum();
    // Whole state when it already fits.
    if all_words <= budget_words {
        let text = render_segments(state, &(0..state.segments.len() as u32).collect::<Vec<_>>());
        let w = text.split_whitespace().count();
        return (text, all_words, w);
    }
    let mut scores = vec![0.0f32; state.segments.len()];
    let add = |term: crate::state::vocab::TermId, weight: f32, scores: &mut Vec<f32>| {
        if let Some(post) = state.postings.get(&term) {
            for &(seg, tf) in post {
                let s = &state.segments[seg as usize];
                let tf = tf as f32;
                let norm = tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * s.content_len as f32 / state.avg_seg_len));
                scores[seg as usize] += weight * norm;
            }
        }
    };
    for t in &view.terms {
        add(t.term, t.weight, &mut scores);
    }
    for c in &view.criteria {
        let w = 1.0 / (c.terms.len().max(1) as f32).sqrt();
        for t in c.terms.iter().filter(|t| t.polarity > 0) {
            add(t.term, t.weight * w, &mut scores);
        }
        for &(syn, weight, _) in c.expanded.iter().take(24) {
            add(syn, weight * 0.5 * w, &mut scores);
        }
    }
    // Focus fields (explicit path references) get a strong prior.
    if !view.focus_fields.is_empty() {
        for (i, seg) in state.segments.iter().enumerate() {
            if view.focus_fields.binary_search(&seg.field).is_ok() {
                scores[i] += 2.0;
            }
        }
    }
    let mut order: Vec<u32> = (0..state.segments.len() as u32).collect();
    order.sort_by(|a, b| scores[*b as usize].partial_cmp(&scores[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    let mut chosen: Vec<u32> = Vec::new();
    let mut used = 0usize;
    for seg in order {
        let w = state.segment_text(seg).split_whitespace().count();
        if w == 0 {
            continue;
        }
        if used + w > budget_words && !chosen.is_empty() {
            continue;
        }
        chosen.push(seg);
        used += w;
        if used >= budget_words {
            break;
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
    let view = QuestionView::build(q, state, res);
    let feats = extract_features(&view, state, res);
    let resolved = resolvers::resolve(&view, state, &feats);
    let (evidence, state_words, evidence_words) = select_evidence(&view, state, budget_words);
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
