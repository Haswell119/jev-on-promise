//! Shared feature extraction for training / calibration: turns dataset
//! records into (features, gold) examples using the exact runtime pipeline.

use super::dataset::Record;
use sextant_core::api::{Question, QuestionKind};
use sextant_core::features::{extract_features, FeatureVec, N_FEATURES};
use sextant_core::lexicon::Resources;
use sextant_core::question::{Family, QuestionView};
use sextant_core::resolvers;
use sextant_core::state::index::StateIndex;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Example {
    pub record_id: String,
    pub kind: QuestionKind,
    pub family: Family,
    /// Per-criterion feature rows (2 rows for Noul: yes, no).
    pub rows: Vec<FeatureVec>,
    /// Gold criterion index (Choice/Score) or 1/0 for Noul (yes = 1).
    pub gold: usize,
    /// Symbolic logits when a resolver fired.
    pub symbolic: Option<Vec<f32>>,
    pub soft: Option<Vec<f64>>,
}

pub fn gold_index(q: &Question, gold: &Value) -> Option<usize> {
    match q {
        Question::Choice(c) => {
            let key = match gold {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            c.criteria.get_index_of(&key)
        }
        Question::Score(s) => {
            let idx = match gold {
                Value::Number(n) => n.as_u64().map(|x| x as usize),
                Value::String(st) => st.parse::<usize>().ok(),
                _ => None,
            }?;
            (idx < s.criteria.len()).then_some(idx)
        }
        Question::Noul(_) => match gold {
            Value::Bool(b) => Some(if *b { 1 } else { 0 }),
            Value::String(s) => Some(if matches!(s.to_ascii_lowercase().as_str(), "yes" | "true" | "1") { 1 } else { 0 }),
            Value::Number(n) => Some(if n.as_f64().unwrap_or(0.0) >= 0.5 { 1 } else { 0 }),
            _ => None,
        },
    }
}

fn soft_labels(q: &Question, gp: Option<&Value>) -> Option<Vec<f64>> {
    let gp = gp?;
    match q {
        Question::Choice(c) => {
            let o = gp.as_object()?;
            Some(c.criteria.keys().map(|k| o.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)).collect())
        }
        Question::Score(s) => match gp {
            Value::Array(a) => Some((0..s.criteria.len()).map(|i| a.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0)).collect()),
            Value::Object(o) => Some((0..s.criteria.len()).map(|i| o.get(&i.to_string()).and_then(|v| v.as_f64()).unwrap_or(0.0)).collect()),
            _ => None,
        },
        Question::Noul(_) => match gp {
            Value::Number(n) => n.as_f64().map(|p| vec![1.0 - p, p]),
            Value::Object(o) => o.get("yes").and_then(|v| v.as_f64()).map(|p| vec![1.0 - p, p]),
            _ => None,
        },
    }
}

/// Extract examples from records (in order). Records whose gold cannot be
/// mapped are skipped and counted.
pub fn extract_examples(records: &[Record], res: &Resources, drop: &[usize]) -> (Vec<Example>, usize) {
    use rayon::prelude::*;
    let results: Vec<(Vec<Example>, usize)> = records
        .par_iter()
        .map(|rec| {
            let mut out = Vec::new();
            let mut skipped = 0usize;
            let state = StateIndex::build(&rec.request.state, res);
            for (qid, q) in &rec.request.questions {
                let Some(gold) = rec.gold.get(qid) else { continue };
                let Some(gi) = gold_index(q, gold) else {
                    skipped += 1;
                    continue;
                };
                let view = QuestionView::build(q, &state, res);
                let mut feats = extract_features(&view, &state, res);
                for row in feats.rows.iter_mut() {
                    for &d in drop {
                        row.0[d] = 0.0;
                    }
                }
                let symbolic = resolvers::resolve(&view, &state, &feats).map(|r| r.logits);
                let soft = soft_labels(q, rec.gold_probs.as_ref().and_then(|m| m.get(qid)));
                out.push(Example { record_id: rec.id.clone(), kind: q.kind(), family: view.family, rows: feats.rows, gold: gi, symbolic, soft });
            }
            (out, skipped)
        })
        .collect();
    let mut all = Vec::new();
    let mut skipped = 0;
    for (v, s) in results {
        all.extend(v);
        skipped += s;
    }
    (all, skipped)
}

pub fn feature_index(name: &str) -> Option<usize> {
    sextant_core::features::FEATURE_NAMES.iter().position(|n| *n == name)
}

pub fn parse_drop(spec: &Option<String>) -> Vec<usize> {
    spec.as_ref()
        .map(|s| s.split(',').filter_map(|n| feature_index(n.trim())).collect())
        .unwrap_or_default()
}

pub const _N: usize = N_FEATURES;
