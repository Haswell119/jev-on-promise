//! Cross-field comparison channel. When a question references two state
//! fields ("Does `answer` address `question`?", "What is the relationship of
//! `hypothesis` to `premise`?") the decisive evidence is the RELATION between
//! those fields: lexical overlap, negation/antonym conflicts and number
//! mismatches between them.

use crate::lexicon::Resources;
use crate::question::QuestionView;
use crate::state::index::{char_grams, sorted_intersection, StateIndex};
use crate::state::vocab::TermId;
use crate::text::negation::NEGATED;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Default)]
pub struct CrossField {
    pub available: bool,
    pub cov_ab: f32,
    pub cov_ba: f32,
    pub jaccard: f32,
    pub cos: f32,
    pub gram: f32,
    pub neg_conflict: f32,
    pub antonym: f32,
    pub num_conflict: f32,
}

struct Side {
    /// term → (weight, negated occurrences, total occurrences)
    terms: FxHashMap<TermId, (f32, u32, u32)>,
    numbers: Vec<f64>,
    text: String,
    norm: f32,
}

fn side(state: &StateIndex, fields: &[u32]) -> Side {
    let mut terms: FxHashMap<TermId, (f32, u32, u32)> = FxHashMap::default();
    let mut numbers = Vec::new();
    let mut text = String::new();
    for &f in fields {
        let fld = &state.fields[f as usize];
        for seg in fld.seg_start..fld.seg_end {
            for tok in state.segment_tokens(seg) {
                if !tok.is_evidence() || tok.attrs & crate::state::index::ATTR_KEY != 0 {
                    continue;
                }
                let info = state.vocab.info(tok.term);
                if info.stop {
                    continue;
                }
                let e = terms.entry(tok.term).or_insert((info.idf.max(0.1), 0, 0));
                e.2 += 1;
                if tok.flags & NEGATED != 0 {
                    e.1 += 1;
                }
            }
            for n in state.numbers.iter().filter(|n| n.seg == seg && !n.from_word) {
                numbers.push(n.value);
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&state.segment_text(seg).to_lowercase());
        }
    }
    let norm = terms.values().map(|(w, _, c)| (w * (1.0 + (*c as f32).ln())).powi(2)).sum::<f32>().sqrt();
    Side { terms, numbers, text, norm }
}

/// Fields referenced by the question, in order of appearance (strong refs
/// first). Falls back to the first two text fields of an object state when
/// the question names none.
pub fn referenced_pair(q: &QuestionView, state: &StateIndex) -> Option<(Vec<u32>, Vec<u32>)> {
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for r in q.path_refs.iter().filter(|r| !r.weak && !r.fields.is_empty()) {
        if !groups.iter().any(|g| g == &r.fields) {
            groups.push(r.fields.clone());
        }
    }
    if groups.len() < 2 {
        for r in q.path_refs.iter().filter(|r| r.weak && !r.fields.is_empty()) {
            if !groups.iter().any(|g| g == &r.fields) {
                groups.push(r.fields.clone());
            }
        }
    }
    if groups.len() < 2 && !state.is_plain_text {
        let text_fields: Vec<u32> = state
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| f.kind == crate::state::flatten::FieldKind::Text && f.text.split_whitespace().count() >= 3)
            .map(|(i, _)| i as u32)
            .collect();
        if text_fields.len() == 2 && groups.is_empty() {
            groups.push(vec![text_fields[0]]);
            groups.push(vec![text_fields[1]]);
        }
    }
    if groups.len() >= 2 {
        Some((groups[0].clone(), groups[1].clone()))
    } else {
        None
    }
}

pub fn cross_field(q: &QuestionView, state: &StateIndex, res: &Resources) -> CrossField {
    let Some((fa, fb)) = referenced_pair(q, state) else { return CrossField::default() };
    if fa == fb {
        return CrossField::default();
    }
    let a = side(state, &fa);
    let b = side(state, &fb);
    if a.terms.is_empty() || b.terms.is_empty() {
        return CrossField { available: true, ..Default::default() };
    }
    let wa: f32 = a.terms.values().map(|x| x.0).sum();
    let wb: f32 = b.terms.values().map(|x| x.0).sum();
    let mut shared_a = 0.0f32;
    let mut shared_b = 0.0f32;
    let mut conflict = 0.0f32;
    let mut cos_num = 0.0f32;
    let mut inter = 0usize;
    for (t, (w, neg, n)) in &a.terms {
        if let Some((_, negb, nb)) = b.terms.get(t) {
            inter += 1;
            shared_a += w;
            let fa = *neg as f32 / (*n).max(1) as f32;
            let fb = *negb as f32 / (*nb).max(1) as f32;
            conflict += w * (fa - fb).abs();
            cos_num += (w * (1.0 + (*n as f32).ln())) * (w * (1.0 + (*nb as f32).ln()));
        }
    }
    for (t, (w, _, _)) in &b.terms {
        if a.terms.contains_key(t) {
            shared_b += w;
        }
    }
    // antonyms across sides (only when the term itself is absent on the other side)
    let mut antonym = 0.0f32;
    if !res.graph.is_empty() {
        for (t, (w, neg, n)) in &a.terms {
            if b.terms.contains_key(t) {
                continue;
            }
            let st = state.vocab.info(*t).stem.to_string();
            for ant in res.graph.antonym_stems(&st) {
                if let Some(aid) = state.vocab.lookup_stem(ant) {
                    if let Some((_, negb, nb)) = b.terms.get(&aid) {
                        let fa = *neg as f32 / (*n).max(1) as f32;
                        let fb = *negb as f32 / (*nb).max(1) as f32;
                        // an antonym asserted on both sides is a conflict; a negated antonym agrees
                        antonym += w * (1.0 - (fa - fb).abs());
                        break;
                    }
                }
            }
        }
    }
    let union = a.terms.len() + b.terms.len() - inter;
    let ga = char_grams(&a.text);
    let gb = char_grams(&b.text);
    let gram = if ga.is_empty() || gb.is_empty() {
        0.0
    } else {
        2.0 * sorted_intersection(&ga, &gb) as f32 / (ga.len() + gb.len()) as f32
    };
    let num_conflict = if a.numbers.is_empty() || b.numbers.is_empty() {
        0.0
    } else {
        let unmatched = a.numbers.iter().filter(|x| !b.numbers.iter().any(|y| (*x - y).abs() < 1e-9)).count();
        unmatched as f32 / a.numbers.len() as f32
    };
    CrossField {
        available: true,
        cov_ab: shared_a / wa.max(1e-6),
        cov_ba: shared_b / wb.max(1e-6),
        jaccard: if union > 0 { inter as f32 / union as f32 } else { 0.0 },
        cos: if a.norm > 0.0 && b.norm > 0.0 { cos_num / (a.norm * b.norm) } else { 0.0 },
        gram,
        neg_conflict: conflict / shared_a.max(1e-6) * (shared_a / wa.max(1e-6)),
        antonym: antonym / wa.max(1e-6),
        num_conflict,
    }
}
