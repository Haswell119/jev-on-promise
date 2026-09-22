//! Symbolic fast paths. When a question is mechanically answerable from the
//! state (a literal option appears verbatim, a number falls into exactly one
//! level range, a referenced value is present or absent, a boolean field
//! answers the question), ordinary code decides and the semantic ensemble
//! is only used to fill in the residual distribution.
//!
//! A resolver returns LOGITS over the criteria (or a single logit for
//! Noul). They are calibrated separately (`symbolic_temperature`,
//! `noul_symbolic_platt`), so their sharpness is learned, not asserted.

use crate::api::QuestionKind;
use crate::features::{FeatureMatrix, F};
use crate::question::{Family, QuestionView};
use crate::state::flatten::FieldKind;
use crate::state::index::StateIndex;
use crate::text::numbers::NumKind;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Resolved {
    /// Resolver name (reported in explain output as `symbolic:<name>`).
    pub name: &'static str,
    /// Logits per criterion (Choice/Score) or `[logit_true]` for Noul.
    pub logits: Vec<f32>,
    pub notes: Vec<String>,
}

pub const STRONG: f32 = 6.0;
pub const PARTIAL: f32 = 2.0;

/// Try every applicable resolver in a fixed order. Returns the first that fires.
pub fn resolve(q: &QuestionView, state: &StateIndex, feats: &FeatureMatrix) -> Option<Resolved> {
    match q.kind {
        QuestionKind::Choice => enum_extraction(q, state, feats).or_else(|| numeric_levels(q, state)),
        QuestionKind::Score => numeric_levels(q, state),
        QuestionKind::Noul => reference_equality(q, state).or_else(|| boolean_field(q, state)),
    }
}

/// Exact enum extraction: options are literal values; the longest option
/// literal that appears verbatim in the (focused) state wins.
fn enum_extraction(q: &QuestionView, _state: &StateIndex, feats: &FeatureMatrix) -> Option<Resolved> {
    if !(q.literal_options || q.family == Family::EnumExtraction) || !q.literal_options {
        return None;
    }
    let mut best_len = 0.0f32;
    let mut hits: Vec<(usize, f32, f32)> = Vec::new(); // (index, len, focus)
    for (i, f) in feats.rows.iter().enumerate() {
        if f.get(F::literal_hit) > 0.0 {
            let len = f.get(F::literal_len) + 0.05 * f.get(F::focus_literal);
            hits.push((i, len, f.get(F::focus_literal)));
            if len > best_len {
                best_len = len;
            }
        }
    }
    if hits.is_empty() {
        return None;
    }
    let winners: Vec<usize> = hits.iter().filter(|(_, l, _)| (*l - best_len).abs() < 1e-6).map(|(i, _, _)| *i).collect();
    if winners.len() != 1 {
        return None;
    }
    let w = winners[0];
    let mut logits = vec![0.0f32; feats.rows.len()];
    for (i, len, _) in &hits {
        logits[*i] = if *i == w { STRONG } else { PARTIAL * (len / best_len) };
    }
    Some(Resolved { name: "enum_extraction", logits, notes: vec![format!("option `{}` appears verbatim as the longest literal match", q.criteria[w].key)] })
}

/// Numeric level/option ranges: pick the criterion whose range contains the
/// referenced number. Fires only when the number is unambiguous.
fn numeric_levels(q: &QuestionView, state: &StateIndex) -> Option<Resolved> {
    let n = q.criteria.len();
    let with_range = q.criteria.iter().filter(|c| c.range.is_some()).count();
    if n < 2 || with_range < n.saturating_sub(1) || with_range < 2 {
        return None;
    }
    let ranges: Vec<Option<&crate::question::NumericRange>> = q.criteria.iter().map(|c| c.range.as_ref()).collect();
    let kind = ranges.iter().flatten().map(|r| r.kind).find(|k| *k != NumKind::Plain).unwrap_or(NumKind::Plain);
    let unit: Option<String> = ranges.iter().flatten().find_map(|r| r.unit.clone());
    // Candidate numbers: focus first, then unit-matching, then kind-matching.
    let focus = &q.focus_fields;
    let mut cands: Vec<f64> = Vec::new();
    for ns in &state.numbers {
        if ns.kind == NumKind::Ordinal {
            continue;
        }
        let kind_ok = match kind {
            NumKind::Currency => ns.kind == NumKind::Currency,
            NumKind::Percent => ns.kind == NumKind::Percent,
            _ => true,
        };
        if !kind_ok {
            continue;
        }
        let field = state.segments[ns.seg as usize].field;
        let in_focus = focus.is_empty() || focus.binary_search(&field).is_ok();
        if !in_focus {
            continue;
        }
        if let Some(u) = &unit {
            // require the unit word next to the number
            let toks = state.segment_tokens(ns.seg);
            let s = &state.segments[ns.seg as usize];
            let local = (ns.token - s.tok_start) as usize;
            let near = |off: i64| -> bool {
                let j = local as i64 + off;
                if j < 0 || j as usize >= toks.len() {
                    return false;
                }
                let t = &toks[j as usize];
                let info = state.vocab.info(t.term);
                info.stem.as_ref() == crate::text::stem::stem(u).as_str() || info.surface.as_ref() == u.as_str()
            };
            if !(near(1) || near(2) || near(-1)) {
                continue;
            }
        }
        cands.push(ns.value);
    }
    if cands.is_empty() && unit.is_some() && focus.is_empty() {
        return None;
    }
    if cands.is_empty() {
        return None;
    }
    // Every candidate must land in the same criterion.
    let mut chosen: Option<usize> = None;
    for v in &cands {
        let mut hit: Option<usize> = None;
        for (i, r) in ranges.iter().enumerate() {
            if let Some(r) = r {
                if r.contains(*v) {
                    hit = Some(i);
                    break;
                }
            }
        }
        match (chosen, hit) {
            (None, Some(h)) => chosen = Some(h),
            (Some(c), Some(h)) if c != h => return None,
            (_, None) => return None,
            _ => {}
        }
    }
    let c = chosen?;
    let v = cands[0];
    let mut logits = vec![0.0f32; n];
    for (i, r) in ranges.iter().enumerate() {
        logits[i] = if i == c {
            STRONG
        } else if let Some(r) = r {
            -(r.distance(v).min(1.0) as f32) * PARTIAL
        } else {
            -PARTIAL
        };
    }
    Some(Resolved { name: "numeric_range", logits, notes: vec![format!("value {v} falls in level `{}`", q.criteria[c].key)] })
}

fn scalar_literals(v: &Value, out: &mut Vec<String>, depth: usize) {
    if depth > 6 {
        return;
    }
    match v {
        Value::String(s) => {
            let lit = crate::question::criteria::normalize_literal(s);
            if lit.len() >= 3 {
                out.push(lit);
            }
        }
        Value::Number(n) => out.push(n.to_string()),
        Value::Array(a) => {
            for x in a {
                scalar_literals(x, out, depth + 1);
            }
        }
        Value::Object(o) => {
            for x in o.values() {
                scalar_literals(x, out, depth + 1);
            }
        }
        _ => {}
    }
}

/// Reference equality: the instruction carries data values and asks whether
/// they match / appear in the state. Specific values (with digits or ≥ 4
/// chars) present verbatim → true; absent → false.
fn reference_equality(q: &QuestionView, state: &StateIndex) -> Option<Resolved> {
    let asks_match = ["match", "matches", "same", "equal", "equals", "correct", "appear", "appears", "contain", "contains", "present", "mention", "mentioned", "consistent", "identical", "found"]
        .iter()
        .any(|w| q.text.split(|c: char| !c.is_alphanumeric()).any(|t| t == *w));
    if !asks_match {
        return None;
    }
    // Only data fields referenced from the question text count.
    let mut lits: Vec<String> = Vec::new();
    for r in &q.path_refs {
        if let Some(d) = &r.data {
            scalar_literals(d, &mut lits, 0);
        }
    }
    lits.retain(|l| l.chars().any(|c| c.is_ascii_digit()) || l.len() >= 4);
    lits.dedup();
    if lits.is_empty() {
        return None;
    }
    let text = &state.lower_text;
    let focus = &q.focus_fields;
    let mut present = 0usize;
    let mut notes = Vec::new();
    for l in &lits {
        let mut found = false;
        let mut start = 0usize;
        while let Some(pos) = text[start..].find(l.as_str()) {
            let abs = start + pos;
            let end = abs + l.len();
            let before = text[..abs].chars().next_back().map(|c| !c.is_alphanumeric()).unwrap_or(true);
            let after = text[end..].chars().next().map(|c| !c.is_alphanumeric()).unwrap_or(true);
            if before && after && (focus.is_empty() || focus.binary_search(&state.field_at_offset(abs)).is_ok()) {
                found = true;
                break;
            }
            start = end.max(abs + 1);
            if start >= text.len() {
                break;
            }
        }
        notes.push(format!("`{l}` {}", if found { "present" } else { "absent" }));
        if found {
            present += 1;
        }
    }
    let logit = if present == lits.len() {
        STRONG * 0.8
    } else if present == 0 {
        -STRONG * 0.6
    } else {
        return None;
    };
    Some(Resolved { name: "reference_equality", logits: vec![logit], notes })
}

/// Boolean field: the question names a boolean/falsey field of the state
/// ("Is the order paid?" with {"paid": false}).
fn boolean_field(q: &QuestionView, state: &StateIndex) -> Option<Resolved> {
    if state.is_plain_text {
        return None;
    }
    let qterms: Vec<_> = q.terms.iter().map(|t| t.term).collect();
    if qterms.is_empty() {
        return None;
    }
    let mut matches: Vec<(u32, bool)> = Vec::new();
    for (i, f) in state.fields.iter().enumerate() {
        if f.kind != FieldKind::Bool {
            continue;
        }
        if f.key_terms.is_empty() {
            continue;
        }
        let key_words: Vec<_> = f.key_terms.iter().filter(|t| !state.vocab.info(**t).func).collect();
        if key_words.is_empty() {
            continue;
        }
        let covered = key_words.iter().filter(|t| qterms.contains(t)).count();
        if covered == key_words.len() {
            matches.push((i as u32, f.boolean.unwrap_or(false)));
        }
    }
    if matches.len() != 1 {
        return None;
    }
    let (fi, val) = matches[0];
    let logit = if val != q.question_negated { STRONG * 0.8 } else { -STRONG * 0.8 };
    Some(Resolved { name: "boolean_field", logits: vec![logit], notes: vec![format!("field `{}` is {}", state.fields[fi as usize].path, val)] })
}
