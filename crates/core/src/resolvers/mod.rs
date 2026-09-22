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
        QuestionKind::Noul => reference_equality(q, state)
            .or_else(|| boolean_field(q, state))
            .or_else(|| numeric_comparison(q, state))
            .or_else(|| date_comparison(q, state)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cmp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

impl Cmp {
    fn eval(self, v: f64, thr: f64) -> bool {
        match self {
            Cmp::Gt => v > thr,
            Cmp::Ge => v >= thr,
            Cmp::Lt => v < thr,
            Cmp::Le => v <= thr,
            Cmp::Eq => (v - thr).abs() < 1e-9,
        }
    }
}

/// Parse a comparator + threshold out of the question text.
/// Returns (comparator, threshold, kind, unit word following the number).
fn parse_comparison(text: &str) -> Option<(Cmp, f64, NumKind, Option<String>)> {
    let toks = crate::text::tokenize::tokenize(text);
    let mut num_idx: Option<usize> = None;
    let mut value = 0.0f64;
    let mut kind = NumKind::Plain;
    for (i, t) in toks.iter().enumerate() {
        if matches!(t.kind, crate::text::tokenize::TokenKind::Number | crate::text::tokenize::TokenKind::Currency | crate::text::tokenize::TokenKind::Percent) {
            if let Some(p) = crate::text::numbers::parse_number(&t.text, t.kind) {
                if p.kind == NumKind::Ordinal {
                    continue;
                }
                num_idx = Some(i);
                value = p.value;
                kind = p.kind;
                break;
            }
        } else if t.kind == crate::text::tokenize::TokenKind::Word {
            if let Some(v) = crate::text::numbers::number_word(&t.text) {
                // only when preceded by a comparator word
                let prev: Vec<&str> = toks[..i].iter().rev().take(3).map(|x| x.text.as_str()).collect();
                if prev.iter().any(|w| matches!(*w, "than" | "least" | "most" | "over" | "under" | "above" | "below" | "exceed" | "exceeds" | "exactly")) {
                    num_idx = Some(i);
                    value = v;
                    break;
                }
            }
        }
    }
    let ni = num_idx?;
    let before: Vec<&str> = toks[..ni].iter().map(|t| t.text.as_str()).collect();
    let joined = before.join(" ");
    let has = |p: &str| joined.ends_with(p) || joined.contains(&format!("{p} ")) || joined.contains(p);
    let cmp = if has("greater than") || has("more than") || has("larger than") || has("higher than") || has("bigger than") || has("longer than") || has("over") || has("above") || has("exceed") || has("exceeds") || has("exceeding") {
        Cmp::Gt
    } else if has("at least") || has("minimum of") || has("no less than") || has("not less than") || has("or more") {
        Cmp::Ge
    } else if has("less than") || has("fewer than") || has("lower than") || has("smaller than") || has("shorter than") || has("under") || has("below") {
        Cmp::Lt
    } else if has("at most") || has("no more than") || has("not more than") || has("up to") || has("maximum of") || has("within") {
        Cmp::Le
    } else if has("exactly") || has("equal to") || has("equals") {
        Cmp::Eq
    } else {
        return None;
    };
    // unit word after the number ("units", "items", "days")
    let unit = toks.get(ni + 1).filter(|t| t.kind == crate::text::tokenize::TokenKind::Word && !crate::text::stem::is_function_word(&t.text) && !matches!(t.text.as_str(), "or" | "and" | "of" | "in" | "per" | "total")).map(|t| t.text.clone());
    Some((cmp, value, kind, unit))
}

/// Numeric comparison: "Is the invoice total greater than $500?",
/// "Were more than 10 units ordered?", "Does the order contain more than 2
/// line items?" (array length). Fires only when the referenced quantity is
/// unambiguous.
fn numeric_comparison(q: &QuestionView, state: &StateIndex) -> Option<Resolved> {
    let (cmp, thr, kind, unit) = parse_comparison(&q.text)?;
    let focus = &q.focus_fields;
    let unit_stem = unit.as_ref().map(|u| crate::text::stem::stem(u));
    let mut cands: Vec<(f64, String)> = Vec::new();
    // 1. Array lengths whose key matches the unit word ("items" → order.items).
    if let Some(us) = &unit_stem {
        for a in &state.arrays {
            let key_match = a.key_terms.iter().any(|t| state.vocab.info(*t).stem.as_ref() == us.as_str());
            if key_match {
                cands.push((a.len as f64, format!("len({})", a.path)));
            }
        }
        if !cands.is_empty() {
            return finish_comparison(q, cmp, thr, cands);
        }
    }
    // 2. Numbers in the state, filtered by kind, focus and adjacent unit word.
    for ns in &state.numbers {
        if ns.kind == NumKind::Ordinal || (ns.from_word && ns.value < 2.0) {
            continue;
        }
        let kind_ok = match kind {
            NumKind::Currency => ns.kind == NumKind::Currency,
            NumKind::Percent => ns.kind == NumKind::Percent,
            _ => ns.kind != NumKind::Currency && ns.kind != NumKind::Percent,
        };
        if !kind_ok {
            continue;
        }
        let field = state.segments[ns.seg as usize].field;
        if !focus.is_empty() && focus.binary_search(&field).is_err() {
            continue;
        }
        let toks = state.segment_tokens(ns.seg);
        let s = &state.segments[ns.seg as usize];
        let local = (ns.token - s.tok_start) as usize;
        let neighbor_matches = |us: &str| -> bool {
            for off in [-2i64, -1, 1, 2] {
                let j = local as i64 + off;
                if j < 0 || j as usize >= toks.len() {
                    continue;
                }
                let info = state.vocab.info(toks[j as usize].term);
                if info.stem.as_ref() == us {
                    return true;
                }
            }
            // JSON key of the field ("quantity": 12 → unit "units"/"quantity")
            let f = &state.fields[field as usize];
            f.key_terms.iter().any(|t| state.vocab.info(*t).stem.as_ref() == us)
        };
        if let Some(us) = &unit_stem {
            if !neighbor_matches(us) {
                // synonyms of the unit word (units ≈ items ≈ pieces)
                let syn_ok = matches!(us.as_str(), "unit" | "item" | "piec" | "articl" | "product") && (neighbor_matches("unit") || neighbor_matches("item") || neighbor_matches("quantiti") || neighbor_matches("qty") || neighbor_matches("piec"));
                if !syn_ok {
                    continue;
                }
            }
        } else if kind == NumKind::Plain {
            // plain-number question ("greater than 500") against currency-only states: allow currency
        }
        cands.push((ns.value, state.fields[field as usize].path.clone()));
    }
    if cands.is_empty() && kind == NumKind::Plain && unit.is_none() {
        // fall back to currency amounts when the question has a bare number
        for ns in &state.numbers {
            if ns.kind == NumKind::Currency {
                let field = state.segments[ns.seg as usize].field;
                if focus.is_empty() || focus.binary_search(&field).is_ok() {
                    cands.push((ns.value, state.fields[field as usize].path.clone()));
                }
            }
        }
    }
    if cands.is_empty() {
        return None;
    }
    finish_comparison(q, cmp, thr, cands)
}

fn finish_comparison(q: &QuestionView, cmp: Cmp, thr: f64, cands: Vec<(f64, String)>) -> Option<Resolved> {
    let outcomes: Vec<bool> = cands.iter().map(|(v, _)| cmp.eval(*v, thr)).collect();
    let all_true = outcomes.iter().all(|x| *x);
    let all_false = outcomes.iter().all(|x| !*x);
    if !(all_true || all_false) {
        return None;
    }
    let truth = all_true != q.question_negated;
    let logit = if truth { STRONG * 0.8 } else { -STRONG * 0.8 };
    let notes = cands.iter().map(|(v, p)| format!("{p}: {v} {:?} {thr} → {}", cmp, cmp.eval(*v, thr))).collect();
    Some(Resolved { name: "numeric_comparison", logits: vec![logit], notes })
}

/// Date comparison: "Is the date mentioned before 2026-04-01?",
/// "Was the order placed after March 3, 2026?". Fires when every date in
/// the (focused) state agrees on the outcome.
fn date_comparison(q: &QuestionView, state: &StateIndex) -> Option<Resolved> {
    use crate::text::dates::{parse_date_token, parse_textual_date};
    let toks = crate::text::tokenize::tokenize(&q.text);
    let words: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
    // find a date in the question
    let mut qdate = None;
    let mut date_pos = 0usize;
    for (i, t) in toks.iter().enumerate() {
        if t.kind == crate::text::tokenize::TokenKind::Date {
            if let Some(d) = parse_date_token(&t.text) {
                qdate = Some(d);
                date_pos = i;
                break;
            }
        }
        if t.kind == crate::text::tokenize::TokenKind::Word && crate::text::dates::month_from_name(&t.text).is_some() {
            if let Some((d, _)) = parse_textual_date(&words, i, 2000) {
                qdate = Some(d);
                date_pos = i;
                break;
            }
        }
        if t.kind == crate::text::tokenize::TokenKind::Number {
            if let Some((d, _)) = parse_textual_date(&words, i, 2000) {
                if words.get(i + 1).map(|w| crate::text::dates::month_from_name(w).is_some()).unwrap_or(false) {
                    qdate = Some(d);
                    date_pos = i;
                    break;
                }
            }
        }
    }
    let qdate = qdate?;
    let before_words: Vec<&str> = words[..date_pos].iter().copied().rev().take(4).collect();
    let cmp = if before_words.iter().any(|w| matches!(*w, "before" | "earlier" | "prior" | "by" | "until")) {
        Cmp::Lt
    } else if before_words.iter().any(|w| matches!(*w, "after" | "later" | "since" | "past" | "beyond")) {
        Cmp::Gt
    } else if before_words.iter().any(|w| matches!(*w, "on" | "exactly")) {
        Cmp::Eq
    } else {
        return None;
    };
    let inclusive = before_words.iter().any(|w| matches!(*w, "by" | "until")) || q.text.contains("or before") || q.text.contains("or after") || q.text.contains("or later") || q.text.contains("or earlier");
    let focus = &q.focus_fields;
    let mut outcomes = Vec::new();
    let mut notes = Vec::new();
    for ds in &state.dates {
        let field = state.segments[ds.seg as usize].field;
        if !focus.is_empty() && focus.binary_search(&field).is_err() {
            continue;
        }
        let (a, b) = (ds.date.days_from_epoch(), qdate.days_from_epoch());
        let r = match cmp {
            Cmp::Lt => if inclusive { a <= b } else { a < b },
            Cmp::Gt => if inclusive { a >= b } else { a > b },
            _ => a == b,
        };
        notes.push(format!("{}-{:02}-{:02} {:?} {}-{:02}-{:02} → {}", ds.date.year, ds.date.month, ds.date.day, cmp, qdate.year, qdate.month, qdate.day, r));
        outcomes.push(r);
    }
    if outcomes.is_empty() {
        return None;
    }
    let all_true = outcomes.iter().all(|x| *x);
    let all_false = outcomes.iter().all(|x| !*x);
    if !(all_true || all_false) {
        return None;
    }
    let truth = all_true != q.question_negated;
    let logit = if truth { STRONG * 0.8 } else { -STRONG * 0.8 };
    Some(Resolved { name: "date_comparison", logits: vec![logit], notes })
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
        if ns.kind == NumKind::Ordinal || ns.from_word {
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
