//! Per-question analysis: flatten instructions into question text, context
//! text and reference data; resolve path references; extract weighted
//! question terms; detect the family; build the criteria.

use super::criteria::{Criterion, QueryTerm};
use super::family::{self, Family};
use crate::api::{NoulCriteria, Question, QuestionKind};
use crate::lexicon::Resources;
use crate::state::flatten::split_key_words;
use crate::state::index::StateIndex;
use crate::state::vocab::{TermId, VocabExt};
use crate::text::negation::{self, CUE, HYPOTHETICAL, NEGATED, REQUEST};
use crate::text::normalize::normalize_nfkc;
use crate::text::tokenize::tokenize;
use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct PathRef {
    pub name: String,
    /// State fields the reference resolves to (empty for instruction data refs).
    pub fields: Vec<u32>,
    /// The reference names a field of the instruction object (reference data).
    pub data: Option<Value>,
    /// Weak reference: a bare word that happens to equal a state key.
    pub weak: bool,
}

pub struct QuestionView<'a> {
    pub kind: QuestionKind,
    /// Lowercase normalized question text (primary instruction).
    pub text: String,
    /// Human readable instruction (for explain output).
    pub display_text: String,
    /// Context text from non-question instruction fields.
    pub context_text: String,
    pub terms: Vec<QueryTerm>,
    pub path_refs: Vec<PathRef>,
    /// Union of state fields referenced by strong path references.
    pub focus_fields: Vec<u32>,
    pub ref_data: Vec<(String, Value)>,
    pub family: Family,
    pub criteria: Vec<Criterion>,
    pub vocab: VocabExt<'a>,
    pub question_negated: bool,
    pub literal_options: bool,
    pub valence: f32,
    pub intensity: f32,
    /// Question asks about the state as a whole vs. a referenced sub-field.
    pub has_focus: bool,
    /// Question is phrased as a request-detection question ("does X ask for").
    pub asks_about_request: bool,
    /// Question is phrased hypothetically ("would", "could", "eligible").
    pub asks_hypothetical: bool,
}

const QUESTION_KEYS: &[&str] = &[
    "question",
    "questions",
    "instructions",
    "instruction",
    "ask",
    "prompt",
    "task",
    "query",
    "goal",
    "rubric",
    "guidance",
    "note",
    "notes",
    "focus",
    "criteria",
    "criterion",
    "context",
    "description",
    "hint",
    "hints",
    "decide",
    "evaluate",
    "judge",
    "assess",
    "compare",
    "check",
    "requirement",
    "requirements",
    "definition",
    "scope",
    "consider",
    "explanation",
    "detail",
    "details",
    "background",
    "policy",
    "rules",
    "rule",
];

fn looks_like_question(s: &str) -> bool {
    let t = s.trim().to_ascii_lowercase();
    t.ends_with('?')
        || t.split_whitespace()
            .next()
            .map(|w| {
                negation::WH_WORDS.contains(&w)
                    || negation::AUX_VERBS.contains(&w)
                    || matches!(
                        w,
                        "rate"
                            | "classify"
                            | "select"
                            | "choose"
                            | "pick"
                            | "decide"
                            | "determine"
                            | "identify"
                            | "evaluate"
                            | "judge"
                            | "assess"
                            | "score"
                            | "estimate"
                    )
            })
            .unwrap_or(false)
}

#[derive(Default)]
struct Flat {
    question: Vec<String>,
    context: Vec<(String, String)>,
    data: Vec<(String, Value)>,
}

fn collect_strings(v: &Value, prefix: &str, out: &mut Vec<(String, String)>, depth: usize) {
    if depth > 10 {
        return;
    }
    match v {
        Value::String(s) => out.push((prefix.to_string(), s.clone())),
        Value::Number(n) => out.push((prefix.to_string(), n.to_string())),
        Value::Bool(b) => out.push((prefix.to_string(), b.to_string())),
        Value::Array(a) => {
            for x in a {
                collect_strings(x, prefix, out, depth + 1);
            }
        }
        Value::Object(o) => {
            for (k, x) in o {
                let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                collect_strings(x, &p, out, depth + 1);
            }
        }
        Value::Null => {}
    }
}

fn flatten_instructions(v: &Value, out: &mut Flat, depth: usize) {
    if depth > 10 {
        return;
    }
    match v {
        Value::String(s) => out.question.push(s.clone()),
        Value::Array(a) => {
            for x in a {
                flatten_instructions(x, out, depth + 1);
            }
        }
        Value::Object(o) => {
            for (k, x) in o {
                let kl = k.to_ascii_lowercase();
                let is_q_key = QUESTION_KEYS.contains(&kl.as_str())
                    || split_key_words(&kl).iter().any(|w| QUESTION_KEYS.contains(&w.as_str()));
                match x {
                    Value::String(s) => {
                        if is_q_key || looks_like_question(s) {
                            out.question.push(s.clone());
                        } else {
                            out.context.push((k.clone(), s.clone()));
                            out.data.push((k.clone(), x.clone()));
                        }
                    }
                    Value::Number(_) | Value::Bool(_) | Value::Null => {
                        out.context.push((k.clone(), x.to_string()));
                        out.data.push((k.clone(), x.clone()));
                    }
                    Value::Array(_) | Value::Object(_) => {
                        if is_q_key && matches!(x, Value::Array(_)) {
                            flatten_instructions(x, out, depth + 1);
                        } else {
                            let mut strs = Vec::new();
                            collect_strings(x, k, &mut strs, 0);
                            out.context.extend(strs);
                            out.data.push((k.clone(), x.clone()));
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

fn backtick_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`([^`]{1,120})`").expect("backtick regex"))
}

fn literal_like(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.split_whitespace().count() <= 3,
        _ => false,
    }
}

impl<'a> QuestionView<'a> {
    pub fn build(q: &Question, state: &'a StateIndex, res: &Resources) -> QuestionView<'a> {
        let mut vocab = VocabExt::new(&state.vocab);
        let mut flat = Flat::default();
        flatten_instructions(q.instructions(), &mut flat, 0);
        let display_text = flat.question.join(" ");
        let text_norm = normalize_nfkc(&display_text).to_lowercase();
        let context_text = flat.context.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(" | ");

        // Path references (backticks) across question + context.
        let mut path_refs: Vec<PathRef> = Vec::new();
        let full_for_refs = format!("{display_text} {context_text}");
        for cap in backtick_re().captures_iter(&full_for_refs) {
            let name = cap[1].trim().to_string();
            if path_refs.iter().any(|p| p.name == name) {
                continue;
            }
            let fields = state.fields_for_path(&name);
            let data = flat.data.iter().find(|(k, _)| k == &name).map(|(_, v)| v.clone());
            if !fields.is_empty() || data.is_some() {
                path_refs.push(PathRef { name, fields, data, weak: false });
            }
        }
        // Weak refs: bare words equal to state keys.
        if !state.is_plain_text {
            let words: Vec<&str> =
                text_norm.split(|c: char| !c.is_alphanumeric() && c != '_').filter(|w| w.len() >= 3).collect();
            for f in &state.fields {
                if f.key.is_empty() {
                    continue;
                }
                let kl = f.key.to_ascii_lowercase();
                if words.iter().any(|w| *w == kl) && !path_refs.iter().any(|p| p.name.eq_ignore_ascii_case(&f.key)) {
                    let fields = state.fields_for_path(&f.key);
                    if !fields.is_empty() {
                        path_refs.push(PathRef { name: f.key.clone(), fields, data: None, weak: true });
                    }
                }
            }
        }
        let mut focus_fields: Vec<u32> =
            path_refs.iter().filter(|p| !p.weak).flat_map(|p| p.fields.iter().copied()).collect();
        focus_fields.sort_unstable();
        focus_fields.dedup();
        let has_focus = !focus_fields.is_empty();

        // Question terms.
        let mut terms: Vec<QueryTerm> = Vec::new();
        let mut question_negated = false;
        let mut asks_about_request = false;
        let mut asks_hypothetical = false;
        let raw_q = tokenize(&text_norm);
        let fl = negation::annotate(&raw_q);
        let (valence, intensity, _) = super::criteria::valence_and_intensity(&raw_q, &fl, res);
        let add_terms = |raw: &[crate::text::tokenize::RawToken],
                         flags: &[u16],
                         factor: f32,
                         terms: &mut Vec<QueryTerm>,
                         vocab: &mut VocabExt<'a>| {
            for (i, rt) in raw.iter().enumerate() {
                if !rt.kind.is_content() {
                    continue;
                }
                let tid = vocab.intern(&rt.text, res);
                let info = vocab.info(tid);
                if info.func || info.stop || flags[i] & CUE != 0 {
                    continue;
                }
                let negated = flags[i] & NEGATED != 0;
                if let Some(e) = terms.iter_mut().find(|t: &&mut QueryTerm| t.term == tid) {
                    e.weight = e.weight.max(info.idf.max(0.1) * factor);
                    continue;
                }
                terms.push(QueryTerm {
                    term: tid,
                    weight: info.idf.max(0.1) * factor,
                    negated,
                    hypothetical: flags[i] & HYPOTHETICAL != 0,
                    request: flags[i] & REQUEST != 0,
                    polarity: 1,
                    is_name: false,
                    idf: info.idf,
                });
            }
        };
        add_terms(&raw_q, &fl, 1.0, &mut terms, &mut vocab);
        for (i, rt) in raw_q.iter().enumerate() {
            if fl[i] & NEGATED != 0 && rt.kind.is_content() {
                let tid = vocab.intern(&rt.text, res);
                if !vocab.info(tid).func {
                    question_negated = true;
                }
            }
            if fl[i] & REQUEST != 0
                || matches!(
                    rt.text.as_str(),
                    "request"
                        | "requests"
                        | "requested"
                        | "requesting"
                        | "ask"
                        | "asks"
                        | "asked"
                        | "asking"
                        | "demand"
                        | "demands"
                )
            {
                asks_about_request = true;
            }
            if matches!(rt.text.as_str(), "would" | "could" | "eligible" | "hypothetically" | "possible" | "might") {
                asks_hypothetical = true;
            }
        }
        if !context_text.is_empty() {
            let ctx_norm = normalize_nfkc(&flat.context.iter().map(|(_, v)| v.as_str()).collect::<Vec<_>>().join(". "))
                .to_lowercase();
            let raw_c = tokenize(&ctx_norm);
            let fl_c = negation::annotate(&raw_c);
            add_terms(&raw_c, &fl_c, 0.5, &mut terms, &mut vocab);
        }

        // Criteria.
        let (criteria, literal_options, level_words) = match q {
            Question::Choice(c) => {
                let literal = c.criteria.values().all(literal_like);
                let crits: Vec<Criterion> = c
                    .criteria
                    .iter()
                    .enumerate()
                    .map(|(i, (k, v))| Criterion::build(k, i, v, true, &mut vocab, res))
                    .collect();
                (crits, literal, String::new())
            }
            Question::Score(s) => {
                let crits: Vec<Criterion> = s
                    .criteria
                    .iter()
                    .enumerate()
                    .map(|(i, v)| Criterion::build(&i.to_string(), i, v, false, &mut vocab, res))
                    .collect();
                let words = crits.iter().map(|c| c.full_text.to_lowercase()).collect::<Vec<_>>().join(" ");
                (crits, false, words)
            }
            Question::Noul(n) => {
                let NoulCriteria { yes, no } = n.criteria.clone().unwrap_or_default();
                let yes_v = yes.unwrap_or(Value::Null);
                let no_v = no.unwrap_or(Value::Null);
                let mut h_yes = Criterion::build("true", 0, &yes_v, false, &mut vocab, res);
                let mut h_no = Criterion::build("false", 1, &no_v, false, &mut vocab, res);
                // Proposition terms: the question's own content, positive for
                // the yes hypothesis and polarity-flipped for the no hypothesis.
                for t in &terms {
                    let mut ty = t.clone();
                    ty.polarity = 1;
                    if !h_yes.terms.iter().any(|e| e.term == ty.term) {
                        h_yes.pos_weight += ty.weight;
                        h_yes.n_pos_terms += 1;
                        h_yes.terms.push(ty);
                    }
                    let mut tn = t.clone();
                    tn.polarity = 1;
                    tn.negated = !t.negated;
                    if !h_no.terms.iter().any(|e| e.term == tn.term) {
                        h_no.pos_weight += tn.weight;
                        h_no.n_pos_terms += 1;
                        h_no.terms.push(tn);
                    }
                }
                (vec![h_yes, h_no], false, String::new())
            }
        };

        let n_options = criteria.len();
        let family = family::detect(q.kind(), &text_norm, n_options, literal_options, &level_words);

        QuestionView {
            kind: q.kind(),
            text: text_norm,
            display_text,
            context_text,
            terms,
            path_refs,
            focus_fields,
            ref_data: flat.data,
            family,
            criteria,
            vocab,
            question_negated,
            literal_options,
            valence,
            intensity,
            has_focus,
            asks_about_request,
            asks_hypothetical,
        }
    }

    /// Term ids of the question's evidence terms.
    pub fn term_ids(&self) -> impl Iterator<Item = TermId> + '_ {
        self.terms.iter().map(|t| t.term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn q(v: Value) -> Question {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn structured_instructions_and_refs() {
        let res = Resources::empty();
        let state = StateIndex::build(
            &json!({"source_text": "Invoice #4471 issued March 3, 2026 to Beaver Dam Logistics", "other": "x"}),
            &res,
        );
        let question = q(json!({
            "type": "noul",
            "instructions": {
                "field": {"name": "invoice_number", "type": "string", "description": "The identifier printed on the invoice."},
                "extracted_value": "4471",
                "question": "Does `extracted_value` match the `field` as it appears in `source_text`?"
            }
        }));
        let v = QuestionView::build(&question, &state, &res);
        assert!(v.text.contains("match"));
        let names: Vec<&str> = v.path_refs.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"source_text"));
        assert!(names.contains(&"extracted_value"));
        assert!(names.contains(&"field"));
        assert!(v.has_focus);
        assert_eq!(v.ref_data.iter().filter(|(k, _)| k == "extracted_value").count(), 1);
        assert_eq!(v.criteria.len(), 2);
    }

    #[test]
    fn choice_families_and_literal_options() {
        let res = Resources::empty();
        let state = StateIndex::build(&json!("Where is my package?"), &res);
        let question = q(
            json!({"type": "choice", "instructions": "Which intent does the user's message express?", "criteria": {"track_order": "Wants to know where an order is", "cancel_order": "Wants to cancel an order"}}),
        );
        let v = QuestionView::build(&question, &state, &res);
        assert_eq!(v.family, Family::Intent);
        assert!(!v.literal_options);
        let question = q(
            json!({"type": "choice", "instructions": "Which option is the value?", "criteria": {"Beaver": null, "Dam": null}}),
        );
        let v = QuestionView::build(&question, &state, &res);
        assert!(v.literal_options);
        assert_eq!(v.family, Family::EnumExtraction);
    }
}
