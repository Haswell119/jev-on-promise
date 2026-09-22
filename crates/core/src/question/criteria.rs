//! Criterion flattening. A criterion (Choice option, Score level or Noul
//! hypothesis) may be a string, an array or a nested object. Objects are
//! flattened while preserving field-name semantics: fields whose names
//! carry negation/exclusion morphemes contribute NEGATIVE evidence, fields
//! that look like example lists become separately-scored phrases, other
//! fields contribute positive evidence. Detection is generic (morpheme
//! based) rather than a fixed list of names.

use super::numeric::{parse_range, NumericRange};
use crate::lexicon::Resources;
use crate::state::flatten::split_key_words;
use crate::state::index::{bigram_hash, char_grams};
use crate::state::vocab::{TermId, VocabExt};
use crate::text::negation::{self, Flags, CUE, HYPOTHETICAL, NEGATED, REQUEST};
use crate::text::normalize::normalize_nfkc;
use crate::text::tokenize::{tokenize, TokenKind};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct QueryTerm {
    pub term: TermId,
    /// idf × source factor (name/text/example).
    pub weight: f32,
    pub negated: bool,
    pub hypothetical: bool,
    pub request: bool,
    /// +1 positive evidence, -1 negative evidence (exclusion fields).
    pub polarity: i8,
    pub is_name: bool,
    pub idf: f32,
}

#[derive(Debug, Clone)]
pub struct Phrase {
    pub text: String,
    pub terms: Vec<TermId>,
    pub bigrams: Vec<u64>,
    pub grams: Vec<u32>,
    pub polarity: i8,
    pub is_example: bool,
}

#[derive(Debug, Clone)]
pub struct Criterion {
    pub key: String,
    pub index: usize,
    pub name_terms: Vec<TermId>,
    /// Deduplicated weighted terms (positive and negative).
    pub terms: Vec<QueryTerm>,
    pub phrases: Vec<Phrase>,
    pub bigrams: Vec<u64>,
    pub grams: Vec<u32>,
    /// Normalized literal strings that could appear verbatim in the state.
    pub literals: Vec<String>,
    pub range: Option<NumericRange>,
    pub valence: f32,
    pub intensity: f32,
    pub has_intensity: bool,
    pub n_pos_terms: usize,
    pub n_neg_terms: usize,
    pub full_text: String,
    pub is_null: bool,
    /// Sum of positive term weights (for coverage normalization).
    pub pos_weight: f32,
    pub neg_weight: f32,
    /// Terms after synonym expansion (stem strings) → weight factor.
    pub expanded: Vec<(TermId, f32, TermId)>,
    /// Antonym stems of positive terms (term id of antonym, weight, source term).
    pub antonyms: Vec<(TermId, f32)>,
    /// Share of positive content tokens under a negation scope (description polarity profile).
    pub neg_share: f32,
    /// Share of positive content tokens under a hypothetical/uncertain scope.
    pub hyp_share: f32,
}

const NEGATIVE_KEY_MORPHEMES: &[&str] = &[
    "not",
    "no",
    "non",
    "exclud",
    "except",
    "never",
    "avoid",
    "negative",
    "anti",
    "unless",
    "without",
    "dont",
    "isnt",
    "wrong",
    "bad",
    "counter",
    "unlike",
    "differ",
    "contrast",
    "mis",
    "oos",
    "out_of_scope",
    "false",
    "incorrect",
    "distractor",
    "reject",
    "deny",
    "forbid",
    "prohibit",
    "disallow",
    "unrelated",
    "irrelevant",
    "doesnt",
    "does_not",
    "is_not",
    "no_match",
    "nope",
    "neg",
    "opposite",
    "disqualif",
    "ineligible",
    "invalid",
    "exception",
];

const EXAMPLE_KEY_MORPHEMES: &[&str] = &[
    "example",
    "sample",
    "e_g",
    "eg",
    "instance",
    "such_as",
    "phrase",
    "utterance",
    "keyword",
    "synonym",
    "trigger",
    "signal",
    "cue",
    "indicator",
    "pattern",
    "typical",
    "like",
    "e.g",
    "alias",
    "variant",
    "wording",
];

/// Field names that are pure labels (their words are not evidence).
const LABEL_KEYS: &[&str] = &[
    "what",
    "description",
    "desc",
    "definition",
    "scope",
    "includes",
    "include",
    "covers",
    "cover",
    "means",
    "meaning",
    "summary",
    "details",
    "detail",
    "notes",
    "note",
    "when",
    "use_when",
    "criteria",
    "criterion",
    "rubric",
    "text",
    "value",
    "name",
    "label",
    "title",
    "info",
    "instructions",
    "instruction",
    "question",
    "focus",
    "explanation",
    "explain",
    "rationale",
    "guidance",
    "hint",
    "hints",
    "context",
    "content",
    "body",
    "message",
    "level",
    "option",
    "answer",
    "type",
    "id",
    "key",
    "positive",
    "yes",
    "true",
    "also",
    "and",
    "or",
    "for",
    "the",
    "signals",
    "indicators",
    "examples",
    "example",
    "keywords",
    "phrases",
    "sample",
    "samples",
    "definitions",
];

pub fn is_negative_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    let words = split_key_words(&k);
    let joined = words.join("_");
    NEGATIVE_KEY_MORPHEMES.iter().any(|m| if m.len() <= 3 { words.iter().any(|w| w == m) } else { joined.contains(m) })
}

pub fn is_example_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    let words = split_key_words(&k);
    let joined = words.join("_");
    EXAMPLE_KEY_MORPHEMES.iter().any(|m| if m.len() <= 3 { words.iter().any(|w| w == m) } else { joined.contains(m) })
}

fn is_label_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    LABEL_KEYS.contains(&k.as_str()) || split_key_words(&k).iter().all(|w| LABEL_KEYS.contains(&w.as_str()))
}

#[derive(Default)]
struct Collector {
    /// (text, polarity, weight factor)
    texts: Vec<(String, i8, f32)>,
    /// (text, polarity, is_example)
    phrases: Vec<(String, i8, bool)>,
}

fn collect(v: &Value, polarity: i8, example: bool, in_array: bool, depth: usize, out: &mut Collector) {
    if depth > 12 {
        return;
    }
    match v {
        Value::Null => {}
        Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                return;
            }
            out.texts.push((s.to_string(), polarity, if example { 0.8 } else { 1.0 }));
            if example || in_array {
                out.phrases.push((s.to_string(), polarity, example));
            }
        }
        Value::Number(n) => out.texts.push((n.to_string(), polarity, 0.8)),
        Value::Bool(b) => out.texts.push((if *b { "true" } else { "false" }.to_string(), polarity, 0.5)),
        Value::Array(items) => {
            for it in items {
                collect(it, polarity, example, true, depth + 1, out);
            }
        }
        Value::Object(map) => {
            for (k, child) in map {
                let kp = if is_negative_key(k) { -polarity } else { polarity };
                let ex = example || is_example_key(k);
                match child {
                    Value::Bool(b) => {
                        // {"refund": true} → "refund"; {"refund": false} → negated "refund"
                        let words = split_key_words(k).join(" ");
                        if !words.is_empty() {
                            let text = if *b { words } else { format!("not {words}") };
                            out.texts.push((text, kp, 0.9));
                        }
                    }
                    Value::Null => {
                        let words = split_key_words(k).join(" ");
                        if !words.is_empty() && !is_label_key(k) {
                            out.texts.push((words, kp, 0.7));
                        }
                    }
                    Value::Number(n) => {
                        let words = split_key_words(k).join(" ");
                        out.texts.push((format!("{words} {n}"), kp, 0.8));
                    }
                    _ => {
                        if !is_label_key(k) && !is_negative_key(k) && !is_example_key(k) {
                            let words = split_key_words(k).join(" ");
                            if !words.is_empty() {
                                out.texts.push((words, kp, 0.6));
                            }
                        }
                        collect(child, kp, ex, false, depth + 1, out);
                    }
                }
            }
        }
    }
}

/// Words that describe magnitude; used for ordinal intensity matching.
pub fn magnitude(word: &str) -> Option<f32> {
    Some(match word {
        "none" | "never" | "nobody" | "zero" | "nothing" | "worst" => 0.0,
        "negligible" | "trivial" | "cosmetic" | "minimal" | "minimally" | "tiny" | "barely" | "hardly" => 0.08,
        "slight" | "slightly" | "minor" | "mild" | "mildly" | "low" | "rarely" | "few" | "little" | "small"
        | "poor" | "weak" | "weakly" | "bad" | "single" | "nonessential" | "optional" | "workaround" | "isolated" => {
            0.2
        }
        "some" | "somewhat" | "partial" | "partially" | "occasionally" | "sometimes" | "several" | "limited"
        | "fair" | "fairly" | "modest" | "okay" | "ok" | "fine" | "average" | "neutral" | "mixed" | "moderate"
        | "moderately" | "medium" | "moderate-ly" => 0.5,
        "often" | "usually" | "mostly" | "many" | "most" | "notable" | "notably" | "considerable" | "considerably"
        | "significant" | "significantly" | "good" | "strong" | "strongly" | "high" | "major" | "serious"
        | "seriously" | "large" | "substantial" | "substantially" | "important" | "elevated" | "numerous"
        | "multiple" | "dozens" | "hundreds" | "thousands" | "millions" | "widely" | "core" | "essential"
        | "critical-path" => 0.75,
        "very" | "highly" | "great" | "greatly" | "severe" | "severely" | "critical" | "critically" | "urgent"
        | "urgently" | "asap" | "immediately" | "heavy" | "heavily" | "intense" | "intensely" | "extensive"
        | "extensively" | "deeply" | "widespread" | "blocked" | "broken" | "excellent" | "outstanding" | "huge"
        | "massive" | "massively" | "extreme" | "extremely" => 0.9,
        "always" | "all" | "every" | "everyone" | "entire" | "entirely" | "complete" | "completely" | "total"
        | "totally" | "fully" | "full" | "absolute" | "absolutely" | "perfect" | "perfectly" | "best"
        | "catastrophic" | "fatal" | "irreversible" | "emergency" | "unbearable" | "furious" | "outraged"
        | "devastating" | "devastated" | "utterly" | "maximum" | "maximal" | "permanent" | "permanently"
        | "unrecoverable" | "wiped" | "recalled" | "burned" | "injured" | "injury" | "harm" | "death" | "deleted"
        | "destroyed" | "lost" | "loss" => 1.0,
        _ => return None,
    })
}

fn is_quantity_word(w: &str) -> bool {
    matches!(w, "many" | "much" | "several" | "few" | "some" | "all" | "most" | "every" | "none" | "any")
}

/// Compute valence and intensity for a token sequence (with scope flags).
pub fn valence_and_intensity(
    toks: &[crate::text::tokenize::RawToken],
    flags: &[Flags],
    res: &Resources,
) -> (f32, f32, bool) {
    valence_and_intensity_with(toks, flags, res, |i| crate::text::stem::stem(&toks[i].text))
}

/// Same as `valence_and_intensity` but with a caller-provided stem lookup
/// (avoids re-stemming when the tokens are already interned).
pub fn valence_and_intensity_with(
    toks: &[crate::text::tokenize::RawToken],
    flags: &[Flags],
    res: &Resources,
    stem_of: impl Fn(usize) -> String,
) -> (f32, f32, bool) {
    let (mut vs, mut vn) = (0.0f32, 0u32);
    let (mut is, mut inn) = (0.0f32, 0u32);
    for (i, t) in toks.iter().enumerate() {
        if t.kind != TokenKind::Word {
            if t.kind == TokenKind::Punct && t.text == "!" {
                is += 0.8;
                inn += 1;
            }
            continue;
        }
        let f = flags[i];
        let st = if res.sentiment.is_empty() { String::new() } else { stem_of(i) };
        let mut v = res.sentiment.valence(&t.text, &st);
        if v != 0.0 {
            if f & NEGATED != 0 {
                v *= -0.74;
            }
            if f & negation::INTENSIFIED != 0 {
                v *= 1.3;
            }
            if f & negation::DIMINISHED != 0 {
                v *= 0.6;
            }
            vs += v;
            vn += 1;
        }
        if let Some(mut m) = magnitude(&t.text) {
            if f & NEGATED != 0 && !is_quantity_word(&t.text) {
                m = 1.0 - m;
            }
            if f & NEGATED != 0 && is_quantity_word(&t.text) {
                m = (1.0 - m).min(0.3);
            }
            is += m;
            inn += 1;
        }
        // Strong sentiment words add intensity too.
        if v.abs() >= 0.6 && magnitude(&t.text).is_none() {
            is += 0.6 + 0.4 * (v.abs() - 0.6) / 0.4;
            inn += 1;
        }
    }
    let valence = if vn > 0 { (vs / (vn as f32).sqrt()).clamp(-1.0, 1.0) } else { 0.0 };
    let intensity = if inn > 0 { (is / inn as f32).clamp(0.0, 1.0) } else { 0.5 };
    (valence, intensity, inn > 0)
}

/// Normalize a literal for verbatim matching: lowercase, NFKC, collapse
/// whitespace, drop surrounding quotes/punctuation.
pub fn normalize_literal(s: &str) -> String {
    let n = normalize_nfkc(s).to_lowercase();
    let n = n.trim().trim_matches(|c: char| c == '"' || c == '\'' || c == '.' || c == ',' || c == ';');
    let mut out = String::with_capacity(n.len());
    let mut last_space = false;
    for ch in n.chars() {
        let c = if ch == '_' || ch == '-' { ' ' } else { ch };
        if c.is_whitespace() {
            if !last_space && !out.is_empty() {
                out.push(' ');
            }
            last_space = true;
        } else {
            out.push(c);
            last_space = false;
        }
    }
    out.trim().to_string()
}

impl Criterion {
    /// Build a criterion from a key (option name / level index / hypothesis
    /// name) and its description entry.
    pub fn build(
        key: &str,
        index: usize,
        entry: &Value,
        key_is_name: bool,
        vocab: &mut VocabExt<'_>,
        res: &Resources,
    ) -> Criterion {
        let mut col = Collector::default();
        collect(entry, 1, false, false, 0, &mut col);
        let is_null = col.texts.is_empty();

        let mut terms: Vec<QueryTerm> = Vec::new();
        let push_term = |t: QueryTerm, terms: &mut Vec<QueryTerm>| {
            if let Some(existing) =
                terms.iter_mut().find(|e| e.term == t.term && e.polarity == t.polarity && e.negated == t.negated)
            {
                existing.weight = (existing.weight + 0.5 * t.weight).min(existing.idf * 2.0);
                existing.hypothetical |= t.hypothetical;
                existing.request |= t.request;
                existing.is_name |= t.is_name;
            } else {
                terms.push(t);
            }
        };

        let mut name_terms = Vec::new();
        if key_is_name {
            let name_text = split_key_words(key).join(" ");
            let raw = tokenize(&name_text);
            let fl = negation::annotate(&raw);
            for (i, rt) in raw.iter().enumerate() {
                if !rt.kind.is_content() {
                    continue;
                }
                let tid = vocab.intern(&rt.text, res);
                let info = vocab.info(tid);
                if info.func || fl[i] & CUE != 0 {
                    continue;
                }
                name_terms.push(tid);
                let w_ = if info.stop { 0.25 } else { 1.0 };
                push_term(
                    QueryTerm {
                        term: tid,
                        weight: info.idf.max(0.15) * 1.1 * w_,
                        negated: fl[i] & NEGATED != 0,
                        hypothetical: fl[i] & HYPOTHETICAL != 0,
                        request: fl[i] & REQUEST != 0,
                        polarity: 1,
                        is_name: true,
                        idf: info.idf,
                    },
                    &mut terms,
                );
            }
        }

        let mut pos_text = String::new();
        let mut all_text = String::new();
        let mut pos_toks: Vec<crate::text::tokenize::RawToken> = Vec::new();
        let mut pos_flags: Vec<Flags> = Vec::new();
        let mut pos_term_ids: Vec<Option<TermId>> = Vec::new();
        let mut bigrams: Vec<u64> = Vec::new();
        for (text, polarity, factor) in &col.texts {
            let norm = normalize_nfkc(text);
            let raw = tokenize(&norm);
            let fl = negation::annotate(&raw);
            if !all_text.is_empty() {
                all_text.push_str(" | ");
            }
            all_text.push_str(if *polarity < 0 { "NOT: " } else { "" });
            all_text.push_str(&norm);
            if *polarity > 0 {
                if !pos_text.is_empty() {
                    pos_text.push(' ');
                }
                pos_text.push_str(&norm.to_lowercase());
            }
            let mut prev: Option<(TermId, usize)> = None;
            for (i, rt) in raw.iter().enumerate() {
                if !rt.kind.is_content() {
                    continue;
                }
                let tid = vocab.intern(&rt.text, res);
                let info = vocab.info(tid);
                if fl[i] & CUE != 0 || info.func {
                    continue;
                }
                let stop_factor = if info.stop { 0.3 } else { 1.0 };
                let t = QueryTerm {
                    term: tid,
                    weight: info.idf.max(0.1) * factor * stop_factor,
                    negated: fl[i] & NEGATED != 0,
                    hypothetical: fl[i] & HYPOTHETICAL != 0,
                    request: fl[i] & REQUEST != 0,
                    polarity: *polarity,
                    is_name: false,
                    idf: info.idf,
                };
                push_term(t, &mut terms);
                if *polarity > 0 && !info.stop {
                    if let Some((p, pi)) = prev {
                        if i - pi <= 3 {
                            bigrams.push(bigram_hash(p, tid));
                        }
                    }
                    prev = Some((tid, i));
                }
            }
            if *polarity > 0 {
                for rt in raw.iter() {
                    pos_term_ids.push(if rt.kind == TokenKind::Word {
                        Some(vocab.intern(&rt.text, res))
                    } else {
                        None
                    });
                }
                pos_toks.extend(raw.iter().cloned());
                pos_flags.extend(fl.iter().copied());
            }
        }
        bigrams.sort_unstable();
        bigrams.dedup();

        let phrases: Vec<Phrase> = col
            .phrases
            .iter()
            .map(|(text, polarity, is_example)| {
                let norm = normalize_nfkc(text).to_lowercase();
                let raw = tokenize(&norm);
                let mut pterms = Vec::new();
                let mut pbig = Vec::new();
                let mut prev: Option<(TermId, usize)> = None;
                for (i, rt) in raw.iter().enumerate() {
                    if !rt.kind.is_content() {
                        continue;
                    }
                    let tid = vocab.intern(&rt.text, res);
                    let info = vocab.info(tid);
                    if info.func || info.stop {
                        continue;
                    }
                    pterms.push(tid);
                    if let Some((p, pi)) = prev {
                        if i - pi <= 3 {
                            pbig.push(bigram_hash(p, tid));
                        }
                    }
                    prev = Some((tid, i));
                }
                Phrase {
                    text: norm.clone(),
                    terms: pterms,
                    bigrams: pbig,
                    grams: char_grams(&norm),
                    polarity: *polarity,
                    is_example: *is_example,
                }
            })
            .collect();

        // Literals: the key (for names) and short positive descriptions.
        let mut literals: Vec<String> = Vec::new();
        if key_is_name {
            let lit = normalize_literal(key);
            if !lit.is_empty() {
                literals.push(lit);
            }
        }
        for (text, polarity, _) in &col.texts {
            if *polarity > 0 {
                let words = text.split_whitespace().count();
                if (1..=6).contains(&words) {
                    let lit = normalize_literal(text);
                    if !lit.is_empty() && !literals.contains(&lit) {
                        literals.push(lit);
                    }
                }
            }
        }

        let (valence, intensity, has_intensity) = {
            let v = &*vocab;
            valence_and_intensity_with(&pos_toks, &pos_flags, res, |i| {
                pos_term_ids[i].map(|t| v.info(t).stem.to_string()).unwrap_or_default()
            })
        };
        let (mut n_content, mut n_neg, mut n_hyp) = (0u32, 0u32, 0u32);
        for (i, t) in pos_toks.iter().enumerate() {
            if !t.kind.is_content() || crate::text::stem::is_function_word(&t.text) || pos_flags[i] & CUE != 0 {
                continue;
            }
            n_content += 1;
            if pos_flags[i] & NEGATED != 0 {
                n_neg += 1;
            }
            if pos_flags[i] & HYPOTHETICAL != 0 {
                n_hyp += 1;
            }
        }
        let neg_share = if n_content > 0 { n_neg as f32 / n_content as f32 } else { 0.0 };
        let hyp_share = if n_content > 0 { n_hyp as f32 / n_content as f32 } else { 0.0 };
        let has_number = |t: &str| {
            t.chars().any(|c| c.is_ascii_digit())
                || t.split(|c: char| !c.is_alphabetic()).any(|w| crate::text::numbers::number_word(w).is_some())
        };
        let key_lit = normalize_literal(key);
        let range = if is_null {
            if has_number(&key_lit) {
                parse_range(&key_lit)
            } else {
                None
            }
        } else if has_number(&pos_text) {
            parse_range(&pos_text).or_else(|| if has_number(&key_lit) { parse_range(&key_lit) } else { None })
        } else if has_number(&key_lit) {
            parse_range(&key_lit)
        } else {
            None
        };

        let n_pos_terms = terms.iter().filter(|t| t.polarity > 0).count();
        let n_neg_terms = terms.iter().filter(|t| t.polarity < 0).count();
        let pos_weight: f32 = terms.iter().filter(|t| t.polarity > 0).map(|t| t.weight).sum();
        let neg_weight: f32 = terms.iter().filter(|t| t.polarity < 0).map(|t| t.weight).sum();

        // Lexical expansion (synonyms / antonyms) for positive, non-stop terms (cached per question).
        let mut expanded: Vec<(TermId, f32, TermId)> = Vec::new();
        let mut antonyms: Vec<(TermId, f32)> = Vec::new();
        if !res.graph.is_empty() {
            let pos_terms: Vec<(TermId, f32)> = terms
                .iter()
                .filter(|t| t.polarity > 0 && !t.negated && !vocab.info(t.term).stop)
                .map(|t| (t.term, t.weight))
                .collect();
            for (tid, w) in pos_terms {
                let e = vocab.expansion(tid, res);
                for &sid in &e.synonyms {
                    if !terms.iter().any(|t| t.term == sid) {
                        expanded.push((sid, w * 0.6, tid));
                    }
                }
                for &aid in &e.antonyms {
                    if !terms.iter().any(|t| t.term == aid) {
                        antonyms.push((aid, w));
                    }
                }
            }
        }

        Criterion {
            key: key.to_string(),
            index,
            name_terms,
            terms,
            phrases,
            bigrams,
            grams: char_grams(&pos_text),
            literals,
            range,
            valence,
            intensity,
            has_intensity,
            n_pos_terms,
            n_neg_terms,
            full_text: all_text,
            is_null,
            pos_weight,
            neg_weight,
            expanded,
            antonyms,
            neg_share,
            hyp_share,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::vocab::Vocab;
    use serde_json::json;

    #[test]
    fn negative_and_example_keys() {
        assert!(is_negative_key("not_for"));
        assert!(is_negative_key("excludes"));
        assert!(is_negative_key("negative_examples"));
        assert!(!is_negative_key("notes"));
        assert!(!is_negative_key("what"));
        assert!(is_example_key("examples"));
        assert!(is_example_key("negative_examples"));
    }

    #[test]
    fn flattens_structured_option() {
        let res = Resources::empty();
        let base = Vocab::new();
        let mut v = VocabExt::new(&base);
        let entry = json!({"what": "Charges, invoices, refunds", "not_for": "Order tracking", "examples": ["I was charged twice", "Where is my refund?"]});
        let c = Criterion::build("billing", 0, &entry, true, &mut v, &res);
        assert!(c.name_terms.len() == 1);
        assert!(c.terms.iter().any(|t| t.polarity < 0 && v.info(t.term).stem.as_ref() == "track"));
        assert!(c.terms.iter().any(|t| t.polarity > 0 && v.info(t.term).stem.as_ref() == "refund"));
        assert_eq!(c.phrases.iter().filter(|p| p.is_example).count(), 2);
        assert!(c.literals.contains(&"billing".to_string()));
    }

    #[test]
    fn negated_description_terms() {
        let res = Resources::empty();
        let base = Vocab::new();
        let mut v = VocabExt::new(&base);
        let c = Criterion::build("no_refund", 0, &json!("The customer did not request a refund"), true, &mut v, &res);
        let refund = c.terms.iter().find(|t| v.info(t.term).stem.as_ref() == "refund").unwrap();
        assert!(refund.negated);
    }
}
