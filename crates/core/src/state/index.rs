//! The shared per-request `StateIndex`: built exactly once from the state,
//! then read concurrently by every question. Holds tokens with scope flags,
//! segments (sentence-level fragments) with term frequencies and char
//! n-grams, BM25 postings, phrase bigrams, numbers, dates, path indexes and
//! sentiment valence.

use super::flatten::{flatten_state_with_arrays, split_key_words, ArrayInfo, FieldKind, FlatField};
use super::vocab::{TermId, Vocab};
use crate::lexicon::Resources;
use crate::text::dates::{parse_date_token, parse_textual_date, Date};
use crate::lexicon::graph::SynId;
use crate::text::negation::{self, Flags, CUE, NEGATED};
use crate::text::numbers::{parse_number, NumKind};
use crate::text::segment::segment_ranges;
use crate::text::tokenize::{tokenize, TokenKind};
use crate::text::normalize::normalize_nfkc;
use rustc_hash::{FxHashMap, FxHasher};
use serde_json::Value;
use std::hash::{Hash, Hasher};

pub const ATTR_KEY: u8 = 1;
pub const ATTR_STOP: u8 = 2;
pub const ATTR_FUNC: u8 = 4;
pub const ATTR_CAP: u8 = 8;

#[derive(Debug, Clone)]
pub struct Token {
    pub term: TermId,
    pub kind: TokenKind,
    /// Scope flags from `text::negation`.
    pub flags: Flags,
    pub attrs: u8,
    pub seg: u32,
    /// Byte offsets into the owning field's normalized text.
    pub start: u32,
    pub end: u32,
}

impl Token {
    #[inline]
    pub fn negated(&self) -> bool {
        self.flags & NEGATED != 0
    }
    #[inline]
    pub fn is_cue(&self) -> bool {
        self.flags & CUE != 0
    }
    #[inline]
    pub fn is_content(&self) -> bool {
        self.kind.is_content() && self.attrs & ATTR_STOP == 0 && self.flags & CUE == 0
    }
    /// Content for evidence purposes: not a pure function word / cue / punctuation.
    #[inline]
    pub fn is_evidence(&self) -> bool {
        self.kind.is_content() && self.attrs & ATTR_FUNC == 0 && self.flags & CUE == 0
    }
}

#[derive(Debug, Clone)]
pub struct Field {
    pub path: String,
    pub key: String,
    /// Stems of the path components (all levels), most specific last.
    pub key_terms: Vec<TermId>,
    pub text: String,
    pub kind: FieldKind,
    pub number: Option<f64>,
    pub boolean: Option<bool>,
    pub seg_start: u32,
    pub seg_end: u32,
    pub depth: u8,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub field: u32,
    pub start: u32,
    pub end: u32,
    pub tok_start: u32,
    pub tok_end: u32,
    /// Number of evidence tokens (for BM25 length normalization).
    pub content_len: u32,
    /// Sorted (term, tf) over evidence tokens.
    pub tf: Vec<(TermId, u16)>,
    /// L2 norm of the tf-idf vector.
    pub norm: f32,
    /// Sorted, de-duplicated char 4-gram hashes of the lowercase text.
    pub grams: Vec<u32>,
    /// Sentiment valence in [-1, 1] (0 when the lexicon is empty).
    pub valence: f32,
    /// Magnitude/intensity in [0, 1] from the magnitude lexicon (0.5 when absent).
    pub intensity: f32,
    pub has_intensity: bool,
    /// Union of scope flags over tokens (cheap "does this segment contain negation" test).
    pub flags_any: Flags,
    /// Byte range of this segment's lowercase text inside `StateIndex::lower_text`.
    pub lower_start: u32,
    pub lower_end: u32,
}

/// An array in the state with its key terms and length (for count questions).
#[derive(Debug, Clone)]
pub struct ArrayEntry {
    pub path: String,
    pub key_terms: Vec<TermId>,
    pub len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct NumberSpan {
    pub token: u32,
    pub seg: u32,
    pub value: f64,
    pub kind: NumKind,
    pub currency: Option<&'static str>,
    /// The number came from a number word ("three"), not digits.
    pub from_word: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct DateSpan {
    pub token: u32,
    pub seg: u32,
    pub ntokens: u8,
    pub date: Date,
}

#[derive(Debug)]
pub struct StateIndex {
    pub fields: Vec<Field>,
    pub segments: Vec<Segment>,
    pub tokens: Vec<Token>,
    pub vocab: Vocab,
    /// term → [(segment, tf)] sorted by segment.
    pub postings: FxHashMap<TermId, Vec<(u32, u16)>>,
    pub global_tf: FxHashMap<TermId, u32>,
    /// Hash of consecutive evidence-term pairs → count.
    pub bigrams: FxHashMap<u64, u32>,
    pub numbers: Vec<NumberSpan>,
    pub dates: Vec<DateSpan>,
    /// Exact dotted path → field index.
    pub path_index: FxHashMap<Box<str>, u32>,
    /// Key stem → fields whose last key contains that stem.
    pub key_index: FxHashMap<TermId, Vec<u32>>,
    pub total_content: u32,
    pub avg_seg_len: f32,
    /// Global valence in [-1, 1].
    pub valence: f32,
    /// Global intensity in [0, 1] (mean over segments that carry magnitude words).
    pub intensity: f32,
    pub has_intensity: bool,
    /// Total token count (used for `usage.input_tokens`).
    pub token_count: u64,
    /// Whether the state was a plain string.
    pub is_plain_text: bool,
    /// term → token indexes (all occurrences, evidence tokens only).
    pub term_tokens: FxHashMap<TermId, Vec<u32>>,
    /// Lowercase concatenation of all field texts separated by `\n` (for literal search).
    pub lower_text: String,
    /// (byte offset in `lower_text`, field index) sorted by offset.
    pub field_offsets: Vec<(usize, u32)>,
    /// Sorted, deduped char 4-gram hashes of the whole state.
    pub global_grams: Vec<u32>,
    /// Sum over segments of content_len (== total_content) and the sqrt-norm of the global tf-idf vector.
    pub global_norm: f32,
    /// Arrays in the state (path, key terms, length).
    pub arrays: Vec<ArrayEntry>,
    /// Hypernym ancestors of state evidence terms → weight (idf × depth discount); built once per state.
    pub ancestors: FxHashMap<SynId, f32>,
    /// Topic domains of state evidence terms → weight.
    pub domains: FxHashMap<SynId, f32>,
}

#[inline]
pub fn bigram_hash(a: TermId, b: TermId) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[inline]
pub fn gram_hash(bytes: &[u8]) -> u32 {
    let mut h = FxHasher::default();
    bytes.hash(&mut h);
    h.finish() as u32
}

/// Character 4-grams of a lowercase string (whitespace collapsed), sorted & deduped.
pub fn char_grams(text: &str) -> Vec<u32> {
    let mut buf: Vec<u8> = Vec::with_capacity(text.len() + 2);
    buf.push(b' ');
    let mut last_space = true;
    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            if !last_space {
                buf.push(b' ');
                last_space = true;
            }
        } else {
            let mut tmp = [0u8; 4];
            for c in ch.to_lowercase() {
                buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
            }
            last_space = false;
        }
    }
    if !last_space {
        buf.push(b' ');
    }
    // Byte-level 4-grams over the collapsed buffer (multi-byte chars simply span more grams).
    let n = 4usize;
    if buf.len() < n {
        return Vec::new();
    }
    let mut grams: Vec<u32> = (0..=buf.len() - n).map(|i| gram_hash(&buf[i..i + n])).collect();
    grams.sort_unstable();
    grams.dedup();
    grams
}

/// Size of the intersection of two sorted, deduped gram lists.
pub fn sorted_intersection(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut c) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

const FALSEY_WORDS: &[&str] = &["false", "no", "none", "null", "n/a", "na", "never", "0", "off", "nil", "nope"];

impl StateIndex {
    pub fn build(state: &Value, res: &Resources) -> StateIndex {
        let (flat, arrays): (Vec<FlatField>, Vec<ArrayInfo>) = flatten_state_with_arrays(state);
        let is_plain_text = matches!(state, Value::String(_));
        let mut idx = StateIndex {
            fields: Vec::with_capacity(flat.len()),
            segments: Vec::new(),
            tokens: Vec::new(),
            vocab: Vocab::new(),
            postings: FxHashMap::default(),
            global_tf: FxHashMap::default(),
            bigrams: FxHashMap::default(),
            numbers: Vec::new(),
            dates: Vec::new(),
            path_index: FxHashMap::default(),
            key_index: FxHashMap::default(),
            total_content: 0,
            avg_seg_len: 0.0,
            valence: 0.0,
            intensity: 0.5,
            has_intensity: false,
            token_count: 0,
            is_plain_text,
            term_tokens: FxHashMap::default(),
            lower_text: String::new(),
            field_offsets: Vec::new(),
            global_grams: Vec::new(),
            global_norm: 0.0,
            arrays: Vec::new(),
            ancestors: FxHashMap::default(),
            domains: FxHashMap::default(),
        };
        for f in flat {
            idx.add_field(f, res);
        }
        for a in arrays {
            let key_terms: Vec<TermId> = split_key_words(&a.key).into_iter().map(|w| idx.vocab.intern(&w, res)).collect();
            idx.arrays.push(ArrayEntry { path: a.path, key_terms, len: a.len });
        }
        idx.finish(res);
        idx
    }

    fn add_field(&mut self, f: FlatField, res: &Resources) {
        let field_idx = self.fields.len() as u32;
        let text = normalize_nfkc(&f.text);
        let key_terms: Vec<TermId> = f
            .path
            .split('.')
            .flat_map(|comp| {
                let comp = comp.split('[').next().unwrap_or("");
                split_key_words(comp)
            })
            .map(|w| self.vocab.intern(&w, res))
            .collect();
        let seg_start = self.segments.len() as u32;
        self.field_offsets.push((self.lower_text.len(), field_idx));
        let ranges: Vec<(usize, usize)> = match f.kind {
            FieldKind::Text => segment_ranges(&text),
            _ => vec![(0, text.len())],
        };
        // Falsey scalar values negate the key tokens ("refund_requested": false).
        let lower = text.trim().to_ascii_lowercase();
        let falsey = match f.kind {
            FieldKind::Bool => f.boolean == Some(false),
            FieldKind::Null => true,
            FieldKind::Number => f.number == Some(0.0),
            FieldKind::Text => FALSEY_WORDS.contains(&lower.as_str()),
        };
        let mut key_words: Vec<String> = split_key_words(&f.key);
        if key_words.is_empty() && !f.path.is_empty() {
            key_words = f.path.rsplit('.').next().map(split_key_words).unwrap_or_default();
        }
        if ranges.is_empty() {
            // Empty string value: still emit the key so that the field is visible.
            if !key_words.is_empty() {
                self.push_segment(field_idx, 0, 0, &key_words, "", falsey, res);
            }
        }
        for (i, (s, e)) in ranges.iter().enumerate() {
            let kw: &[String] = if i == 0 { &key_words } else { &[] };
            self.push_segment(field_idx, *s, *e, kw, &text[*s..*e], falsey, res);
        }
        let seg_end = self.segments.len() as u32;
        self.path_index.insert(f.path.clone().into_boxed_str(), field_idx);
        for &kt in &key_terms {
            self.key_index.entry(kt).or_default().push(field_idx);
        }
        self.fields.push(Field {
            path: f.path,
            key: f.key,
            key_terms,
            text,
            kind: f.kind,
            number: f.number,
            boolean: f.boolean,
            seg_start,
            seg_end,
            depth: f.depth,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn push_segment(&mut self, field: u32, start: usize, end: usize, key_words: &[String], text: &str, falsey: bool, res: &Resources) {
        let seg_idx = self.segments.len() as u32;
        let tok_start = self.tokens.len() as u32;
        // Lowercase text for literal search (key words first so `status: shipped` is searchable).
        let lower_start = self.lower_text.len() as u32;
        if !key_words.is_empty() {
            self.lower_text.push_str(&key_words.join(" "));
            self.lower_text.push_str(": ");
        }
        self.lower_text.push_str(&text.to_lowercase());
        let lower_end = self.lower_text.len() as u32;
        self.lower_text.push('\n');
        // Key tokens first (attr KEY), negated when the value is falsey.
        let raw = tokenize(text);
        let flags = negation::annotate(&raw);
        let mut tf: FxHashMap<TermId, u16> = FxHashMap::default();
        let mut content_len = 0u32;
        let mut flags_any: Flags = 0;
        let mut val_sum = 0.0f32;
        let mut val_n = 0u32;
        let mut int_sum = 0.0f32;
        let mut int_n = 0u32;
        for w in key_words {
            let term = self.vocab.intern(w, res);
            let info = self.vocab.info(term);
            let mut attrs = ATTR_KEY;
            if info.stop {
                attrs |= ATTR_STOP;
            }
            if info.func {
                attrs |= ATTR_FUNC;
            }
            let fl = if falsey { NEGATED } else { 0 };
            flags_any |= fl;
            let tok = Token { term, kind: TokenKind::Word, flags: fl, attrs, seg: seg_idx, start: 0, end: 0 };
            if tok.is_evidence() {
                *tf.entry(term).or_insert(0) += 1;
                content_len += 1;
                self.term_tokens.entry(term).or_default().push(self.tokens.len() as u32);
            }
            self.tokens.push(tok);
        }
        let key_count = self.tokens.len() as u32 - tok_start;
        let mut prev_evidence: Option<TermId> = None;
        let mut prev_evidence_pos: usize = 0;
        for (i, rt) in raw.iter().enumerate() {
            let term = self.vocab.intern(&rt.text, res);
            let info = self.vocab.info(term);
            let mut attrs = 0u8;
            if info.stop {
                attrs |= ATTR_STOP;
            }
            if info.func {
                attrs |= ATTR_FUNC;
            }
            if rt.capitalized {
                attrs |= ATTR_CAP;
            }
            let tok = Token {
                term,
                kind: rt.kind,
                flags: flags[i],
                attrs,
                seg: seg_idx,
                start: (start + rt.start) as u32,
                end: (start + rt.end) as u32,
            };
            flags_any |= flags[i];
            if tok.is_evidence() {
                *tf.entry(term).or_insert(0) += 1;
                content_len += 1;
                self.term_tokens.entry(term).or_default().push(self.tokens.len() as u32);
                if let Some(p) = prev_evidence {
                    if i - prev_evidence_pos <= 3 {
                        *self.bigrams.entry(bigram_hash(p, term)).or_insert(0) += 1;
                    }
                }
                prev_evidence = Some(term);
                prev_evidence_pos = i;
            }
            // Magnitude words ("severe", "minor", "all", "none") drive ordinal intensity.
            if rt.kind == TokenKind::Word {
                if let Some(mut m) = crate::question::criteria::magnitude(&rt.text) {
                    if flags[i] & NEGATED != 0 {
                        m = 1.0 - m;
                    }
                    int_sum += m;
                    int_n += 1;
                }
                if rt.text == "!" {
                    int_sum += 0.8;
                    int_n += 1;
                }
            }
            // Sentiment valence with negation flip and intensity scaling.
            if rt.kind == TokenKind::Word && !res.sentiment.is_empty() {
                let v = res.sentiment.valence(&rt.text, &info.stem);
                if v != 0.0 {
                    if v.abs() >= 0.6 && crate::question::criteria::magnitude(&rt.text).is_none() {
                        int_sum += 0.6 + 0.4 * (v.abs() - 0.6) / 0.4;
                        int_n += 1;
                    }
                    let mut v = v;
                    if flags[i] & NEGATED != 0 {
                        v = -0.74 * v;
                    }
                    if flags[i] & negation::INTENSIFIED != 0 {
                        v *= 1.3;
                    }
                    if flags[i] & negation::DIMINISHED != 0 {
                        v *= 0.6;
                    }
                    val_sum += v;
                    val_n += 1;
                }
            }
            let tok_idx = self.tokens.len() as u32;
            match rt.kind {
                TokenKind::Number | TokenKind::Currency | TokenKind::Percent => {
                    if let Some(p) = parse_number(&rt.text, rt.kind) {
                        self.numbers.push(NumberSpan { token: tok_idx, seg: seg_idx, value: p.value, kind: p.kind, currency: p.currency, from_word: false });
                    }
                }
                TokenKind::Word => {
                    if let Some(p) = parse_number(&rt.text, rt.kind) {
                        // number words: only when not part of an idiom like "one of"
                        self.numbers.push(NumberSpan { token: tok_idx, seg: seg_idx, value: p.value, kind: NumKind::Plain, currency: None, from_word: true });
                    }
                }
                TokenKind::Date => {
                    if let Some(d) = parse_date_token(&rt.text) {
                        self.dates.push(DateSpan { token: tok_idx, seg: seg_idx, ntokens: 1, date: d });
                    }
                }
                _ => {}
            }
            self.tokens.push(tok);
        }
        // Textual dates ("March 3, 2026", "3 September") over the raw word sequence.
        let words: Vec<&str> = raw.iter().map(|t| t.text.as_str()).collect();
        let mut i = 0usize;
        while i < words.len() {
            if crate::text::dates::month_from_name(words[i]).is_some() || (words[i].chars().all(|c| c.is_ascii_digit()) && i + 1 < words.len()) {
                if let Some((d, n)) = parse_textual_date(&words, i, 2000) {
                    // Only accept forms that contain a month name (pure numbers handled above).
                    let has_month = words[i..i + n].iter().any(|w| crate::text::dates::month_from_name(w).is_some());
                    if has_month {
                        self.dates.push(DateSpan { token: tok_start + key_count + i as u32, seg: seg_idx, ntokens: n as u8, date: d });
                        i += n;
                        continue;
                    }
                }
            }
            i += 1;
        }
        let mut tf_vec: Vec<(TermId, u16)> = tf.into_iter().collect();
        tf_vec.sort_unstable_by_key(|(t, _)| *t);
        let mut norm = 0.0f32;
        for &(t, c) in &tf_vec {
            let w = (1.0 + (c as f32).ln()) * self.vocab.idf(t);
            norm += w * w;
            self.postings.entry(t).or_default().push((seg_idx, c));
            *self.global_tf.entry(t).or_insert(0) += c as u32;
        }
        let valence = if val_n > 0 { (val_sum / (val_n as f32).sqrt()).clamp(-1.0, 1.0) } else { 0.0 };
        let intensity = if int_n > 0 { (int_sum / int_n as f32).clamp(0.0, 1.0) } else { 0.5 };
        self.total_content += content_len;
        self.token_count += raw.len() as u64;
        self.segments.push(Segment {
            field,
            start: start as u32,
            end: end as u32,
            tok_start,
            tok_end: self.tokens.len() as u32,
            content_len,
            tf: tf_vec,
            norm: norm.sqrt(),
            grams: char_grams(text),
            valence,
            intensity,
            has_intensity: int_n > 0,
            flags_any,
            lower_start,
            lower_end,
        });
    }

    fn finish(&mut self, _res: &Resources) {
        let mut grams: Vec<u32> = self.segments.iter().flat_map(|s| s.grams.iter().copied()).collect();
        grams.sort_unstable();
        grams.dedup();
        self.global_grams = grams;
        let mut norm = 0.0f32;
        for (&t, &c) in &self.global_tf {
            let w = (1.0 + (c as f32).ln()) * self.vocab.idf(t);
            norm += w * w;
        }
        self.global_norm = norm.sqrt();
        let n = self.segments.len().max(1) as f32;
        self.avg_seg_len = (self.total_content as f32 / n).max(1.0);
        let (mut vs, mut vn) = (0.0f32, 0u32);
        for s in &self.segments {
            if s.valence != 0.0 {
                vs += s.valence;
                vn += 1;
            }
        }
        self.valence = if vn > 0 { (vs / (vn as f32).sqrt()).clamp(-1.0, 1.0) } else { 0.0 };
        let (mut is, mut inn) = (0.0f32, 0u32);
        for s in &self.segments {
            if s.has_intensity {
                is += s.intensity;
                inn += 1;
            }
        }
        self.has_intensity = inn > 0;
        self.intensity = if inn > 0 { is / inn as f32 } else { 0.5 };
        // Lexical-graph maps: hypernym ancestors and topic domains of the state's evidence terms.
        if !_res.graph.is_empty() {
            let g = &_res.graph;
            let mut terms: Vec<(TermId, u32)> = self.global_tf.iter().map(|(t, c)| (*t, *c)).collect();
            terms.sort_unstable();
            for (t, _c) in terms {
                let info = self.vocab.info(t);
                if info.stop || info.func || info.idf < 0.25 || info.stem.len() < 3 {
                    continue;
                }
                let w = info.idf;
                for sense in g.senses_of_stem(&info.stem, 2) {
                    let e = self.ancestors.entry(sense).or_insert(0.0);
                    *e = e.max(w);
                    for (d, anc) in g.hypernym_closure_with_depth(sense, 3) {
                        if g.root_depth(anc) < 4 {
                            continue;
                        }
                        let v = w * 0.75f32.powi(d as i32);
                        let e = self.ancestors.entry(anc).or_insert(0.0);
                        *e = e.max(v);
                    }
                    for &dom in g.domains(sense) {
                        let e = self.domains.entry(dom).or_insert(0.0);
                        *e = e.max(w);
                    }
                }
            }
        }
    }

    #[inline]
    pub fn segment_text(&self, seg: u32) -> &str {
        let s = &self.segments[seg as usize];
        let f = &self.fields[s.field as usize];
        &f.text[s.start as usize..s.end as usize]
    }

    #[inline]
    pub fn segment_tokens(&self, seg: u32) -> &[Token] {
        let s = &self.segments[seg as usize];
        &self.tokens[s.tok_start as usize..s.tok_end as usize]
    }

    #[inline]
    pub fn segment_path(&self, seg: u32) -> &str {
        &self.fields[self.segments[seg as usize].field as usize].path
    }

    /// Does the state contain the term anywhere (as evidence)?
    #[inline]
    pub fn contains_term(&self, t: TermId) -> bool {
        self.global_tf.contains_key(&t)
    }

    /// Segment that owns byte offset `pos` of `lower_text`.
    pub fn segment_at_lower_offset(&self, pos: usize) -> Option<u32> {
        let pos = pos as u32;
        match self.segments.binary_search_by(|s| s.lower_start.cmp(&pos)) {
            Ok(i) => Some(i as u32),
            Err(0) => None,
            Err(i) => {
                let s = &self.segments[i - 1];
                (pos < s.lower_end).then_some((i - 1) as u32)
            }
        }
    }

    /// Field that owns byte offset `pos` of `lower_text`.
    pub fn field_at_offset(&self, pos: usize) -> u32 {
        match self.field_offsets.binary_search_by(|(o, _)| o.cmp(&pos)) {
            Ok(i) => self.field_offsets[i].1,
            Err(0) => 0,
            Err(i) => self.field_offsets[i - 1].1,
        }
    }

    /// Fields whose path equals `path` or ends with `.path` (case-insensitive).
    pub fn fields_for_path(&self, path: &str) -> Vec<u32> {
        let p = path.trim().trim_start_matches('$').trim_start_matches('.');
        if p.is_empty() {
            return Vec::new();
        }
        if let Some(&f) = self.path_index.get(p) {
            return vec![f];
        }
        let lower = p.to_ascii_lowercase();
        let mut out = Vec::new();
        for (i, f) in self.fields.iter().enumerate() {
            let fp = f.path.to_ascii_lowercase();
            if fp == lower || fp.ends_with(&format!(".{lower}")) || fp.starts_with(&format!("{lower}.")) || fp.starts_with(&format!("{lower}[")) || fp.contains(&format!(".{lower}[")) || fp.contains(&format!(".{lower}.")) {
                out.push(i as u32);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builds_index_from_json() {
        let res = Resources::empty();
        let idx = StateIndex::build(&json!({"ticket": {"text": "I did not get a refund. Please help!", "refund_requested": false, "amount": 42}}), &res);
        assert_eq!(idx.fields.len(), 3);
        assert!(idx.segments.len() >= 3);
        // key tokens for refund_requested are negated because the value is false
        let f = idx.fields.iter().position(|f| f.path == "ticket.refund_requested").unwrap();
        let seg = idx.fields[f].seg_start;
        let toks = idx.segment_tokens(seg);
        assert!(toks.iter().filter(|t| t.attrs & ATTR_KEY != 0).all(|t| t.negated()));
        // number captured
        assert!(idx.numbers.iter().any(|n| n.value == 42.0));
        let refund = idx.vocab.lookup("refund").unwrap();
        assert!(idx.contains_term(refund));
        assert!(!idx.fields_for_path("ticket.text").is_empty());
        assert!(!idx.fields_for_path("text").is_empty());
    }

    #[test]
    fn plain_text_state_and_dates() {
        let res = Resources::empty();
        let idx = StateIndex::build(&json!("Order shipped on 3 September 2025. Invoice 2025-09-01 for $30."), &res);
        assert!(idx.is_plain_text);
        assert!(idx.dates.len() >= 2);
        assert!(idx.numbers.iter().any(|n| n.kind == NumKind::Currency && n.value == 30.0));
        assert!(idx.avg_seg_len > 0.0);
    }

    #[test]
    fn char_gram_similarity() {
        let a = char_grams("refund request");
        let b = char_grams("refund requested");
        let inter = sorted_intersection(&a, &b);
        assert!(inter as f32 / a.len() as f32 > 0.8);
    }
}
