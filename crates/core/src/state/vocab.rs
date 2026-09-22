//! Term interning. A `Vocab` is built once per request from the state; every
//! question then extends it read-only through a private `VocabExt` so that
//! questions can run in parallel without locks.

use crate::lexicon::Resources;
use crate::text::stem::{is_function_word, is_stopword, stem};
use rustc_hash::FxHashMap;

pub type TermId = u32;

#[derive(Debug, Clone)]
pub struct TermInfo {
    /// Stemmed form.
    pub stem: Box<str>,
    /// One surface form (lowercase) that produced this stem.
    pub surface: Box<str>,
    /// Background inverse document frequency in [0, 1] (1 = very rare / unknown).
    pub idf: f32,
    pub stop: bool,
    pub func: bool,
}

#[derive(Debug, Default)]
pub struct Vocab {
    stems: FxHashMap<Box<str>, TermId>,
    surface_to_term: FxHashMap<Box<str>, TermId>,
    terms: Vec<TermInfo>,
}

impl Vocab {
    pub fn new() -> Self {
        Vocab::default()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Intern a lowercase surface token, returning its stem term id.
    pub fn intern(&mut self, surface: &str, res: &Resources) -> TermId {
        if let Some(&id) = self.surface_to_term.get(surface) {
            return id;
        }
        let st = stem(surface);
        let id = if let Some(&id) = self.stems.get(st.as_str()) {
            id
        } else {
            let id = self.terms.len() as TermId;
            let idf = res.idf(&st, surface);
            self.terms.push(TermInfo {
                stem: st.clone().into_boxed_str(),
                surface: surface.into(),
                idf,
                stop: is_stopword(surface) || is_stopword(&st),
                func: is_function_word(surface),
            });
            self.stems.insert(st.into_boxed_str(), id);
            id
        };
        self.surface_to_term.insert(surface.into(), id);
        id
    }

    #[inline]
    pub fn lookup(&self, surface: &str) -> Option<TermId> {
        if let Some(&id) = self.surface_to_term.get(surface) {
            return Some(id);
        }
        let st = stem(surface);
        self.stems.get(st.as_str()).copied()
    }

    #[inline]
    pub fn lookup_stem(&self, st: &str) -> Option<TermId> {
        self.stems.get(st).copied()
    }

    #[inline]
    pub fn info(&self, id: TermId) -> &TermInfo {
        &self.terms[id as usize]
    }

    #[inline]
    pub fn idf(&self, id: TermId) -> f32 {
        self.terms[id as usize].idf
    }
}

/// Read-only view over a shared `Vocab` plus a private extension for terms
/// that only appear in one question. Ids in the extension start at
/// `base.len()`.
/// Cached lexical expansion of one term (synonym and antonym term ids).
#[derive(Debug, Clone, Default)]
pub struct Expansion {
    pub synonyms: smallvec::SmallVec<[TermId; 12]>,
    pub antonyms: smallvec::SmallVec<[TermId; 4]>,
}

pub struct VocabExt<'a> {
    base: &'a Vocab,
    local_stems: FxHashMap<Box<str>, TermId>,
    /// surface → term for surfaces seen in this question (avoids re-stemming).
    local_surface: FxHashMap<Box<str>, TermId>,
    local_terms: Vec<TermInfo>,
    expansions: FxHashMap<TermId, Expansion>,
}

impl<'a> VocabExt<'a> {
    pub fn new(base: &'a Vocab) -> Self {
        VocabExt { base, local_stems: FxHashMap::default(), local_surface: FxHashMap::default(), local_terms: Vec::new(), expansions: FxHashMap::default() }
    }

    pub fn base(&self) -> &'a Vocab {
        self.base
    }

    pub fn intern(&mut self, surface: &str, res: &Resources) -> TermId {
        if let Some(&id) = self.base.surface_to_term.get(surface) {
            return id;
        }
        if let Some(&id) = self.local_surface.get(surface) {
            return id;
        }
        let st = stem(surface);
        let id = if let Some(id) = self.base.lookup_stem(&st) {
            id
        } else if let Some(&id) = self.local_stems.get(st.as_str()) {
            id
        } else {
            let id = (self.base.len() + self.local_terms.len()) as TermId;
            let idf = res.idf(&st, surface);
            self.local_terms.push(TermInfo {
                stem: st.clone().into_boxed_str(),
                surface: surface.into(),
                idf,
                stop: is_stopword(surface) || is_stopword(&st),
                func: is_function_word(surface),
            });
            self.local_stems.insert(st.into_boxed_str(), id);
            id
        };
        self.local_surface.insert(surface.into(), id);
        id
    }

    /// Intern an already-stemmed form (e.g. a lexical-graph synonym stem).
    pub fn intern_stem(&mut self, st: &str, res: &Resources) -> TermId {
        if let Some(id) = self.base.lookup_stem(st) {
            return id;
        }
        if let Some(&id) = self.local_stems.get(st) {
            return id;
        }
        let id = (self.base.len() + self.local_terms.len()) as TermId;
        let idf = res.idf(st, st);
        self.local_terms.push(TermInfo { stem: st.into(), surface: st.into(), idf, stop: is_stopword(st), func: is_function_word(st) });
        self.local_stems.insert(st.into(), id);
        id
    }

    /// Synonym / antonym expansion of a term, computed once per question.
    pub fn expansion(&mut self, id: TermId, res: &Resources) -> Expansion {
        if let Some(e) = self.expansions.get(&id) {
            return e.clone();
        }
        let mut e = Expansion::default();
        if !res.graph.is_empty() {
            let st: Box<str> = self.info(id).stem.clone();
            let syns: smallvec::SmallVec<[&str; 8]> = res.graph.synonym_stems(&st, crate::lexicon::graph::MAX_SENSES);
            for s in syns.iter().take(12) {
                let sid = self.intern_stem(s, res);
                if sid != id && !e.synonyms.contains(&sid) {
                    e.synonyms.push(sid);
                }
            }
            let ants: smallvec::SmallVec<[&str; 4]> = res.graph.antonym_stems(&st);
            for a in ants.iter().take(6) {
                let aid = self.intern_stem(a, res);
                if aid != id && !e.antonyms.contains(&aid) {
                    e.antonyms.push(aid);
                }
            }
        }
        self.expansions.insert(id, e.clone());
        e
    }

    /// True when the term exists in the shared state vocabulary.
    #[inline]
    pub fn in_state(&self, id: TermId) -> bool {
        (id as usize) < self.base.len()
    }

    #[inline]
    pub fn info(&self, id: TermId) -> &TermInfo {
        let n = self.base.len();
        if (id as usize) < n {
            self.base.info(id)
        } else {
            &self.local_terms[id as usize - n]
        }
    }

    #[inline]
    pub fn idf(&self, id: TermId) -> f32 {
        self.info(id).idf
    }

    #[inline]
    pub fn lookup_stem(&self, st: &str) -> Option<TermId> {
        self.base.lookup_stem(st).or_else(|| self.local_stems.get(st).copied())
    }
}
