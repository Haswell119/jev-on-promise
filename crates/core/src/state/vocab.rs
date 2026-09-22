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
pub struct VocabExt<'a> {
    base: &'a Vocab,
    local_stems: FxHashMap<Box<str>, TermId>,
    local_terms: Vec<TermInfo>,
}

impl<'a> VocabExt<'a> {
    pub fn new(base: &'a Vocab) -> Self {
        VocabExt { base, local_stems: FxHashMap::default(), local_terms: Vec::new() }
    }

    pub fn base(&self) -> &'a Vocab {
        self.base
    }

    pub fn intern(&mut self, surface: &str, res: &Resources) -> TermId {
        if let Some(id) = self.base.lookup(surface) {
            return id;
        }
        let st = stem(surface);
        if let Some(&id) = self.local_stems.get(st.as_str()) {
            return id;
        }
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
