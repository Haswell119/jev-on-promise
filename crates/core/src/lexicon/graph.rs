//! Compact WordNet-style lexical graph: lemma → synsets (sense order),
//! synset → lemmas / hypernyms / similar-to, lemma → antonyms.
//!
//! File formats (all lowercase, tab separated):
//! * `wn_lemmas.tsv`:  `lemma<TAB>synset_id[,synset_id...]` (sense order)
//! * `wn_synsets.tsv`: `synset_id<TAB>lemma|lemma...<TAB>hypernym_id,...<TAB>similar_id,...`
//! * `wn_antonyms.tsv`: `lemma<TAB>antonym_lemma`
//!
//! Synset ids are arbitrary strings (e.g. `n02084071`); they are re-indexed
//! to dense u32 ids on load. Multi-word lemmas use underscores.

use crate::text::stem::stem;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

pub type SynId = u32;
pub type LemmaId = u32;

#[derive(Debug, Default)]
pub struct LexGraph {
    lemma_ids: FxHashMap<Box<str>, LemmaId>,
    lemma_names: Vec<Box<str>>,
    lemma_stem: Vec<Box<str>>,
    /// lemma id → synsets in sense order
    lemma_synsets: Vec<SmallVec<[SynId; 4]>>,
    /// stem → lemma ids (single-word lemmas only)
    stem_lemmas: FxHashMap<Box<str>, SmallVec<[LemmaId; 2]>>,
    synset_lemmas: Vec<SmallVec<[LemmaId; 4]>>,
    synset_hypernyms: Vec<SmallVec<[SynId; 2]>>,
    synset_similar: Vec<SmallVec<[SynId; 2]>>,
    /// Topic-domain synsets (WordNet `;c` pointers), e.g. "computer science".
    synset_domains: Vec<SmallVec<[SynId; 1]>>,
    antonyms: FxHashMap<LemmaId, SmallVec<[LemmaId; 2]>>,
    /// Minimum hypernym distance to a root (memoized at load).
    root_depth: Vec<u8>,
    pub n_synsets: usize,
}

/// Maximum number of senses considered when expanding a lemma. Keeping this
/// small limits noise from rare senses (WordNet lists senses by frequency).
pub const MAX_SENSES: usize = 3;

impl LexGraph {
    pub fn is_empty(&self) -> bool {
        self.n_synsets == 0
    }

    pub fn parse(lemmas_tsv: &str, synsets_tsv: &str, antonyms_tsv: &str) -> LexGraph {
        let mut g = LexGraph::default();
        let mut syn_ids: FxHashMap<Box<str>, SynId> = FxHashMap::default();
        let syn_id = |s: &str, syn_ids: &mut FxHashMap<Box<str>, SynId>, g: &mut LexGraph| -> SynId {
            if let Some(&id) = syn_ids.get(s) {
                return id;
            }
            let id = g.synset_lemmas.len() as SynId;
            g.synset_lemmas.push(SmallVec::new());
            g.synset_hypernyms.push(SmallVec::new());
            g.synset_similar.push(SmallVec::new());
            g.synset_domains.push(SmallVec::new());
            syn_ids.insert(s.into(), id);
            id
        };
        fn lemma_id(g: &mut LexGraph, name: &str) -> LemmaId {
            if let Some(&id) = g.lemma_ids.get(name) {
                return id;
            }
            let id = g.lemma_names.len() as LemmaId;
            g.lemma_names.push(name.into());
            let st = if name.contains('_') { name.to_string() } else { stem(name) };
            g.lemma_stem.push(st.clone().into_boxed_str());
            g.lemma_synsets.push(SmallVec::new());
            if !name.contains('_') {
                g.stem_lemmas.entry(st.into_boxed_str()).or_default().push(id);
            }
            g.lemma_ids.insert(name.into(), id);
            id
        }
        for line in lemmas_tsv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split('\t');
            let (Some(lemma), Some(syns)) = (it.next(), it.next()) else { continue };
            let lid = lemma_id(&mut g, lemma);
            for s in syns.split(',') {
                if s.is_empty() {
                    continue;
                }
                let sid = syn_id(s, &mut syn_ids, &mut g);
                g.lemma_synsets[lid as usize].push(sid);
            }
        }
        for line in synsets_tsv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split('\t');
            let Some(sid_s) = it.next() else { continue };
            let sid = syn_id(sid_s, &mut syn_ids, &mut g);
            if let Some(lemmas) = it.next() {
                for l in lemmas.split('|') {
                    if l.is_empty() {
                        continue;
                    }
                    let lid = lemma_id(&mut g, l);
                    g.synset_lemmas[sid as usize].push(lid);
                }
            }
            if let Some(hyps) = it.next() {
                for h in hyps.split(',') {
                    if h.is_empty() {
                        continue;
                    }
                    let hid = syn_id(h, &mut syn_ids, &mut g);
                    g.synset_hypernyms[sid as usize].push(hid);
                }
            }
            if let Some(sims) = it.next() {
                for s in sims.split(',') {
                    if s.is_empty() {
                        continue;
                    }
                    let id2 = syn_id(s, &mut syn_ids, &mut g);
                    g.synset_similar[sid as usize].push(id2);
                }
            }
            if let Some(doms) = it.next() {
                for d in doms.split(',') {
                    if d.is_empty() {
                        continue;
                    }
                    let id2 = syn_id(d, &mut syn_ids, &mut g);
                    g.synset_domains[sid as usize].push(id2);
                }
            }
        }
        for line in antonyms_tsv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split('\t');
            let (Some(a), Some(b)) = (it.next(), it.next()) else { continue };
            let (a, b) = (lemma_id(&mut g, a), lemma_id(&mut g, b));
            g.antonyms.entry(a).or_default().push(b);
            g.antonyms.entry(b).or_default().push(a);
        }
        g.n_synsets = g.synset_lemmas.len();
        g.compute_root_depths();
        g
    }

    /// Memoized minimum distance to a root synset (one with no hypernyms).
    fn compute_root_depths(&mut self) {
        let n = self.synset_lemmas.len();
        let mut depth = vec![u8::MAX; n];
        // Iterative relaxation (graph is a DAG in practice; bounded passes guard against cycles).
        for (s, d) in depth.iter_mut().enumerate() {
            if self.synset_hypernyms[s].is_empty() {
                *d = 0;
            }
        }
        for _ in 0..24 {
            let mut changed = false;
            for s in 0..n {
                let mut best = depth[s];
                for &h in &self.synset_hypernyms[s] {
                    let d = depth[h as usize];
                    if d != u8::MAX && d.saturating_add(1) < best {
                        best = d + 1;
                    }
                }
                if best != depth[s] {
                    depth[s] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for d in depth.iter_mut() {
            if *d == u8::MAX {
                *d = 8;
            }
        }
        self.root_depth = depth;
    }

    #[inline]
    pub fn root_depth(&self, s: SynId) -> u8 {
        self.root_depth.get(s as usize).copied().unwrap_or(8)
    }

    #[inline]
    pub fn domains(&self, s: SynId) -> &[SynId] {
        self.synset_domains.get(s as usize).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Hypernym closure with depths: (depth, synset) for depth 1..=depth.
    pub fn hypernym_closure_with_depth(&self, s: SynId, depth: usize) -> SmallVec<[(u8, SynId); 8]> {
        let mut out: SmallVec<[(u8, SynId); 8]> = SmallVec::new();
        let mut frontier: SmallVec<[SynId; 4]> = SmallVec::new();
        frontier.push(s);
        for d in 1..=depth {
            let mut next: SmallVec<[SynId; 4]> = SmallVec::new();
            for &f in &frontier {
                for &h in self.hypernyms(f) {
                    if !out.iter().any(|(_, x)| *x == h) {
                        out.push((d as u8, h));
                        next.push(h);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        out
    }

    /// Lemma ids whose stem equals `st` (single-word lemmas).
    #[inline]
    pub fn lemmas_for_stem(&self, st: &str) -> &[LemmaId] {
        self.stem_lemmas.get(st).map(|v| v.as_slice()).unwrap_or(&[])
    }

    #[inline]
    pub fn lemma_id(&self, lemma: &str) -> Option<LemmaId> {
        self.lemma_ids.get(lemma).copied()
    }

    #[inline]
    pub fn lemma_name(&self, id: LemmaId) -> &str {
        &self.lemma_names[id as usize]
    }

    #[inline]
    pub fn lemma_stem(&self, id: LemmaId) -> &str {
        &self.lemma_stem[id as usize]
    }

    /// Synsets of a stem across its lemmas, first `MAX_SENSES` senses each.
    pub fn senses_of_stem(&self, st: &str, max_senses: usize) -> SmallVec<[SynId; 6]> {
        let mut out: SmallVec<[SynId; 6]> = SmallVec::new();
        for &l in self.lemmas_for_stem(st) {
            for &s in self.lemma_synsets[l as usize].iter().take(max_senses) {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        out
    }

    #[inline]
    pub fn synset_lemmas(&self, s: SynId) -> &[LemmaId] {
        &self.synset_lemmas[s as usize]
    }

    #[inline]
    pub fn hypernyms(&self, s: SynId) -> &[SynId] {
        &self.synset_hypernyms[s as usize]
    }

    #[inline]
    pub fn similar(&self, s: SynId) -> &[SynId] {
        &self.synset_similar[s as usize]
    }

    /// Hypernym closure up to `depth` levels (excluding `s` itself).
    pub fn hypernym_closure(&self, s: SynId, depth: usize) -> SmallVec<[SynId; 8]> {
        let mut out: SmallVec<[SynId; 8]> = SmallVec::new();
        let mut frontier: SmallVec<[SynId; 4]> = SmallVec::new();
        frontier.push(s);
        for _ in 0..depth {
            let mut next: SmallVec<[SynId; 4]> = SmallVec::new();
            for &f in &frontier {
                for &h in self.hypernyms(f) {
                    if !out.contains(&h) {
                        out.push(h);
                        next.push(h);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        out
    }

    /// Stems of synonyms of `st` (lemmas sharing one of its first senses),
    /// plus lemmas of "similar to" synsets (adjectives).
    pub fn synonym_stems(&self, st: &str, max_senses: usize) -> SmallVec<[&str; 8]> {
        let mut out: SmallVec<[&str; 8]> = SmallVec::new();
        for s in self.senses_of_stem(st, max_senses) {
            for &l in self.synset_lemmas(s) {
                let ls = self.lemma_stem(l);
                if ls != st && !ls.contains('_') && !out.contains(&ls) {
                    out.push(ls);
                }
            }
            for &sim in self.similar(s) {
                for &l in self.synset_lemmas(sim) {
                    let ls = self.lemma_stem(l);
                    if ls != st && !ls.contains('_') && !out.contains(&ls) {
                        out.push(ls);
                    }
                }
            }
        }
        out
    }

    /// Stems of antonyms of `st`.
    pub fn antonym_stems(&self, st: &str) -> SmallVec<[&str; 4]> {
        let mut out: SmallVec<[&str; 4]> = SmallVec::new();
        for &l in self.lemmas_for_stem(st) {
            if let Some(ants) = self.antonyms.get(&l) {
                for &a in ants {
                    let a_s = self.lemma_stem(a);
                    if !out.contains(&a_s) {
                        out.push(a_s);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LexGraph {
        LexGraph::parse(
            "refund\tn1,v1\nrepayment\tn1\nmoney\tn2\ndog\tn3\nanimal\tn4\nhappy\ta1\nunhappy\ta2\nglad\ta3\n",
            "n1\trefund|repayment\tn2\t\nn2\tmoney\t\t\nn3\tdog\tn4\t\nn4\tanimal\t\t\na1\thappy\t\ta3\na2\tunhappy\t\t\na3\tglad\t\ta1\nv1\trefund\t\t\n",
            "happy\tunhappy\n",
        )
    }

    #[test]
    fn synonyms_hypernyms_antonyms() {
        let g = sample();
        let syn = g.synonym_stems("refund", 3);
        assert!(syn.contains(&"repay"));
        let dog = g.senses_of_stem("dog", 3);
        let hyp = g.hypernym_closure(dog[0], 2);
        let animal = g.senses_of_stem("anim", 3);
        assert!(hyp.contains(&animal[0]));
        assert!(g.antonym_stems("happi").contains(&"unhappi"));
        assert!(g.synonym_stems("happi", 3).contains(&"glad"));
    }
}
