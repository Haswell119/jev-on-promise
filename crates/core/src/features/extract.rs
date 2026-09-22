//! Multi-channel feature extraction for every criterion of a question.
//! All channels are deterministic lexical/statistical measurements over
//! the shared `StateIndex`; nothing here is learned except the weights that
//! later combine the features.

use super::{FeatureVec, F};
use crate::lexicon::Resources;
use crate::question::{Criterion, QuestionView};
use crate::state::index::{sorted_intersection, StateIndex, ATTR_KEY};
use crate::state::vocab::TermId;
use crate::text::negation::{DIRECTIVE, HYPOTHETICAL, INTERROGATIVE, NEGATED, REQUEST};
use crate::text::numbers::NumKind;

pub const BM25_K1: f32 = 1.2;
pub const BM25_B: f32 = 0.75;
/// Segments kept per criterion for the expensive channels and for explain output.
pub const TOP_K_SEGMENTS: usize = 5;

#[derive(Debug, Clone, Default)]
pub struct CriterionEvidence {
    /// (segment, retrieval score) best first.
    pub top: Vec<(u32, f32)>,
    /// Literal matches: (byte offset in lower_text, literal length).
    pub literal_hits: Vec<(usize, usize)>,
    /// Best matching literal text.
    pub best_literal: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    pub rows: Vec<FeatureVec>,
    pub evidence: Vec<CriterionEvidence>,
}

#[inline]
fn bm25_tf(tf: f32, len: f32, avg: f32) -> f32 {
    tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * len / avg))
}

#[inline]
fn seg_tf(seg: &crate::state::index::Segment, t: TermId) -> u16 {
    match seg.tf.binary_search_by_key(&t, |(a, _)| *a) {
        Ok(i) => seg.tf[i].1,
        Err(_) => 0,
    }
}

fn top_k(scores: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (i, &s) in scores.iter().enumerate() {
        if s <= 0.0 {
            continue;
        }
        if out.len() < k {
            out.push((i as u32, s));
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        } else if s > out[k - 1].1 {
            out[k - 1] = (i as u32, s);
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        }
    }
    out
}

#[inline]
fn is_word_boundary(text: &str, start: usize, end: usize) -> bool {
    let before = text[..start].chars().next_back().map(|c| !c.is_alphanumeric()).unwrap_or(true);
    let after = text[end..].chars().next().map(|c| !c.is_alphanumeric()).unwrap_or(true);
    before && after
}

/// Occurrence statistics of a term in the state.
#[derive(Default, Clone, Copy)]
struct Occ {
    n: u32,
    negated: u32,
    hyp: u32,
    req: u32,
    key: u32,
    key_negated: u32,
    in_focus: u32,
    directive: u32,
}

fn occurrences(state: &StateIndex, t: TermId, focus: &[u32]) -> Occ {
    let mut o = Occ::default();
    if let Some(pos) = state.term_tokens.get(&t) {
        for &p in pos {
            let tok = &state.tokens[p as usize];
            o.n += 1;
            if tok.flags & NEGATED != 0 {
                o.negated += 1;
            }
            if tok.flags & (HYPOTHETICAL | INTERROGATIVE) != 0 {
                o.hyp += 1;
            }
            if tok.flags & REQUEST != 0 {
                o.req += 1;
            }
            if tok.flags & DIRECTIVE != 0 {
                o.directive += 1;
            }
            if tok.attrs & ATTR_KEY != 0 {
                o.key += 1;
                if tok.flags & NEGATED != 0 {
                    o.key_negated += 1;
                }
            }
            if !focus.is_empty() {
                let field = state.segments[tok.seg as usize].field;
                if focus.binary_search(&field).is_ok() {
                    o.in_focus += 1;
                }
            }
        }
    }
    o
}

struct Scratch {
    bm25: Vec<f32>,
    cov: Vec<f32>,
    negcov: Vec<f32>,
    qcov: Vec<f32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Scratch { bm25: vec![0.0; n], cov: vec![0.0; n], negcov: vec![0.0; n], qcov: vec![0.0; n] }
    }
    fn reset(&mut self) {
        for v in self.bm25.iter_mut() {
            *v = 0.0;
        }
        for v in self.cov.iter_mut() {
            *v = 0.0;
        }
        for v in self.negcov.iter_mut() {
            *v = 0.0;
        }
    }
}

/// Add a term's postings into per-segment accumulators.
fn accumulate(state: &StateIndex, t: TermId, w: f32, bm25: &mut [f32], cov: &mut [f32]) {
    if let Some(post) = state.postings.get(&t) {
        for &(seg, tf) in post {
            let s = &state.segments[seg as usize];
            bm25[seg as usize] += w * bm25_tf(tf as f32, s.content_len as f32, state.avg_seg_len);
            cov[seg as usize] += w;
        }
    }
}

fn focus_valence(state: &StateIndex, focus: &[u32]) -> (f32, f32, bool) {
    if focus.is_empty() {
        return (state.valence, state.intensity, state.has_intensity);
    }
    let (mut vs, mut vn, mut is, mut inn) = (0.0f32, 0u32, 0.0f32, 0u32);
    for &f in focus {
        let fld = &state.fields[f as usize];
        for seg in fld.seg_start..fld.seg_end {
            let s = &state.segments[seg as usize];
            if s.valence != 0.0 {
                vs += s.valence;
                vn += 1;
            }
            if s.has_intensity {
                is += s.intensity;
                inn += 1;
            }
        }
    }
    let v = if vn > 0 { (vs / (vn as f32).sqrt()).clamp(-1.0, 1.0) } else { state.valence };
    let i = if inn > 0 { is / inn as f32 } else { state.intensity };
    (v, i, inn > 0 || state.has_intensity)
}

/// Verbatim literal search. Matches inside directive segments (text that
/// addresses the classifier) are counted separately and earn no credit.
fn literal_search(state: &StateIndex, lit: &str, focus: &[u32]) -> (u32, bool, Vec<(usize, usize)>, u32) {
    let mut count = 0u32;
    let mut directive = 0u32;
    let mut in_focus = false;
    let mut hits = Vec::new();
    let text = &state.lower_text;
    let mut start = 0usize;
    while let Some(pos) = text[start..].find(lit) {
        let abs = start + pos;
        let end = abs + lit.len();
        if is_word_boundary(text, abs, end) {
            let in_directive = state
                .segment_at_lower_offset(abs)
                .map(|s| state.segments[s as usize].flags_any & DIRECTIVE != 0)
                .unwrap_or(false);
            if in_directive {
                directive += 1;
            } else {
                count += 1;
                if hits.len() < 8 {
                    hits.push((abs, lit.len()));
                }
                if !focus.is_empty() && focus.binary_search(&state.field_at_offset(abs)).is_ok() {
                    in_focus = true;
                }
            }
        }
        start = end.max(abs + 1);
        if start >= text.len() {
            break;
        }
    }
    (count, in_focus, hits, directive)
}

/// +1 when every elapsed duration fits inside every window duration in the
/// state, −1 when an elapsed duration exceeds a window, 0 when unknown.
fn window_check(state: &StateIndex, focus: &[u32]) -> f32 {
    use crate::state::index::DurationKind;
    let in_focus = |seg: u32| focus.is_empty() || focus.binary_search(&state.segments[seg as usize].field).is_ok();
    let windows: Vec<f64> =
        state.durations.iter().filter(|d| d.kind == DurationKind::Window && in_focus(d.seg)).map(|d| d.days).collect();
    if windows.is_empty() {
        return 0.0;
    }
    let mut elapsed: Vec<f64> =
        state.durations.iter().filter(|d| d.kind == DurationKind::Ago && in_focus(d.seg)).map(|d| d.days).collect();
    if elapsed.is_empty() {
        if let Some(reference) = state.reference_date {
            let base = reference.days_from_epoch();
            for ds in &state.dates {
                let delta = base - ds.date.days_from_epoch();
                if delta > 0 && in_focus(ds.seg) {
                    elapsed.push(delta as f64);
                }
            }
        }
    }
    if elapsed.is_empty() {
        return 0.0;
    }
    let window = windows.iter().cloned().fold(f64::INFINITY, f64::min);
    let worst = elapsed.iter().cloned().fold(0.0f64, f64::max);
    if worst <= window {
        1.0
    } else {
        -1.0
    }
}

fn question_weight(q: &QuestionView) -> f32 {
    q.terms.iter().map(|t| t.weight).sum::<f32>().max(1e-6)
}

/// Extract features for all criteria of a question.
pub fn extract_features(q: &QuestionView, state: &StateIndex, _res: &Resources) -> FeatureMatrix {
    let n_seg = state.segments.len();
    let xf = super::cross::cross_field(q, state, _res);
    let mut scratch = Scratch::new(n_seg);
    let focus = &q.focus_fields;
    let (fvalence, fintensity, fhas_int) = focus_valence(state, focus);
    let q_w = question_weight(q);
    // Question-term coverage per segment (shared across criteria).
    for t in &q.terms {
        if let Some(post) = state.postings.get(&t.term) {
            for &(seg, _) in post {
                scratch.qcov[seg as usize] += t.weight;
            }
        }
    }
    let total_content = state.total_content.max(1) as f32;
    let mut rows = Vec::with_capacity(q.criteria.len());
    let mut evidence = Vec::with_capacity(q.criteria.len());

    for c in &q.criteria {
        scratch.reset();
        let mut f = FeatureVec::default();
        let pos_w = c.pos_weight.max(1e-6);
        let neg_w = c.neg_weight.max(1e-6);
        let (mut cov_w, mut cov_rare, mut rare_w) = (0.0f32, 0.0f32, 0.0f32);
        let (mut syn_cov, mut ant_hits) = (0.0f32, 0.0f32);
        let (mut neg_agree, mut neg_conflict, mut hyp_conflict, mut req_agree, mut hyp_state) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut focus_cov, mut matched_w) = (0.0f32, 0.0f32);
        let mut matched_occ = 0u32;
        let (mut directive_w, mut ant_neg) = (0.0f32, 0.0f32);
        let (mut hyper_num, mut hyper_den, mut domain_num) = (0.0f32, 0.0f32, 0.0f32);
        let (mut name_hit, mut name_n, mut key_match, mut key_neg) = (0.0f32, 0u32, 0.0f32, 0.0f32);
        let mut bm25_global = 0.0f32;
        let mut cos_num = 0.0f32;
        let mut c_norm = 0.0f32;
        let mut crit_terms: Vec<TermId> = Vec::with_capacity(c.terms.len());
        let mut neg_field_cov = 0.0f32;
        let mut present_terms: Vec<(TermId, f32)> = Vec::new();

        for t in &c.terms {
            if t.polarity < 0 {
                if state.contains_term(t.term) {
                    let o = occurrences(state, t.term, focus);
                    // negative-field evidence counts only when the state asserts it (not negated)
                    let assert_frac = 1.0 - o.negated as f32 / o.n.max(1) as f32;
                    neg_field_cov += t.weight * assert_frac;
                    if let Some(post) = state.postings.get(&t.term) {
                        for &(seg, _) in post {
                            scratch.negcov[seg as usize] += t.weight * assert_frac;
                        }
                    }
                }
                continue;
            }
            crit_terms.push(t.term);
            c_norm += t.weight * t.weight;
            if t.is_name {
                name_n += 1;
            }
            let present = state.global_tf.get(&t.term).copied();
            if t.idf >= 0.6 {
                rare_w += t.weight;
            }
            if present.is_none() && state.directive_tokens.contains_key(&t.term) {
                directive_w += t.weight;
                matched_w += t.weight;
            }
            if let Some(tf) = present {
                cov_w += t.weight;
                if t.idf >= 0.6 {
                    cov_rare += t.weight;
                }
                if t.is_name {
                    name_hit += 1.0;
                }
                bm25_global += t.weight * bm25_tf(tf as f32, total_content, total_content);
                cos_num += t.weight * (1.0 + (tf as f32).ln()) * state.vocab.idf(t.term);
                accumulate(state, t.term, t.weight, &mut scratch.bm25, &mut scratch.cov);
                present_terms.push((t.term, t.weight));
                let o = occurrences(state, t.term, focus);
                let n = o.n.max(1) as f32;
                matched_occ += o.n;
                matched_w += t.weight;
                let neg_frac = o.negated as f32 / n;
                let dir_n = state.directive_tokens.get(&t.term).copied().unwrap_or(0) as f32;
                directive_w += t.weight * (dir_n / (dir_n + n));
                let crit_neg = if t.negated { 1.0 } else { 0.0 };
                let agree = 1.0 - (crit_neg - neg_frac).abs();
                neg_agree += t.weight * agree;
                neg_conflict += t.weight * (1.0 - agree);
                let hyp_frac = o.hyp as f32 / n;
                hyp_state += t.weight * hyp_frac;
                if t.hypothetical {
                    hyp_conflict += t.weight * (1.0 - hyp_frac) * 0.5;
                } else {
                    hyp_conflict += t.weight * hyp_frac;
                }
                if t.request && o.req > 0 {
                    req_agree += t.weight * (o.req as f32 / n);
                }
                if !focus.is_empty() && o.in_focus > 0 {
                    focus_cov += t.weight;
                }
                if t.is_name && o.key > 0 {
                    key_match += 1.0;
                    if o.key_negated > 0 {
                        key_neg += 1.0;
                    }
                }
            }
        }
        // Hypernym / topic-domain compatibility for positive terms absent from the state.
        if !_res.graph.is_empty() && (!state.ancestors.is_empty() || !state.domains.is_empty()) {
            let g = &_res.graph;
            for t in &c.terms {
                if t.polarity < 0 || t.negated {
                    continue;
                }
                let info = q.vocab.info(t.term);
                if info.stop || info.func || info.stem.len() < 3 {
                    continue;
                }
                hyper_den += t.weight;
                if state.contains_term(t.term) {
                    hyper_num += t.weight;
                    domain_num += t.weight;
                    continue;
                }
                let mut best = 0.0f32;
                let mut dom_hit = 0.0f32;
                for sense in g.senses_of_stem(&info.stem, 2) {
                    if let Some(&w) = state.ancestors.get(&sense) {
                        best = best.max(w);
                    }
                    for (d, anc) in g.hypernym_closure_with_depth(sense, 3) {
                        if let Some(&w) = state.ancestors.get(&anc) {
                            best = best.max(w * 0.75f32.powi(d as i32));
                        }
                    }
                    for &dom in g.domains(sense) {
                        if let Some(&w) = state.domains.get(&dom) {
                            dom_hit = dom_hit.max(w);
                        }
                    }
                }
                hyper_num += t.weight * best.min(1.0);
                domain_num += t.weight * dom_hit.min(1.0);
            }
        }
        // Synonym-only matches for absent terms.
        if !c.expanded.is_empty() {
            let mut last_src: Option<TermId> = None;
            let mut credited = false;
            for &(syn, w, src) in &c.expanded {
                if last_src != Some(src) {
                    last_src = Some(src);
                    credited = false;
                }
                if credited || state.contains_term(src) {
                    continue;
                }
                if let Some(_tf) = state.global_tf.get(&syn) {
                    let o = occurrences(state, syn, focus);
                    let neg_frac = o.negated as f32 / o.n.max(1) as f32;
                    syn_cov += w * (1.0 - neg_frac);
                    accumulate(state, syn, w, &mut scratch.bm25, &mut scratch.cov);
                    credited = true;
                }
            }
        }
        for &(ant, w) in &c.antonyms {
            if state.contains_term(ant) {
                let o = occurrences(state, ant, focus);
                let neg_frac = o.negated as f32 / o.n.max(1) as f32;
                ant_hits += w * (1.0 - neg_frac);
                ant_neg += w * neg_frac;
            }
        }

        // Best segments.
        let top = top_k(&scratch.bm25, TOP_K_SEGMENTS);
        let best_bm25 = top.first().map(|x| x.1).unwrap_or(0.0);
        let cov_best = scratch.cov.iter().cloned().fold(0.0f32, f32::max);
        let neg_field_best = scratch.negcov.iter().cloned().fold(0.0f32, f32::max);
        let c_norm = c_norm.sqrt().max(1e-6);
        let cos_global = if state.global_norm > 0.0 { cos_num / (c_norm * state.global_norm) } else { 0.0 };

        // Expensive channels on the top segments only.
        let (mut cos_best, mut jacc_best, mut dice_best, mut q_cov_best) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        crit_terms.sort_unstable();
        crit_terms.dedup();
        for &(seg, _) in &top {
            let s = &state.segments[seg as usize];
            let mut num = 0.0f32;
            let mut inter = 0usize;
            for t in &c.terms {
                if t.polarity < 0 {
                    continue;
                }
                let tf = seg_tf(s, t.term);
                if tf > 0 {
                    num += t.weight * (1.0 + (tf as f32).ln()) * state.vocab.idf(t.term);
                }
            }
            for &t in &crit_terms {
                if seg_tf(s, t) > 0 {
                    inter += 1;
                }
            }
            if s.norm > 0.0 {
                cos_best = cos_best.max(num / (c_norm * s.norm));
            }
            let union = crit_terms.len() + s.tf.len() - inter;
            if union > 0 {
                jacc_best = jacc_best.max(inter as f32 / union as f32);
            }
            if !c.grams.is_empty() && !s.grams.is_empty() {
                let i = sorted_intersection(&c.grams, &s.grams) as f32;
                dice_best = dice_best.max(2.0 * i / (c.grams.len() + s.grams.len()) as f32);
            }
            q_cov_best = q_cov_best.max(scratch.qcov[seg as usize] / q_w);
        }
        let gram_cov = if c.grams.is_empty() {
            0.0
        } else {
            sorted_intersection(&c.grams, &state.global_grams) as f32 / c.grams.len() as f32
        };
        let bigram_hits = if c.bigrams.is_empty() {
            0.0
        } else {
            c.bigrams.iter().filter(|b| state.bigrams.contains_key(b)).count() as f32 / c.bigrams.len() as f32
        };

        // Literals.
        let mut lit_hit = 0.0f32;
        let mut lit_len = 0.0f32;
        let mut lit_count = 0u32;
        let mut lit_focus = 0.0f32;
        let mut ev = CriterionEvidence { top: top.clone(), literal_hits: Vec::new(), best_literal: None };
        for lit in &c.literals {
            let ntoks = lit.split_whitespace().count();
            if lit.len() < 3 || (ntoks == 1 && crate::text::stem::is_stopword(lit)) {
                continue;
            }
            let (count, in_focus, hits, _directive) = literal_search(state, lit, focus);
            if count > 0 {
                lit_hit = 1.0;
                lit_count += count;
                let l = (ntoks.min(8) as f32) / 8.0;
                if l > lit_len {
                    lit_len = l;
                    ev.best_literal = Some(lit.clone());
                }
                if in_focus {
                    lit_focus = 1.0;
                }
                ev.literal_hits.extend(hits);
            }
        }

        // Example / facet phrases.
        let mut phrase_scores: Vec<f32> = Vec::new();
        for p in c.phrases.iter().filter(|p| p.polarity > 0) {
            if p.terms.is_empty() {
                continue;
            }
            let present = p.terms.iter().filter(|t| state.contains_term(**t)).count() as f32 / p.terms.len() as f32;
            let g = if p.grams.is_empty() {
                0.0
            } else {
                sorted_intersection(&p.grams, &state.global_grams) as f32 / p.grams.len() as f32
            };
            let b = if p.bigrams.is_empty() {
                present
            } else {
                p.bigrams.iter().filter(|b| state.bigrams.contains_key(b)).count() as f32 / p.bigrams.len() as f32
            };
            phrase_scores.push(0.4 * present + 0.3 * g + 0.3 * b);
        }
        phrase_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let example_max = phrase_scores.first().copied().unwrap_or(0.0);
        let example_mean = if phrase_scores.is_empty() {
            0.0
        } else {
            phrase_scores.iter().take(2).sum::<f32>() / phrase_scores.iter().take(2).count() as f32
        };

        // Numeric ranges.
        let (mut range_hit, mut range_dist) = (0.0f32, 0.0f32);
        if let Some(r) = &c.range {
            let mut best = f64::INFINITY;
            let candidates: Vec<&crate::state::index::NumberSpan> = state
                .numbers
                .iter()
                .filter(|n| match r.kind {
                    NumKind::Currency => n.kind == NumKind::Currency || n.kind == NumKind::Plain,
                    NumKind::Percent => n.kind == NumKind::Percent,
                    _ => n.kind != NumKind::Ordinal,
                })
                .filter(|n| focus.is_empty() || focus.binary_search(&state.segments[n.seg as usize].field).is_ok())
                .collect();
            for n in candidates {
                let d = r.distance(n.value);
                if d < best {
                    best = d;
                }
            }
            if best.is_finite() {
                range_hit = if best == 0.0 { 1.0 } else { 0.0 };
                range_dist = -(best.min(1.0) as f32);
            }
        }

        f.set(F::bm25_best, best_bm25 / pos_w);
        f.set(F::bm25_global, bm25_global / pos_w);
        f.set(F::cov_w, cov_w / pos_w);
        f.set(F::cov_best, cov_best / pos_w);
        f.set(F::cov_rare, if rare_w > 0.0 { cov_rare / rare_w } else { 0.0 });
        f.set(F::cov_name, if name_n > 0 { name_hit / name_n as f32 } else { 0.0 });
        f.set(F::cos_global, cos_global);
        f.set(F::cos_best, cos_best);
        f.set(F::jaccard_best, jacc_best);
        f.set(F::bigram_hits, bigram_hits);
        f.set(F::gram_dice_best, dice_best);
        f.set(F::gram_cov, gram_cov);
        f.set(F::literal_hit, lit_hit);
        f.set(F::literal_len, lit_len);
        f.set(F::literal_count, (1.0 + lit_count as f32).ln());
        f.set(F::syn_cov, syn_cov / pos_w);
        f.set(F::antonym_hits, ant_hits / pos_w);
        f.set(F::neg_agree, neg_agree / pos_w);
        f.set(F::neg_conflict, neg_conflict / pos_w);
        f.set(F::hyp_conflict, hyp_conflict / pos_w);
        f.set(F::req_agree, req_agree / pos_w);
        f.set(F::neg_field_cov, if c.n_neg_terms > 0 { neg_field_cov / neg_w } else { 0.0 });
        f.set(F::neg_field_best, if c.n_neg_terms > 0 { neg_field_best / neg_w } else { 0.0 });
        f.set(F::example_max, example_max);
        f.set(F::example_mean, example_mean);
        f.set(F::focus_cov, if focus.is_empty() { cov_w / pos_w } else { focus_cov / pos_w });
        f.set(F::focus_literal, lit_focus);
        f.set(F::valence_agree, c.valence * fvalence);
        f.set(F::valence_abs, c.valence.abs());
        f.set(F::intensity_dist, if c.has_intensity && fhas_int { -(c.intensity - fintensity).abs() } else { 0.0 });
        f.set(F::intensity_cov, if c.has_intensity && fhas_int { 1.0 } else { 0.0 });
        f.set(F::range_hit, range_hit);
        f.set(F::range_dist, range_dist);
        f.set(F::evidence_density, (matched_occ as f32) / total_content.sqrt());
        f.set(F::term_count_log, (1.0 + c.n_pos_terms as f32).ln());
        f.set(F::ood, if rare_w > 0.0 { 1.0 - cov_rare / rare_w } else { 0.0 });
        f.set(F::q_cov_best, q_cov_best);
        f.set(F::key_match, if name_n > 0 { key_match / name_n as f32 } else { 0.0 });
        f.set(F::key_negated, if name_n > 0 { key_neg / name_n as f32 } else { 0.0 });
        f.set(F::hyp_state, if matched_w > 0.0 { hyp_state / matched_w } else { 0.0 });
        f.set(F::null_desc, if c.is_null { 1.0 } else { 0.0 });
        f.set(F::hyper_match, if hyper_den > 0.0 { hyper_num / hyper_den } else { 0.0 });
        f.set(F::domain_match, if hyper_den > 0.0 { domain_num / hyper_den } else { 0.0 });
        f.set(F::antonym_negated, ant_neg / pos_w);
        f.set(F::directive_frac, if matched_w > 0.0 { directive_w / matched_w } else { 0.0 });
        // Cross-field channel: on every row for Choice/Score (crossed with the
        // option's polarity profile), only on the YES hypothesis for Noul so
        // that the yes−no difference carries the signal.
        let apply_xf = xf.available && !(q.kind == crate::api::QuestionKind::Noul && c.index == 1);
        if apply_xf {
            let sim = (xf.cov_ab + xf.cov_ba) / 2.0;
            let conflict = xf.neg_conflict + xf.antonym + 0.5 * xf.num_conflict;
            let pos_share = (1.0 - c.neg_share - c.hyp_share).max(0.0);
            f.set(F::xfield_available, 1.0);
            f.set(F::xfield_cov_ab, xf.cov_ab);
            f.set(F::xfield_cov_ba, xf.cov_ba);
            f.set(F::xfield_jaccard, xf.jaccard);
            f.set(F::xfield_cos, xf.cos);
            f.set(F::xfield_gram, xf.gram);
            f.set(F::xfield_neg_conflict, xf.neg_conflict);
            f.set(F::xfield_antonym, xf.antonym);
            f.set(F::xfield_num_conflict, xf.num_conflict);
            f.set(F::x_sim_pos, sim * pos_share);
            f.set(F::x_conflict_neg, conflict.min(1.0) * c.neg_share);
            f.set(F::x_low_neutral, (1.0 - sim) * c.hyp_share);
        }
        f.set(F::opt_neg_share, c.neg_share);
        f.set(F::opt_hyp_share, c.hyp_share);
        // Duration window check ("within 30 days" vs "20 days ago" / dated event vs reference date).
        if q.kind == crate::api::QuestionKind::Noul && c.index == 0 {
            f.set(F::window_ok, window_check(state, focus));
        }
        f.set(F::bias, 1.0);
        rows.push(f);
        evidence.push(ev);
    }
    // Ordinal position channel for Score: if the levels' intensity or valence
    // is monotone in the level index, map the state's intensity/valence onto
    // the scale and reward levels near that position.
    if q.kind == crate::api::QuestionKind::Score && q.criteria.len() >= 2 {
        let k = q.criteria.len();
        let idx: Vec<f32> = (0..k).map(|i| i as f32 / (k - 1) as f32).collect();
        let ints: Vec<f32> = q.criteria.iter().map(|c| c.intensity).collect();
        let vals: Vec<f32> = q.criteria.iter().map(|c| c.valence).collect();
        let corr = |x: &[f32], y: &[f32]| -> f32 {
            let n = x.len() as f32;
            let mx = x.iter().sum::<f32>() / n;
            let my = y.iter().sum::<f32>() / n;
            let num: f32 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
            let dx: f32 = x.iter().map(|a| (a - mx).powi(2)).sum::<f32>().sqrt();
            let dy: f32 = y.iter().map(|b| (b - my).powi(2)).sum::<f32>().sqrt();
            if dx < 1e-6 || dy < 1e-6 {
                0.0
            } else {
                num / (dx * dy)
            }
        };
        let has_int = q.criteria.iter().filter(|c| c.has_intensity).count() >= k.div_ceil(2) && fhas_int;
        let c_int = if has_int { corr(&idx, &ints) } else { 0.0 };
        let c_val = corr(&idx, &vals);
        let mut positions: Vec<f32> = Vec::new();
        if c_int.abs() >= 0.5 {
            let p = if c_int > 0.0 { fintensity } else { 1.0 - fintensity };
            positions.push(p);
        }
        if c_val.abs() >= 0.5 && fvalence != 0.0 {
            let v = (fvalence + 1.0) / 2.0;
            positions.push(if c_val > 0.0 { v } else { 1.0 - v });
        }
        for (i, row) in rows.iter_mut().enumerate() {
            let level_pos = idx[i];
            if c_int.abs() >= 0.5 {
                let p = if c_int > 0.0 { fintensity } else { 1.0 - fintensity };
                row.set(F::ord_pos_int, 1.0 - (p - level_pos).abs());
            }
            if c_val.abs() >= 0.5 && fvalence != 0.0 {
                let v = (fvalence + 1.0) / 2.0;
                let p = if c_val > 0.0 { v } else { 1.0 - v };
                row.set(F::ord_pos_val, 1.0 - (p - level_pos).abs());
            }
            if !positions.is_empty() {
                let p = positions.iter().sum::<f32>() / positions.len() as f32;
                let nearest = (p * (k - 1) as f32).round() as usize;
                row.set(F::ord_hit, if nearest == i { 1.0 } else { 0.0 });
            }
        }
    }
    FeatureMatrix { rows, evidence }
}

/// Helper used by explain output: evidence items for a criterion.
pub fn evidence_items(state: &StateIndex, ev: &CriterionEvidence, limit: usize) -> Vec<crate::api::EvidenceItem> {
    ev.top
        .iter()
        .take(limit)
        .map(|(seg, score)| crate::api::EvidenceItem {
            path: state.segment_path(*seg).to_string(),
            text: state.segment_text(*seg).chars().take(300).collect(),
            score: *score as f64,
        })
        .collect()
}

#[allow(dead_code)]
fn _unused(_c: &Criterion) {}
