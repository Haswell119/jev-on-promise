//! `sextant export-retrieval`: dump per-segment retrieval features and
//! gold-evidence labels, so the learned ranker can be fitted offline.
//!
//! One row per question. Every segment that overlaps an annotated evidence
//! span is a positive; negatives are subsampled, because a 16k-token state
//! has hundreds of segments and almost all of them are irrelevant.

use super::common::{build_engine, expand_jsonl};
use super::dataset::load_records;
use serde::Serialize;
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;

#[derive(Serialize)]
struct Row<'a> {
    id: String,
    record_id: &'a str,
    question_id: &'a str,
    source: &'a str,
    tier: &'a str,
    family: &'a str,
    kind: String,
    /// Segments in the state before subsampling.
    n_segments: usize,
    /// Word count of each emitted segment (for budget-aware evaluation).
    seg_words: Vec<u16>,
    /// Document index of each emitted segment.
    seg_index: Vec<u32>,
    features: Vec<Vec<f32>>,
    labels: Vec<u8>,
    feature_names: Vec<&'static str>,
}

/// Shortest segment that may be labelled positive purely because the
/// annotated sentence contains it.
const MIN_SEGMENT_MATCH_CHARS: usize = 12;

fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = true;
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            for c in ch.to_lowercase() {
                out.push(c);
            }
            space = false;
        } else if !space {
            out.push(' ');
            space = true;
        }
    }
    out.trim_end().to_string()
}

fn gold_spans(extra: &indexmap::IndexMap<String, Value>) -> Vec<String> {
    let Some(meta) = extra.get("meta").and_then(|m| m.as_object()) else {
        return Vec::new();
    };
    let Some(ev) = meta.get("evidence").and_then(|e| e.as_array()) else {
        return Vec::new();
    };
    ev.iter().filter_map(|v| v.as_str()).map(normalize).filter(|s| !s.is_empty()).collect()
}

pub struct Opts {
    pub limit: usize,
    /// Word budget the ranker will have to fill. Questions whose whole
    /// state fits inside it are skipped: retrieval makes no choice there,
    /// so those lists teach nothing and saturate every metric.
    pub budget_words: usize,
    pub local_idf: bool,
    pub q_expand: bool,
    pub max_negatives: usize,
    pub seed: u64,
}

/// Deterministic per-row shuffle: an LCG seeded by the row id, so negative
/// sampling is reproducible without carrying an RNG across records.
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}

pub fn run(model_dir: Option<PathBuf>, threads: usize, inputs: Vec<PathBuf>, out: PathBuf, opts: Opts) -> i32 {
    let engine = match build_engine(model_dir, threads) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let files = expand_jsonl(&inputs);
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = match std::fs::File::create(&out) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {}: {e}", out.display());
            return 2;
        }
    };
    let mut w = std::io::BufWriter::with_capacity(1 << 20, file);
    let (mut n_rows, mut n_records, mut no_gold, mut no_positive) = (0usize, 0usize, 0usize, 0usize);
    let mut fits_budget = 0usize;
    let t0 = std::time::Instant::now();
    for f in files {
        let recs = match load_records(&[f.clone()]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {e}");
                return 2;
            }
        };
        for rec in recs.iter().take(if opts.limit > 0 { opts.limit } else { usize::MAX }) {
            n_records += 1;
            let spans = gold_spans(&rec.extra);
            if spans.is_empty() {
                no_gold += 1;
                continue;
            }
            let state = sextant_core::state::index::StateIndex::build(&rec.request.state, &engine.res);
            let state_words: usize = state.fields.iter().map(|f| f.text.split_whitespace().count()).sum();
            if opts.budget_words > 0 && state_words <= opts.budget_words {
                fits_budget += 1;
                continue;
            }
            let n_seg = state.segments.len();
            // Word-boundary containment in either direction: the annotated
            // sentence may be one segment, or span several. A minimum length
            // on the segment->span direction stops a one-word segment from
            // being labelled positive because that word happens to occur in
            // the annotation.
            let padded: Vec<String> = spans.iter().map(|g| format!(" {g} ")).collect();
            let mut positive = vec![false; n_seg];
            for i in 0..n_seg {
                let seg = normalize(state.segment_text(i as u32));
                if seg.is_empty() {
                    continue;
                }
                let pseg = format!(" {seg} ");
                positive[i] = padded.iter().zip(spans.iter()).any(|(pg, g)| {
                    (seg.len() >= MIN_SEGMENT_MATCH_CHARS && pg.contains(&pseg)) || pseg.contains(&format!(" {g} "))
                });
            }
            if !positive.iter().any(|p| *p) {
                no_positive += 1;
                continue;
            }
            for (qid, q) in &rec.request.questions {
                let view = sextant_core::question::QuestionView::build(q, &state, &engine.res);
                let feats = sextant_core::export::retrieval_features(&view, &state, opts.local_idf, opts.q_expand);
                let mut keep: Vec<usize> = (0..n_seg).filter(|i| positive[*i]).collect();
                let mut negatives: Vec<usize> = (0..n_seg).filter(|i| !positive[*i]).collect();
                if opts.max_negatives > 0 && negatives.len() > opts.max_negatives {
                    let mut st = opts.seed ^ (n_rows as u64).wrapping_mul(0x9E3779B97F4A7C15);
                    for i in (1..negatives.len()).rev() {
                        let j = (lcg(&mut st) as usize) % (i + 1);
                        negatives.swap(i, j);
                    }
                    negatives.truncate(opts.max_negatives);
                }
                keep.extend(negatives);
                keep.sort_unstable();
                let row = Row {
                    id: format!("{}#{}", rec.id, qid),
                    record_id: &rec.id,
                    question_id: qid,
                    source: &rec.source,
                    tier: &rec.tier,
                    family: &rec.family,
                    kind: format!("{:?}", view.kind).to_lowercase(),
                    n_segments: n_seg,
                    seg_words: keep
                        .iter()
                        .map(|&i| state.segment_text(i as u32).split_whitespace().count().min(u16::MAX as usize) as u16)
                        .collect(),
                    seg_index: keep.iter().map(|&i| i as u32).collect(),
                    features: keep.iter().map(|&i| feats[i].to_vec()).collect(),
                    labels: keep.iter().map(|&i| u8::from(positive[i])).collect(),
                    feature_names: sextant_core::retrieval::FEATURE_NAMES.to_vec(),
                };
                if let Err(e) = writeln!(w, "{}", serde_json::to_string(&row).expect("serialize")) {
                    eprintln!("error: write: {e}");
                    return 1;
                }
                n_rows += 1;
            }
        }
    }
    let _ = w.flush();
    eprintln!(
        "exported {n_rows} rows from {n_records} records in {:.1}s ({no_gold} without evidence \
         annotations, {fits_budget} whose whole state fits the budget, {no_positive} whose evidence \
         matched no segment) -> {}",
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    0
}
