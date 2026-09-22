//! `sextant export-pairs`: dump neural training/inference inputs for a
//! dataset, using the production question analysis and evidence retrieval.

use super::common::{build_engine, expand_jsonl};
use super::dataset::{load_records, Record};
use super::learn::gold_index;
use serde::Serialize;
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;

#[derive(Serialize)]
struct Row<'a> {
    id: String,
    record_id: &'a str,
    question_id: &'a str,
    source: &'a str,
    tier: &'a str,
    split: &'a str,
    record_family: &'a str,
    transformation: &'a str,
    group: Option<&'a String>,
    #[serde(flatten)]
    export: sextant_core::export::QuestionExport,
    gold: usize,
    gold_probs: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra: Option<serde_json::Value>,
}

fn soft_probs(rec: &Record, qid: &str, n: usize) -> Option<Vec<f64>> {
    let gp = rec.gold_probs.as_ref()?.get(qid)?;
    match gp {
        serde_json::Value::Array(a) => Some((0..n).map(|i| a.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0)).collect()),
        serde_json::Value::Number(x) => {
            let p = x.as_f64().unwrap_or(0.5);
            Some(vec![1.0 - p, p])
        }
        serde_json::Value::Object(_) => None,
        _ => None,
    }
}

/// Retrieval / export knobs (one struct so experiments can add switches
/// without reshuffling a long argument list).
pub struct Opts {
    pub budget_words: usize,
    pub limit: usize,
    pub no_features: bool,
    pub strategy: String,
    pub adaptive_cap: usize,
    pub local_idf: bool,
    pub q_expand: bool,
    pub doc_order_fallback: bool,
    pub neighbour_glue: usize,
}

pub fn run(model_dir: Option<PathBuf>, threads: usize, inputs: Vec<PathBuf>, out: PathBuf, opts: Opts) -> i32 {
    let Opts { budget_words, limit, no_features, strategy, adaptive_cap, local_idf, q_expand, doc_order_fallback, neighbour_glue } = opts;
    let engine = match build_engine(model_dir, threads) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let params = sextant_core::export::EvidenceParams::from_name(&strategy, budget_words)
        .with_adaptive_cap(adaptive_cap)
        .with_local_idf(local_idf)
        .with_q_expand(q_expand)
        .with_doc_order_fallback(doc_order_fallback)
        .with_neighbour_glue(neighbour_glue);
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
    let mut n_rows = 0usize;
    let mut n_records = 0usize;
    let mut skipped = 0usize;
    let mut stats: std::collections::BTreeMap<String, usize> = Default::default();
    let t0 = std::time::Instant::now();
    for f in files {
        let recs = match load_records(&[f.clone()]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {e}");
                return 2;
            }
        };
        for rec in recs.iter().take(if limit > 0 { limit } else { usize::MAX }) {
            n_records += 1;
            let state = sextant_core::state::index::StateIndex::build(&rec.request.state, &engine.res);
            for (qid, q) in &rec.request.questions {
                let Some(gold_value) = rec.gold.get(qid) else { continue };
                let Some(gold) = gold_index(q, gold_value) else {
                    skipped += 1;
                    continue;
                };
                let mut export = sextant_core::export::export_question_with(q, &state, &engine.res, &engine.model, params);
                if no_features {
                    for c in export.candidates.iter_mut() {
                        c.features.clear();
                    }
                    export.feature_names.clear();
                }
                *stats.entry(export.kind.clone()).or_insert(0) += 1;
                let n = export.candidates.len();
                let row = Row {
                    id: format!("{}#{}", rec.id, qid),
                    record_id: &rec.id,
                    question_id: qid,
                    source: &rec.source,
                    tier: &rec.tier,
                    split: &rec.split,
                    record_family: &rec.family,
                    transformation: &rec.transformation,
                    group: rec.group.as_ref(),
                    gold,
                    gold_probs: soft_probs(rec, qid, n),
                    extra: (!rec.extra.is_empty()).then(|| json!(rec.extra)),
                    export,
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
    // Sidecar manifest: retrieval must be identical in training and
    // inference, so the parameters travel with the data instead of being
    // re-typed on the command line at export time.
    let meta = serde_json::json!({
        "budget_words": budget_words,
        "strategy": strategy,
        "adaptive_cap": adaptive_cap,
        "local_idf": local_idf,
        "q_expand": q_expand,
        "doc_order_fallback": doc_order_fallback,
        "neighbour_glue": neighbour_glue,
        "rows": n_rows,
        "records": n_records,
        "engine_version": env!("CARGO_PKG_VERSION"),
    });
    let meta_path = out.with_extension("meta.json");
    if let Err(e) = std::fs::write(&meta_path, format!("{}\n", serde_json::to_string_pretty(&meta).expect("serialize"))) {
        eprintln!("warning: could not write {}: {e}", meta_path.display());
    }
    eprintln!(
        "exported {n_rows} rows from {n_records} records ({skipped} skipped) in {:.1}s -> {} [{}]",
        t0.elapsed().as_secs_f64(),
        out.display(),
        stats.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
    );
    0
}
