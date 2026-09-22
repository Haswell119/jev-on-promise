#![allow(clippy::too_many_arguments)]
//! `sextant eval`: run the engine over JSONL records, compute metrics grouped
//! by a metadata field, optionally dump raw outputs and failures.

use super::common::build_engine;
use super::dataset::{load_records, write_jsonl};
use super::metrics::{score_answer, summarize, Outcome, Summary};
use indexmap::IndexMap;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

pub fn run(model_dir: Option<PathBuf>, threads: usize, inputs: Vec<PathBuf>, raw_out: Option<PathBuf>, out: Option<PathBuf>, group_by: &str, limit: usize, show_failures: bool, filter: Option<String>) -> i32 {
    let engine = match build_engine(model_dir, threads) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let mut records = match load_records(&inputs) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    if let Some(f) = &filter {
        records.retain(|r| r.id.contains(f.as_str()));
    }
    if limit > 0 {
        records.truncate(limit);
    }
    let mut outcomes: Vec<(usize, Outcome)> = Vec::new();
    let mut raws: Vec<serde_json::Value> = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();
    let mut errors = 0usize;
    let t_all = Instant::now();
    for (ri, rec) in records.iter().enumerate() {
        let mut req = rec.request.clone();
        req.explain = Some(true);
        let t0 = Instant::now();
        let resp = engine.evaluate(&req);
        latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
        match resp {
            Ok(resp) => {
                for (qid, gold) in &rec.gold {
                    if let Some(ans) = resp.answers.get(qid) {
                        let gp = rec.gold_probs.as_ref().and_then(|m| m.get(qid));
                        let o = score_answer(&rec.id, qid, ans, gold, gp);
                        if show_failures && !o.correct {
                            let ex = ans.explain();
                            eprintln!("FAIL {} [{}] {}: predicted={} gold={} p_gold={:.3} path={} family={}", rec.id, rec.field(group_by), qid, o.predicted, o.gold, o.p_gold, ex.map(|e| e.path.as_str()).unwrap_or("-"), ex.map(|e| e.family.as_str()).unwrap_or("-"));
                            if let Some(e) = ex {
                                for ev in e.top_evidence.iter().take(2) {
                                    eprintln!("     evidence[{}]: {}", ev.path, ev.text.chars().take(160).collect::<String>());
                                }
                                let mut rs: Vec<(&String, &f64)> = e.raw_scores.iter().collect();
                                rs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                                let top: Vec<String> = rs.iter().take(4).map(|(k, v)| format!("{k}={v:.2}")).collect();
                                eprintln!("     raw: {}", top.join(" "));
                            }
                        }
                        let mut stripped = ans.clone();
                        stripped.strip_explain();
                        raws.push(json!({"id": rec.id, "question": qid, "answer": stripped, "gold": gold, "correct": o.correct, "p_gold": o.p_gold, "path": o.path, "family": o.family, "group": rec.field(group_by)}));
                        outcomes.push((ri, o));
                    } else {
                        errors += 1;
                    }
                }
            }
            Err(e) => {
                errors += 1;
                eprintln!("ERROR {}: {}", rec.id, e);
                for (qid, gold) in &rec.gold {
                    raws.push(json!({"id": rec.id, "question": qid, "error": e, "gold": gold, "correct": false}));
                }
            }
        }
    }
    let total_s = t_all.elapsed().as_secs_f64();
    // Group summaries.
    let mut groups: IndexMap<String, Vec<&Outcome>> = IndexMap::new();
    for (ri, o) in &outcomes {
        groups.entry(records[*ri].field(group_by)).or_default().push(o);
    }
    let all: Vec<&Outcome> = outcomes.iter().map(|(_, o)| o).collect();
    let overall = summarize(&all);
    let mut by_kind: IndexMap<String, Summary> = IndexMap::new();
    for kind in ["choice", "score", "noul"] {
        let subset: Vec<&Outcome> = all.iter().copied().filter(|o| o.kind == kind).collect();
        if !subset.is_empty() {
            by_kind.insert(kind.into(), summarize(&subset));
        }
    }
    let mut by_group: IndexMap<String, Summary> = IndexMap::new();
    let mut keys: Vec<String> = groups.keys().cloned().collect();
    keys.sort();
    println!("{:<28} {:>6} {:>8} {:>7} {:>7} {:>7} {:>7}", group_by, "n", "acc", "nll", "brier", "ece", "valid");
    for k in keys {
        let s = summarize(&groups[&k]);
        println!("{:<28} {:>6} {:>8.4} {:>7.3} {:>7.3} {:>7.3} {:>7.3}", k, s.n, s.accuracy, s.nll, s.brier, s.ece, s.schema_validity);
        by_group.insert(k, s);
    }
    println!("{:<28} {:>6} {:>8.4} {:>7.3} {:>7.3} {:>7.3} {:>7.3}", "ALL", overall.n, overall.accuracy, overall.nll, overall.brier, overall.ece, overall.schema_validity);
    for (k, s) in &by_kind {
        println!("{:<28} {:>6} {:>8.4} {:>7.3} {:>7.3} {:>7.3} {:>7.3}", format!("kind:{k}"), s.n, s.accuracy, s.nll, s.brier, s.ece, s.schema_validity);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 { if latencies.is_empty() { 0.0 } else { latencies[(((latencies.len() - 1) as f64) * p).round() as usize] } };
    println!("records={} questions={} errors={} p50={:.2}ms p95={:.2}ms total={:.1}s", records.len(), outcomes.len(), errors, pct(0.5), pct(0.95), total_s);
    // Paraphrase / permutation stability over groups.
    let mut group_answers: IndexMap<String, Vec<(String, f64)>> = IndexMap::new();
    for (ri, o) in &outcomes {
        if let Some(g) = &records[*ri].group {
            // Only compare answers to the same question over the same label set.
            let labels = match records[*ri].request.questions.get(&o.question_id) {
                Some(sextant_core::Question::Choice(c)) => {
                    let mut k: Vec<&String> = c.criteria.keys().collect();
                    k.sort();
                    k.iter().map(|x| x.as_str()).collect::<Vec<_>>().join("|")
                }
                Some(sextant_core::Question::Score(sc)) => format!("score{}", sc.criteria.len()),
                _ => "noul".to_string(),
            };
            group_answers.entry(format!("{g}#{}#{labels}", o.question_id)).or_default().push((o.predicted.clone(), o.p_max));
        }
    }
    let mut stable = 0usize;
    let mut n_groups = 0usize;
    let mut prob_dist = 0.0f64;
    for (_, v) in group_answers.iter() {
        if v.len() < 2 {
            continue;
        }
        n_groups += 1;
        if v.iter().all(|(p, _)| p == &v[0].0) {
            stable += 1;
        }
        let mean = v.iter().map(|(_, p)| p).sum::<f64>() / v.len() as f64;
        prob_dist += v.iter().map(|(_, p)| (p - mean).abs()).sum::<f64>() / v.len() as f64;
    }
    let stability = if n_groups > 0 { Some(json!({"groups": n_groups, "same_answer_rate": stable as f64 / n_groups as f64, "mean_prob_distance": prob_dist / n_groups as f64})) } else { None };
    if let Some(s) = &stability {
        println!("group stability: {}", s);
    }
    let report = json!({
        "engine_version": sextant_core::VERSION,
        "model": engine.model.source,
        "inputs": inputs,
        "records": records.len(),
        "questions": outcomes.len(),
        "errors": errors,
        "group_by": group_by,
        "overall": overall,
        "by_kind": by_kind,
        "by_group": by_group,
        "group_stability": stability,
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99), "total_s": total_s},
    });
    if let Some(p) = out {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&p, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
        println!("wrote {}", p.display());
    }
    if let Some(p) = raw_out {
        write_jsonl(&p, &raws).expect("write raw");
        println!("wrote {}", p.display());
    }
    0
}
