//! `sextant neural-probe`: score exported pair rows with the Rust neural
//! runtime. Used to verify Python/Rust parity and to evaluate a scorer
//! without loading the full engine.
use serde_json::Value;
use std::io::{BufRead, Write};
use std::path::PathBuf;

pub fn run(neural_dir: PathBuf, input: PathBuf, out: PathBuf, limit: usize) -> i32 {
    #[cfg(not(feature = "neural"))]
    {
        let _ = (neural_dir, input, out, limit);
        eprintln!("error: this binary was built without the `neural` feature");
        return 2;
    }
    #[cfg(feature = "neural")]
    {
        let scorer = match sextant_core::neural::NeuralScorer::load(&neural_dir) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {e}");
                return 2;
            }
        };
        let f = match std::fs::File::open(&input) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error: {}: {e}", input.display());
                return 2;
            }
        };
        if let Some(p) = out.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let mut w = std::io::BufWriter::new(std::fs::File::create(&out).expect("create output"));
        let mut n = 0usize;
        let t0 = std::time::Instant::now();
        for line in std::io::BufReader::new(f).lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            let row: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: malformed row: {e}");
                    return 1;
                }
            };
            let question = row["question"].as_str().unwrap_or("").to_string();
            let evidence = row["evidence"].as_str().unwrap_or("").to_string();
            let cands = row["candidates"].as_array().cloned().unwrap_or_default();
            let texts: Vec<String> = cands
                .iter()
                .map(|c| {
                    let key = c["key"].as_str().unwrap_or("");
                    let text = c["text"].as_str().unwrap_or("").trim();
                    let kind = row["kind"].as_str().unwrap_or("choice");
                    if kind == "noul" {
                        let base = if key == "true" { "yes, the statement holds" } else { "no, the statement does not hold" };
                        if text.is_empty() || text == key {
                            base.to_string()
                        } else {
                            format!("{base}; {text}")
                        }
                    } else if text.is_empty() {
                        key.to_string()
                    } else {
                        text.to_string()
                    }
                })
                .collect();
            let features: Option<Vec<Vec<f32>>> = (scorer.config.n_features > 0)
                .then(|| cands.iter().map(|c| c["features"].as_array().map(|a| a.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect()).unwrap_or_default()).collect());
            let sym: Option<Vec<f32>> = scorer
                .config
                .use_symbolic_logit
                .then(|| cands.iter().map(|c| c["symbolic_logit"].as_f64().unwrap_or(0.0) as f32).collect());
            let logits = match scorer.score(&question, &evidence, &texts, features.as_deref(), sym.as_deref()) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: score: {e}");
                    return 1;
                }
            };
            writeln!(w, "{}", serde_json::json!({"id": row["id"], "logits": logits, "gold": row["gold"], "kind": row["kind"], "tier": row["tier"], "record_family": row["record_family"], "difficulty": row.get("extra").and_then(|e| e.get("difficulty"))})).expect("write");
            n += 1;
            if limit > 0 && n >= limit {
                break;
            }
        }
        let _ = w.flush();
        eprintln!("scored {n} questions in {:.1}s ({:.1}/s) -> {}", t0.elapsed().as_secs_f64(), n as f64 / t0.elapsed().as_secs_f64(), out.display());
        0
    }
}
