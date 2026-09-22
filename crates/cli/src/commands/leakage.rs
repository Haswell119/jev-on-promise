//! `sextant leakage`: contamination scan between training/calibration data
//! and evaluation data (JevBench public items, jev-bench records, internal
//! eval sets). Detects exact duplicates, normalized duplicates, high token
//! 5-shingle overlap and MinHash near-duplicates. Writes `reports/leakage.json`.
//!
//! Accepted eval formats: our record format (`request`), JevBench tasks
//! (`state` + `question` + `labels`), jev-bench rows (`state` and `question`
//! as JSON strings + `primitive`).

use super::common::expand_jsonl;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use serde_json::{json, Value};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

const N_PERM: usize = 128;
const BANDS: usize = 16;
const ROWS: usize = 8;
const SHINGLE: usize = 5;

#[derive(Debug, Clone)]
struct Unit {
    id: String,
    file: String,
    kind: &'static str, // "state" | "instructions"
    exact: u64,
    normalized: u64,
    shingles: FxHashSet<u64>,
    minhash: Vec<u64>,
    text_preview: String,
}

fn h64(s: &str) -> u64 {
    let mut h = FxHasher::default();
    s.hash(&mut h);
    h.finish()
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn normalize(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut last_space = true;
    for c in lower.chars() {
        if c.is_alphanumeric() {
            out.push(c);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_string()
}

fn collect_text(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| collect_text(x, out)),
        Value::Object(o) => o.values().for_each(|x| collect_text(x, out)),
        Value::Number(n) => out.push(n.to_string()),
        _ => {}
    }
}

fn text_of(v: &Value) -> String {
    let mut parts = Vec::new();
    collect_text(v, &mut parts);
    parts.join(" ")
}

fn decode_maybe_json_string(v: &Value) -> Value {
    if let Value::String(s) = v {
        if let Ok(inner) = serde_json::from_str::<Value>(s) {
            return inner;
        }
    }
    v.clone()
}

/// (id, state text, instructions text)
fn extract(row: &Value) -> Option<(String, String, String)> {
    let id = row.get("id").map(|v| match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    })?;
    if let Some(req) = row.get("request") {
        let state = req.get("state").map(text_of).unwrap_or_default();
        let instr = req
            .get("questions")
            .map(|qs| {
                qs.as_object()
                    .map(|o| {
                        o.values()
                            .map(|q| {
                                text_of(q.get("instructions").unwrap_or(&Value::Null))
                                    + " "
                                    + &text_of(q.get("criteria").unwrap_or(&Value::Null))
                            })
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        return Some((id, state, instr));
    }
    if let Some(state) = row.get("state") {
        let state_v = decode_maybe_json_string(state);
        let q = row.get("question").map(decode_maybe_json_string).unwrap_or(Value::Null);
        let instr = text_of(q.get("instructions").unwrap_or(&Value::Null))
            + " "
            + &text_of(q.get("criteria").unwrap_or(&Value::Null));
        return Some((id, text_of(&state_v), instr));
    }
    None
}

struct Hasher128 {
    a: Vec<u64>,
    b: Vec<u64>,
}

impl Hasher128 {
    fn new() -> Self {
        // deterministic pseudo-random coefficients (splitmix64)
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let a: Vec<u64> = (0..N_PERM).map(|_| next() | 1).collect();
        let b: Vec<u64> = (0..N_PERM).map(|_| next()).collect();
        Hasher128 { a, b }
    }
    fn minhash(&self, shingles: &FxHashSet<u64>) -> Vec<u64> {
        let mut sig = vec![u64::MAX; N_PERM];
        for &s in shingles {
            for ((sv, a), b) in sig.iter_mut().zip(self.a.iter()).zip(self.b.iter()) {
                let v = a.wrapping_mul(s).wrapping_add(*b);
                if v < *sv {
                    *sv = v;
                }
            }
        }
        sig
    }
}

fn shingles(norm: &str) -> FxHashSet<u64> {
    let toks: Vec<&str> = norm.split(' ').filter(|t| !t.is_empty()).collect();
    let mut set = FxHashSet::default();
    if toks.len() < SHINGLE {
        if !toks.is_empty() {
            set.insert(h64(&toks.join(" ")));
        }
        return set;
    }
    for w in toks.windows(SHINGLE) {
        set.insert(h64(&w.join(" ")));
    }
    set
}

fn make_units(paths: &[PathBuf], hasher: &Hasher128, min_chars: usize) -> Vec<Unit> {
    let mut out = Vec::new();
    for p in expand_jsonl(paths) {
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(row) = serde_json::from_str::<Value>(line) else { continue };
            let Some((id, state, instr)) = extract(&row) else { continue };
            for (kind, txt) in [("state", state), ("instructions", instr)] {
                let norm = normalize(&txt);
                if norm.len() < min_chars {
                    continue;
                }
                let sh = shingles(&norm);
                out.push(Unit {
                    id: id.clone(),
                    file: p.display().to_string(),
                    kind,
                    exact: h64(&txt) ^ fnv(&txt).rotate_left(17),
                    normalized: h64(&norm) ^ fnv(&norm).rotate_left(17),
                    minhash: hasher.minhash(&sh),
                    shingles: sh,
                    text_preview: norm.chars().take(120).collect(),
                });
            }
        }
    }
    out
}

fn jaccard(a: &FxHashSet<u64>, b: &FxHashSet<u64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.iter().filter(|x| b.contains(x)).count();
    inter as f64 / (a.len() + b.len() - inter) as f64
}

pub fn run(
    train: Vec<PathBuf>,
    eval: Vec<PathBuf>,
    out: PathBuf,
    exclusions_out: Option<PathBuf>,
    internal: String,
) -> i32 {
    let internal_markers: Vec<String> =
        internal.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let is_internal = |file: &str| internal_markers.iter().any(|m| file.contains(m.as_str()));
    let hasher = Hasher128::new();
    let train_units = make_units(&train, &hasher, 20);
    let eval_units = make_units(&eval, &hasher, 20);
    eprintln!("train units: {} eval units: {}", train_units.len(), eval_units.len());
    // Every training unit per hash (duplicates inside the training data must all be excluded).
    let mut exact: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
    let mut normalized: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
    for (i, u) in train_units.iter().enumerate() {
        exact.entry(u.exact).or_default().push(i);
        normalized.entry(u.normalized).or_default().push(i);
    }
    // LSH buckets over train
    let mut buckets: FxHashMap<(usize, u64), Vec<usize>> = FxHashMap::default();
    for (i, u) in train_units.iter().enumerate() {
        for b in 0..BANDS {
            let mut h = FxHasher::default();
            u.minhash[b * ROWS..(b + 1) * ROWS].hash(&mut h);
            buckets.entry((b, h.finish())).or_default().push(i);
        }
    }
    let mut exact_hits = Vec::new();
    let mut norm_hits = Vec::new();
    let mut near_hits = Vec::new();
    let mut max_jaccard = 0.0f64;
    for (ei, e) in eval_units.iter().enumerate() {
        if let Some(tis) = exact.get(&e.exact) {
            for &ti in tis {
                exact_hits.push(json!({"eval_id": e.id, "eval_file": e.file, "kind": e.kind, "train_id": train_units[ti].id, "train_file": train_units[ti].file}));
            }
            continue;
        }
        if let Some(tis) = normalized.get(&e.normalized) {
            for &ti in tis {
                norm_hits.push(json!({"eval_id": e.id, "eval_file": e.file, "kind": e.kind, "train_id": train_units[ti].id, "train_file": train_units[ti].file}));
            }
            continue;
        }
        let mut cands: FxHashSet<usize> = FxHashSet::default();
        for b in 0..BANDS {
            let mut h = FxHasher::default();
            e.minhash[b * ROWS..(b + 1) * ROWS].hash(&mut h);
            if let Some(v) = buckets.get(&(b, h.finish())) {
                cands.extend(v.iter().copied());
            }
        }
        let mut best = (0.0f64, None);
        let mut cands_scored: Vec<(usize, f64)> = Vec::new();
        for ti in cands {
            let j = jaccard(&e.shingles, &train_units[ti].shingles);
            cands_scored.push((ti, j));
            if j > best.0 {
                best = (j, Some(ti));
            }
        }
        if best.0 > max_jaccard {
            max_jaccard = best.0;
        }
        if best.0 >= 0.5 {
            let mut above: Vec<(usize, f64)> = cands_scored.iter().copied().filter(|(_, j)| *j >= 0.5).collect();
            above.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            for (ti, j) in above {
                near_hits.push(json!({"eval_id": e.id, "eval_file": e.file, "kind": e.kind, "train_id": train_units[ti].id, "train_file": train_units[ti].file, "jaccard_5shingle": j, "eval_preview": e.text_preview, "train_preview": train_units[ti].text_preview}));
            }
        }
        let _ = ei;
    }
    // External = not an internal dev set; only external matches drive the verdict and the exclusion list.
    let external = |h: &Value| !is_internal(h["eval_file"].as_str().unwrap_or(""));
    let n_eval_state = eval_units.iter().filter(|u| u.kind == "state" && !is_internal(&u.file)).count().max(1) as f64;
    let state_exact = exact_hits.iter().filter(|h| h["kind"] == "state" && external(h)).count();
    let state_norm = norm_hits.iter().filter(|h| h["kind"] == "state" && external(h)).count();
    let state_near = near_hits.iter().filter(|h| h["kind"] == "state" && external(h)).count();
    let mut exclude_ids: Vec<String> = exact_hits
        .iter()
        .chain(norm_hits.iter())
        .chain(near_hits.iter())
        .filter(|h| h["kind"] == "state" && external(h))
        .filter_map(|h| h["train_id"].as_str().map(|s| s.to_string()))
        .collect();
    exclude_ids.sort();
    exclude_ids.dedup();
    if let Some(p) = &exclusions_out {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(p, serde_json::to_string_pretty(&json!({"generated_by": "sextant leakage", "rule": "training records whose state text exactly/normalized/near-duplicate matches an external evaluation state", "exclude_ids": exclude_ids})).unwrap() + "\n").expect("write exclusions");
        println!("wrote {} ({} ids)", p.display(), exclude_ids.len());
    }
    let instr_matches =
        exact_hits.iter().chain(norm_hits.iter()).filter(|h| h["kind"] == "instructions" && external(h)).count();
    // Verdict is driven by STATE text: instruction templates ("Is this message spam?")
    // can legitimately coincide and are reported separately for manual review.
    let verdict = state_exact == 0 && state_norm == 0 && (state_near as f64 / n_eval_state) < 0.005;
    let report = json!({
        "generated_by": "sextant leakage",
        "method": {
            "exact": "128-bit hash of the raw text (FxHash + FNV-1a)",
            "normalized": "same hash over lowercased, punctuation-stripped, whitespace-collapsed text",
            "near_duplicate": format!("MinHash ({N_PERM} permutations, {BANDS} bands x {ROWS} rows) over token {SHINGLE}-shingles, flagged at Jaccard >= 0.5"),
            "units": "state text and instruction+criteria text of every record, separately (min 20 normalized chars)"
        },
        "train_files": expand_jsonl(&train).iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "eval_files": expand_jsonl(&eval).iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "train_units": train_units.len(),
        "eval_units": eval_units.len(),
        "exact_matches": exact_hits.len(),
        "normalized_matches": norm_hits.len(),
        "near_duplicates": near_hits.len(),
        "state_exact_matches": state_exact,
        "state_normalized_matches": state_norm,
        "state_near_duplicates": state_near,
        "state_near_duplicate_rate": state_near as f64 / n_eval_state,
        "instruction_matches_for_review": instr_matches,
        "internal_eval_markers": internal_markers,
        "training_ids_to_exclude": exclude_ids.len(),
        "max_jaccard_observed": max_jaccard,
        "verdict": if verdict { "PASS" } else { "FAIL" },
        "verdict_rule": "PASS iff zero exact/normalized STATE matches against external eval sets (after exclusions) and the external state near-duplicate rate < 0.5%; instruction/criteria matches are listed for manual review; internal dev sets never count",
        "exact_match_pairs": exact_hits,
        "normalized_match_pairs": norm_hits,
        "near_duplicate_pairs": near_hits,
    });
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap() + "\n").expect("write report");
    println!("state: exact={} normalized={} near={} (rate {:.4}) | instruction matches for review={} | max_jaccard={:.3} verdict={}", state_exact, state_norm, state_near, state_near as f64 / n_eval_state, instr_matches, max_jaccard, report["verdict"]);
    println!("wrote {}", out.display());
    if verdict {
        0
    } else {
        1
    }
}
