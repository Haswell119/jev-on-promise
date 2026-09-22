//! Latency / throughput benchmark on synthetic requests covering the
//! typical request, option cardinality (2..255), question counts (1..512)
//! and long states. Reports p50/p95/p99 and requests per second.

use super::common::build_engine;
use serde_json::{json, Value};
use sextant_core::SystemOneRequest;
use std::path::PathBuf;
use std::time::Instant;

fn lorem(n_sentences: usize, seed: u64) -> String {
    const S: &[&str] = &[
        "The customer reported that the package arrived two days late and the box was slightly damaged.",
        "Our billing team confirmed the invoice was paid on March 3, 2026 for $1,240.00.",
        "The user asked whether refunds are possible for digital goods purchased last week.",
        "Tracking shows the shipment left the warehouse but has not been scanned since Tuesday.",
        "Support agent escalated the ticket because the login page returns an error after the password reset.",
        "The policy states that returns are accepted within 30 days except for perishable items.",
        "No refund was issued yet, but a replacement was promised by the end of the month.",
        "The account was flagged for unusual activity and the API key was rotated.",
        "Pricing questions about the enterprise tier should go to the sales team.",
        "The customer is very frustrated and threatened to cancel the subscription immediately.",
    ];
    let mut out = String::new();
    let mut x = seed;
    for _ in 0..n_sentences {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push_str(S[((x >> 33) % S.len() as u64) as usize]);
        out.push(' ');
    }
    out
}

fn choice_q(k: usize) -> Value {
    let mut criteria = serde_json::Map::new();
    let topics = ["billing", "shipping", "returns", "account", "technical", "sales", "legal", "privacy", "fraud", "feedback"];
    for i in 0..k {
        let t = topics[i % topics.len()];
        criteria.insert(format!("{t}_{i}"), json!(format!("Questions about {t}, variant {i}: charges, delivery, access or product issues")));
    }
    json!({"type": "choice", "instructions": "Which team should handle this request?", "criteria": criteria})
}

fn typical_request(n_questions: usize, sentences: usize, k: usize) -> SystemOneRequest {
    let mut questions = serde_json::Map::new();
    for i in 0..n_questions {
        let q = match i % 3 {
            0 => choice_q(k),
            1 => json!({"type": "noul", "instructions": "Did the customer request a refund?", "criteria": {"true": "The customer explicitly asks for money back", "false": "No refund request"}}),
            _ => json!({"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Slightly annoyed", "Frustrated", "Very angry"]}),
        };
        questions.insert(format!("q{i}"), q);
    }
    serde_json::from_value(json!({"model": "sextant-1", "state": lorem(sentences, 7), "questions": questions})).unwrap()
}

fn json_request() -> SystemOneRequest {
    serde_json::from_value(json!({
        "model": "sextant-1",
        "state": {"ticket": {"id": "T-1042", "messages": [{"from": "customer", "text": lorem(6, 3)}, {"from": "agent", "text": lorem(4, 5)}], "amount": 1240.0, "paid": false, "tags": ["late", "damaged"]}},
        "questions": {
            "dept": choice_q(5),
            "paid": {"type": "noul", "instructions": "Has the invoice been paid?"},
            "amount": {"type": "score", "instructions": "How large is `ticket.amount`?", "criteria": ["Under $100", "$100 to $1,000", "$1,000 to $10,000", "Over $10,000"]}
        }
    }))
    .unwrap()
}

struct Scenario {
    name: String,
    req: SystemOneRequest,
}

fn scenarios() -> Vec<Scenario> {
    let mut v = vec![
        Scenario { name: "typical_8q_4k_tokens".into(), req: typical_request(8, 240, 8) },
        Scenario { name: "json_state_3q".into(), req: json_request() },
        Scenario { name: "large_16q_16k_tokens_k64".into(), req: typical_request(16, 960, 64) },
    ];
    for k in [2usize, 5, 10, 32, 64, 128, 255] {
        v.push(Scenario { name: format!("choice_k{k}"), req: typical_request(1, 40, k) });
    }
    for n in [1usize, 8, 32, 128, 512] {
        v.push(Scenario { name: format!("questions_{n}"), req: typical_request(n, 120, 6) });
    }
    v
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn run(model_dir: Option<PathBuf>, threads: usize, iters: usize, out: Option<PathBuf>, filter: Option<String>) -> i32 {
    let engine = match build_engine(model_dir, threads) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let mut results = Vec::new();
    println!("{:<32} {:>8} {:>9} {:>9} {:>9} {:>9}", "scenario", "iters", "p50_ms", "p95_ms", "p99_ms", "req/s");
    for sc in scenarios() {
        if let Some(f) = &filter {
            if !sc.name.contains(f.as_str()) {
                continue;
            }
        }
        // warmup
        for _ in 0..3 {
            let _ = engine.evaluate(&sc.req);
        }
        let n = if sc.name.starts_with("questions_512") || sc.name.starts_with("large") { iters.min(40).max(5) } else { iters.max(5) };
        let mut lat = Vec::with_capacity(n);
        let t0 = Instant::now();
        for _ in 0..n {
            let s = Instant::now();
            let r = engine.evaluate(&sc.req);
            lat.push(s.elapsed().as_secs_f64() * 1000.0);
            assert!(r.is_ok());
        }
        let total = t0.elapsed().as_secs_f64();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&lat, 0.5);
        let p95 = percentile(&lat, 0.95);
        let p99 = percentile(&lat, 0.99);
        let rps = n as f64 / total;
        println!("{:<32} {:>8} {:>9.2} {:>9.2} {:>9.2} {:>9.1}", sc.name, n, p50, p95, p99, rps);
        results.push(json!({"scenario": sc.name, "iters": n, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "mean_ms": lat.iter().sum::<f64>() / n as f64, "requests_per_second": rps, "questions": sc.req.questions.len(), "state_bytes": serde_json::to_string(&sc.req.state).unwrap().len()}));
    }
    let report = json!({
        "engine_version": sextant_core::VERSION,
        "model": engine.model.source,
        "threads": if threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0) } else { threads },
        "results": results,
    });
    if let Some(p) = out {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&p, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
        println!("wrote {}", p.display());
    }
    0
}
