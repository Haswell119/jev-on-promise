//! Per-layer microbenchmarks: state indexing, question analysis, feature
//! extraction and full single-question evaluation at several cardinalities.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sextant_core::features::extract_features;
use sextant_core::lexicon::Resources;
use sextant_core::question::QuestionView;
use sextant_core::state::index::StateIndex;
use sextant_core::{Engine, Question};
use serde_json::json;

fn text(n: usize) -> String {
    let sents = [
        "The customer reported that the package arrived two days late and the box was slightly damaged.",
        "Our billing team confirmed the invoice was paid on March 3, 2026 for $1,240.00.",
        "The user asked whether refunds are possible for digital goods purchased last week.",
        "Tracking shows the shipment left the warehouse but has not been scanned since Tuesday.",
        "Support escalated the ticket because the login page returns an error after the password reset.",
    ];
    (0..n).map(|i| sents[i % sents.len()]).collect::<Vec<_>>().join(" ")
}

fn choice(k: usize) -> Question {
    let mut crit = serde_json::Map::new();
    let topics = ["billing", "shipping", "returns", "account", "technical", "sales", "legal", "privacy"];
    for i in 0..k {
        crit.insert(format!("{}_{i}", topics[i % topics.len()]), json!(format!("Questions about {} variant {i}: charges, delivery, access or product issues", topics[i % topics.len()])));
    }
    serde_json::from_value(json!({"type": "choice", "instructions": "Which team should handle this request?", "criteria": crit})).unwrap()
}

fn bench_layers(c: &mut Criterion) {
    let res = Resources::embedded();
    let state_v = json!(text(200)); // ≈ 4k tokens
    c.bench_function("index/state_4k_tokens", |b| b.iter(|| StateIndex::build(black_box(&state_v), &res)));
    let state = StateIndex::build(&state_v, &res);
    for k in [10usize, 64, 255] {
        let q = choice(k);
        c.bench_with_input(BenchmarkId::new("question/view", k), &q, |b, q| b.iter(|| QuestionView::build(black_box(q), &state, &res)));
        let view = QuestionView::build(&q, &state, &res);
        c.bench_with_input(BenchmarkId::new("features/extract", k), &view, |b, v| b.iter(|| extract_features(black_box(v), &state, &res)));
    }
    let engine = Engine::default_embedded();
    for k in [10usize, 255] {
        let q = choice(k);
        c.bench_with_input(BenchmarkId::new("engine/answer_one", k), &q, |b, q| b.iter(|| engine.answer_one(black_box(q), &state, false)));
    }
}

criterion_group!(benches, bench_layers);
criterion_main!(benches);
