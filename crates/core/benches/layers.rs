use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sextant_core::lexicon::Resources;
use sextant_core::state::index::StateIndex;

fn bench_index(c: &mut Criterion) {
    let res = Resources::embedded();
    let text = "The customer wrote: my order arrived late and the box was damaged. I did not request a refund yet but I would like a replacement. ".repeat(20);
    let state = serde_json::json!(text);
    c.bench_function("state_index_2k_tokens", |b| b.iter(|| StateIndex::build(black_box(&state), &res)));
}

criterion_group!(benches, bench_index);
criterion_main!(benches);
