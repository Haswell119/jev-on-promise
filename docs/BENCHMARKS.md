# Benchmarks

Every number in this document is **measured** by the commands shown, on the
hardware recorded in `reports/environment.txt` and `reports/latest.md`.
Numbers are labelled by their source:

| Label | Meaning |
|---|---|
| **internal** | our own synthetic development split (`data/synthetic/dev.jsonl`), labels verifiable by construction |
| **external dev** | dev splits of public permissively-licensed datasets prepared by `scripts/prepare_data.py` (never JevBench sources) |
| **JevBench public** | the 231 public items of JevBench (Benchmark Heaven) scored by the harness itself at commit `75e6224e…` |
| **jev-bench** | the 22 test splits of `Praveenrajus/jev-bench` (22,773 records) scored by the Jevify harness at commit `2891025b…` |
| **published reference** | numbers published by third parties for Jev 1.13 (not measured by us) |

## How to reproduce

```bash
cargo build --release
scripts/train.sh                       # or keep the committed model/ artifact
./target/release/sextant bench --iters 200 --out reports/bench.json
./target/release/sextant eval data/synthetic/dev.jsonl --group-by family --out reports/internal_dev.json
./target/release/sextant eval data/processed/*/dev.jsonl --group-by source --out reports/external_dev.json
benchmarks/jevbench/fetch.sh && benchmarks/jevbench/run.sh          # → reports/jevbench/<sha>-<date>/
benchmarks/jev_bench_hf/fetch.sh && benchmarks/jev_bench_hf/run.sh  # → reports/jev_bench_hf/<sha>-<date>/
./target/release/sextant leakage --train reports/raw/training_used.jsonl \
    --eval benchmarks/jevbench/harness/datasets/public benchmarks/jev_bench_hf/data \
    --out reports/leakage.json
```

The harnesses are used unmodified; their scoring rules apply (argmax with
lexicographic tie-break, invalid or failed answers count as wrong, JevBench
ECE = 10-bin top-label, Jevify ECE = 15-bin).

## Evaluation protocol and contamination

* Training, calibration and criteria wording never use benchmark data or any
  benchmark source corpus (`docs/RESEARCH.md` §2, `data/README.md`).
* `reports/leakage.json` is the contamination scan of the data actually used
  for the shipped artifact against all evaluation sets.
* Two benchmark runs were made (`reports/experiments.md`, "Final-evaluation
  protocol"): run 1 revealed two general defects through aggregate
  diagnostics; run 2 is the reported result. No item-level inspection or
  tuning took place.

## Results

See `reports/latest.md` for the frozen tables (the same tables are summarized
in the README). Sections: internal dev by family, external dev by source,
JevBench public tiers / families / primitives / paraphrase consistency,
jev-bench per config and macro, calibration (NLL, Brier, ECE, reliability),
robustness (option-order stability, paraphrase stability, injection),
latency and throughput by scenario, memory.

## Comparison with the published Jev 1.13 reference

The published reference (JevBench v1.3, full tiers including private items)
is: easy 1.000, standard 0.990, judge 0.945, hard 0.741, schema validity
100 %, hard ECE 0.061; on the public subset an independent audited run
reports easy 48/48, standard 71/72, hard 81/111. On jev-bench: macro
accuracy 0.733, macro ECE 0.113, Brier 0.349.

Measured (frozen engine, run 2, `reports/latest.md`): JevBench public easy
39/48 = 81.2 %, standard 32/72 = 44.4 %, hard 44/111 = 39.6 %, schema
validity 100 % strict, easy ECE 0.059; jev-bench macro accuracy 0.415, macro
ECE 0.192, macro Brier 0.646 (0 errors on 22,773 records); typical-request
latency p50 5.9 ms / p95 6.7 ms. The per-tier gaps are −18.8 / −54.2 /
−33.4 points on the public subsets and −0.318 macro accuracy on jev-bench
(`docs/LIMITATIONS.md`). Sextant is a non-neural engine: it is expected to
trail a trained neural decision model on judgement-heavy tiers, while
matching or exceeding it on latency, determinism and cost, and on
mechanical questions (extraction, numbers, dates). Claims of "Jev-class"
accuracy are made only where the measurements support them.
