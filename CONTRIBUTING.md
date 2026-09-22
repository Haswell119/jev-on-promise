# Contributing

Thanks for your interest in Sextant. The project has one hard constraint that
every contribution must respect: **the runtime must remain non-neural and
offline**. No language models, embeddings, neural checkpoints, ONNX graphs or
hosted AI APIs — at inference *or* as hidden dependencies of training
artifacts. Classical deterministic and statistical methods only.

## Workflow

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release
./target/release/sextant validate examples/request_basic.json
./target/release/sextant decide examples/request_basic.json --pretty
```

Reproducing the model artifact (see `docs/TRAINING.md`):

```bash
python3 scripts/synth/generate.py            # synthetic recipes
python3 scripts/prepare_data.py              # permissive public datasets
./scripts/train.sh                           # train + calibrate → model/
./target/release/sextant eval data/synthetic/dev.jsonl --group-by family
```

## Rules for changes

* Every algorithmic change needs a row in `reports/experiments.md`
  (hypothesis, modification, before/after metrics on the internal dev set,
  latency and calibration impact, keep/revert).
* Never tune against JevBench / jev-bench. They are final evaluation only;
  `sextant leakage` must stay green.
* Never add a rule that fixes a single benchmark example. Fix failure
  *classes*.
* New lexical resources need a verified license, an entry in
  `THIRD_PARTY.md` and `data/provenance.jsonl`, and a reproducible build
  script.
* Feature additions must be ablated (`sextant train --drop-features`) and
  kept only when they pay for their latency.
* Keep the parameter artifact human-readable (`model/weights.json`,
  `model/calibration.json`).

## Code style

Rust 2021, `rustfmt` defaults (max width 120 in `rustfmt.toml`), clippy clean.
Deterministic everywhere: no randomness at inference, fixed seeds in training
and data generation, no HashMap iteration order leaking into results.
