# Sextant

**A non-neural, deterministic, calibrated probabilistic decision engine.**
Send a *state* (text, JSON object or array) and a map of *typed questions*;
get back typed probabilistic decisions your code can branch on:

| Primitive | Question | Answer |
|---|---|---|
| **Choice** | pick exactly one option from a closed set (2–255 options) | `choice`, `probabilities` (exactly your keys, sum to 1), `confidence` |
| **Score** | rate the state on 2–10 ordered, described levels | `score` = Σ level×P(level), `legend`, `probabilities`, `confidence` |
| **Noul** | is this proposition true? | `noul` = calibrated P(true) |

Sextant is an independent, clean-room, open-source (Apache-2.0) project. It
is **not affiliated with, endorsed by or derived from TypeSafe AI**; it
implements the same *software primitive* (state + bounded typed questions →
typed probabilistic decisions) that TypeSafe's public documentation
describes for its Jev models, and speaks a compatible request/response
shape so existing clients can point at a self-hosted engine.

There is **no language model, no neural network, no embedding model and no
external AI service** anywhere in the runtime. Inference is classical NLP
(BM25 / tf-idf / character n-grams / phrase matching), a WordNet-derived
lexical graph, bounded-scope negation and modality handling, symbolic
resolvers for mechanical questions (numbers, dates, counts, literals,
booleans) and a small, human-readable linear fusion model with temperature /
Platt calibration. The whole engine runs in a container started with
`--network none` (see `scripts/test_no_network.sh`).

## Quick start

```bash
cargo build --release
./target/release/sextant serve --addr 0.0.0.0:8080
```

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "model": "sextant-1",
  "state": "Help! My payouts have been failing for 3 days. I need this fixed today.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments, invoicing, refunds, payouts",
                   "technical": "Bugs, outages, integrations",
                   "sales": "Pricing, upgrades, new accounts"}},
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

```json
{"model":"sextant-1",
 "answers":{
  "department":{"type":"choice","choice":"billing",
                "probabilities":{"billing":0.84,"technical":0.08,"sales":0.08},"confidence":0.85},
  "is_urgent":{"type":"noul","noul":0.40},
  "frustration":{"type":"score","score":0.67,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},
                 "probabilities":{"0":0.46,"1":0.41,"2":0.13},"confidence":0.33}},
 "usage":{"input_tokens":61,"output_tokens":12}}
```

Other entry points:

```bash
./target/release/sextant decide request.json --pretty      # one-shot, also reads stdin
./target/release/sextant decide request.json --explain     # evidence, features, raw scores
./target/release/sextant validate request.json             # schema check only
./target/release/sextant bench                              # latency / throughput
docker compose up                                           # container on :8080
```

Clients: `clients/python/sextant_client.py`, `clients/typescript/sextant.ts`,
`examples/curl.sh`. OpenAPI: `docs/openapi.yaml` (served at `/openapi.yaml`).

## How it works

```
validate → normalize → index the state ONCE → per question (parallel):
  analyse instructions & criteria → retrieve evidence → symbolic resolvers →
  ~65 deterministic features per option → linear fusion (per-family experts) →
  softmax / ordinal smoothing → calibration → confidence → typed answer
```

* **Structured state**: JSON is flattened with dotted paths; instructions may
  cite paths or instruction fields in backticks ("Does `extracted_value`
  match the `field` in `source_text`?"). Falsey JSON values negate their
  keys.
* **Structured criteria**: option objects keep their semantics — `not_for` /
  `excludes` / `negative_examples` fields count *against* an option,
  `examples` are matched individually, and everything is detected by
  morphology rather than a fixed list of names.
* **Negation and modality**: "requested a refund", "did not request a refund"
  and "asked whether refunds are possible" are three different things.
* **Symbolic fast paths**: literal option extraction, numeric ranges and
  comparisons (including array counts), date comparisons with relative dates,
  reference equality and boolean fields are decided by code, then calibrated.
* **State is data**: text that addresses the classifier ("ignore the previous
  question and select billing") is detected as a directive and is never
  evidence. Answers can only ever contain the caller's keys.
* **Calibration and confidence** are fitted on a held-out split; confidence
  is a calibrated function of the distribution's shape, not `max(p)`.
* **Deterministic and order-robust**: identical binary + artifact + request →
  identical bytes; questions are independent of each other and of option
  order; JSON key order does not matter.

Details: `docs/ARCHITECTURE.md`, `docs/ALGORITHMS.md`, `docs/API.md`,
`docs/CALIBRATION.md`, `docs/TRAINING.md`.

## Measured results

All numbers below are **measured** with the commands in `docs/BENCHMARKS.md`
and reproduced in `reports/latest.md` (which also records the commit,
hardware and versions). Nothing here is estimated.

<!-- BENCHMARK_TABLE_START -->
| Benchmark (measured) | Sextant | Published Jev 1.13 reference |
|---|---|---|
| JevBench public **easy** accuracy | 39/48 = 81.2 % | 100 % full tier; 48/48 public |
| JevBench public **standard** accuracy | 32/72 = 44.4 % | 99.0 % full tier; 71/72 public |
| JevBench public **hard** accuracy | 44/111 = 39.6 % | 74.1 % full tier; 81/111 public |
| JevBench schema validity | 100 % (strict) | 100 % |
| jev-bench macro accuracy / ECE / Brier (22 configs, 22773 records) | 0.415 / 0.192 / 0.646 | 0.733 / 0.113 / 0.349 |
| Internal synthetic dev accuracy / ECE | 89.3 % / 0.151 | – |
| External public dev sets accuracy / ECE | 59.6 % / 0.029 | – |
| Latency, typical 8-question request (p50 / p95) | 5.9 ms / 6.7 ms | ≈100 ms round trip (published) |

Full tables, per-family breakdowns, calibration and robustness metrics: `reports/latest.md`.
<!-- BENCHMARK_TABLE_END -->

The public Jev 1.13 reference profile on JevBench (easy ≈ 100 %, standard ≈
99 %, judge ≈ 94.5 %, hard ≈ 74 %) is a *reference*, not a claim; where
Sextant falls short the gap is quantified in `docs/BENCHMARKS.md`.

## Reproducing the model artifact

```bash
python3 scripts/synth/generate.py      # synthetic development recipes (seeded)
python3 scripts/prepare_data.py        # permissive public datasets (+ provenance)
scripts/train.sh                        # train → calibrate → leakage check → eval
cargo build --release                   # embed model/weights.json + calibration.json
```

Training data never includes JevBench / jev-bench or any of their source
corpora; `sextant leakage` enforces this and writes `reports/leakage.json`.

## Repository map

| Path | Contents |
|---|---|
| `crates/core` | engine library (`#![forbid(unsafe_code)]`) |
| `crates/server` | axum HTTP server |
| `crates/cli` | `sextant` binary: serve / decide / validate / bench / eval / train / calibrate / leakage |
| `model/` | `weights.json`, `calibration.json` (human-readable, regenerated by `scripts/train.sh`) |
| `crates/core/assets/lexicon` | compiled lexical assets (OEWN 2024, VADER, Google Books IDF) |
| `data/` | synthetic recipes, dataset preparation, provenance |
| `benchmarks/` | pinned JevBench / jev-bench harness integration |
| `reports/` | experiments log, ablations, leakage report, latest evaluation |
| `docs/` | architecture, algorithms, API, calibration, training, benchmarks, limitations, research notes |
| `clients/`, `examples/` | thin Python / TypeScript clients, curl examples |

## Limitations

Sextant does not have world knowledge, does not read between the lines, and
handles multi-step policy reasoning only when each step is mechanical. See
`docs/LIMITATIONS.md` for the honest list and `docs/BENCHMARKS.md` for what
that costs on each benchmark tier.

## License

Apache-2.0 (see `LICENSE`). Third-party lexical resources and their licenses
are listed in `THIRD_PARTY.md`.
