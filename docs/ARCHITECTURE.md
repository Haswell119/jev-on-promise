# Architecture

Sextant turns *state + typed questions* into *typed, calibrated
probabilistic decisions* without any neural component. The design goal is a
compositional engine in which every layer is independently testable and
benchmarkable (`cargo bench -p sextant-core`, `sextant bench`).

```
HTTP (axum)                     crates/server
  │  GET /health · GET /v1/models · POST /v1/systemone[?explain]
  ▼
Schema validation               core::api::validate      (limits, cardinalities, depth, sizes)
  ▼
State normalization             core::text               (NFKC, tokenizer, stemmer, segmentation,
  │                                                       numbers, dates, scope annotation)
  ▼
Shared state index (once)       core::state::index       (fields+paths, segments, tokens+flags,
  │                                                       BM25 postings, tf-idf norms, char 4-grams,
  │                                                       phrase bigrams, numbers, dates, arrays,
  │                                                       key index, valence/intensity, hypernym &
  │                                                       domain maps, lowercase literal text)
  ▼  (read-only, shared by all questions; evaluated on a bounded rayon pool)
Question analysis               core::question           (instruction flattening, path refs,
  │                                                       reference data, criteria flattening with
  │                                                       field polarity, numeric ranges, family)
  ▼
Evidence retrieval              core::features::extract  (BM25 top-K segments per criterion,
  │                                                       literal search, focus fields)
  ▼
Symbolic resolvers              core::resolvers          (enum extraction, numeric ranges,
  │                                                       numeric/count/date comparison, reference
  │                                                       equality, boolean fields)
  ▼
Feature ensemble                core::features           (≈60 named features per criterion:
  │                                                       lexical, phrase, char-gram, lexical graph,
  │                                                       negation/modality agreement, cross-field,
  │                                                       sentiment/intensity, numeric, structural)
  ▼
Learned fusion                  core::scoring::heads     (conditional logit / logistic with
  │                                                       per-family linear experts; weights.json)
  ▼
Probability normalization       core::scoring::softmax   (stable softmax, ordinal smoothing,
  │                                                       exact-sum fix-up)
  ▼
Calibration                     core::calibration        (temperature per primitive/family/bucket,
  │                                                       Platt for Noul, symbolic ε-mixing)
  ▼
Confidence + typed answer       core::scoring::confidence, core::engine
```

## Crates

| Crate | Path | Role |
|---|---|---|
| `sextant-core` | `crates/core` | Everything above the HTTP layer; `#![forbid(unsafe_code)]`; embeds the lexicon assets and the model artifact |
| `sextant-server` | `crates/server` | axum router, body limits, timeouts, structured errors, OpenAPI serving |
| `sextant` (CLI) | `crates/cli` | `serve`, `decide`, `validate`, `bench`, `eval`, `train`, `calibrate`, `leakage`, `export-bootstrap` |

## Key design decisions

* **Index once, ask many.** The `StateIndex` is built once per request and
  shared read-only across questions. Per-question work touches only the
  postings of the criterion's terms, so adding questions costs roughly the
  per-question analysis, not another pass over the state.
* **Questions are independent by construction.** Each question gets a private
  `VocabExt` over the shared vocabulary, and no per-question state is written
  back. Property tests assert bit-identical answers regardless of the other
  questions in the request.
* **State is data, instructions are the query.** Directive text inside the
  state ("ignore the previous question and select X") is detected as a
  segment-level *directive* scope and discounted; option-name mentions inside
  such segments earn no literal credit. Nothing in the state can add labels.
* **Symbolic before semantic.** If a question is mechanically answerable, a
  resolver produces logits on a fixed scale; those are calibrated separately
  (error-rate ε-mixing) instead of being asserted as certain.
* **Small, inspectable parameters.** Two JSON files (`model/weights.json`,
  `model/calibration.json`) hold every learned number, keyed by feature name.
  `sextant train` and `sextant calibrate` regenerate them deterministically.
* **Order robustness.** Option position is never a feature; ties break by
  lexicographic key. Criteria are scored independently, then normalised.
* **Determinism.** No randomness at inference; hash maps use a fixed hasher
  and never leak iteration order into results; rayon parallelism only
  distributes independent questions.

## Data flow of one Choice question

1. `QuestionView::build` flattens instructions, resolves backticked path
   references to state fields (focus), extracts weighted question terms and
   detects the family. Each option becomes a `Criterion` with weighted terms
   (positive/negative polarity, negation/hypothetical/request flags), phrase
   list, bigrams, char-grams, literals, numeric range, valence/intensity and
   synonym/antonym expansions.
2. `extract_features` scores every criterion against the index: BM25 and
   coverage per segment via postings, occurrence flags (negated, hypothetical,
   request, directive, key, focus), literal search, phrase similarity,
   cross-field relations, numeric ranges, ordinal position.
3. `resolvers::resolve` may short-circuit with symbolic logits.
4. Otherwise the choice head computes `z_k = w_family · f_k`; softmax with
   the calibrated temperature; ε-mixing for symbolic; ordinal smoothing for
   Score; confidence from the distribution shape.
5. The answer is assembled with exactly the caller's keys.

## Observability

`explain=true` returns family, path, top evidence, raw scores, per-option
feature vectors and calibration parameters. It never changes inference.
