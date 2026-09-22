# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog; the project follows semantic versioning.

## [0.1.0] - 2026-09-22

Initial public release.

* Choice / Score / Noul primitives over string, object or array state.
* Deterministic pipeline: validation → NFKC normalisation → typed tokenizer →
  shared state index (BM25 postings, tf-idf, char 4-grams, phrase bigrams,
  numbers, dates, negation/modality/request/exception/directive scopes) →
  question analysis → symbolic resolvers → 60+ feature ensemble →
  conditional-logit / logistic fusion with per-family experts → temperature /
  Platt calibration → confidence map.
* Symbolic fast paths: enum extraction, numeric ranges, numeric and count
  comparisons (incl. array lengths), date comparisons, reference equality,
  boolean fields.
* Open English WordNet 2024 lexical graph (synonyms, hypernyms, antonyms,
  topic domains), VADER valence, Google Books IDF prior.
* `sextant serve | decide | validate | bench | eval | train | calibrate |
  leakage | export-bootstrap` CLI; axum HTTP server with OpenAPI document.
* Reproducible training/calibration pipeline, synthetic recipe generator and
  permissive-dataset preparation script with provenance.
* Contamination scanner (exact / normalized / MinHash) and pinned JevBench and
  jev-bench harness integration.
