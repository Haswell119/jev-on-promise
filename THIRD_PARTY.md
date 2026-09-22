# Third-party resources and notices

Sextant is an independent project. It is not affiliated with, endorsed by, or
derived from TypeSafe AI or its Jev models. No TypeSafe API output was used at
any stage of development.

## Lexical data compiled into the binary (`crates/core/assets/lexicon/`)

| Resource | Version | License | Files derived | Notes |
|---|---|---|---|---|
| Open English WordNet (WNDB distribution) | 2024 edition (2024-11-01) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `wn_lemmas.tsv`, `wn_synsets.tsv`, `wn_antonyms.tsv` | Derived from the Open English WordNet, itself derived from Princeton WordNet. Attribution required to both the Open English Wordnet team and Princeton University (see notice below). Download: https://en-word.net/downloads/english-wordnet-2024.zip |
| VADER sentiment lexicon | master (2024) | MIT (Copyright (c) 2016 C.J. Hutto) | `sentiment.tsv` | Mean valence per token, rescaled to [-1, 1] at load. Source: https://github.com/cjhutto/vaderSentiment |
| Google Books Ngram derived English 1-gram frequency list (orgtre/google-books-ngram-frequency) | Ngram v3 (20200217) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | `word_freq.tsv` | 10,000 most frequent words; used as a background IDF prior. Upstream corpus: Google Books Ngram Corpus (CC BY 3.0). |

Build script: `scripts/build_lexicon.py` (deterministic). Provenance with
SHA-256 of inputs and outputs: `data/provenance/lexicon.json`.

### Princeton WordNet / Open English WordNet notice

> WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved.
> THIS SOFTWARE AND DATABASE IS PROVIDED "AS IS" AND PRINCETON UNIVERSITY
> MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF
> EXAMPLE, BUT NOT LIMITATION, PRINCETON UNIVERSITY MAKES NO REPRESENTATIONS
> OR WARRANTIES OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR
> THAT THE USE OF THE LICENSED SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT
> INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS.
>
> Open English WordNet is provided by the Open English Wordnet team under the
> Creative Commons Attribution 4.0 International License (CC-BY 4.0).

### VADER notice

> The MIT License (MIT). Copyright (c) 2016 C.J. Hutto. Permission is hereby
> granted, free of charge, to any person obtaining a copy of this software and
> associated documentation files (the "Software"), to deal in the Software
> without restriction […] THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
> OF ANY KIND.

## Training / development datasets (downloaded, not redistributed)

See `data/README.md` and `data/provenance.jsonl` for every dataset used by
`scripts/prepare_data.py`, with its license and revision. None of them are
sources of the JevBench or jev-bench evaluation sets (see
`docs/RESEARCH.md`, section 2, and `reports/leakage.json`).

## Evaluation harnesses (used unmodified, fetched at pinned commits)

| Harness | Commit | License |
|---|---|---|
| JevBench (Benchmark Heaven), `fstandhartinger/jevbench` | `75e6224ed8103bbc3485ca74820a2eaf7ce8abe0` | MIT |
| Jevify, `uspraveen/Jevify` (harness for `Praveenrajus/jev-bench`) | `2891025b8a4520d0eb639f2cd1b3fddaf4b922b3` | Apache-2.0 (rows carry upstream licenses) |

## Rust crates

All runtime dependencies are permissively licensed (MIT and/or Apache-2.0,
BSD-3-Clause for `rust-stemmers`, Unlicense/MIT for `aho-corasick`). `cargo
deny check licenses` enforces the allow-list in `deny.toml`. Main crates:
`axum`, `tokio`, `tower-http`, `serde`, `serde_json`, `indexmap`, `regex`,
`aho-corasick`, `unicode-normalization`, `rust-stemmers`, `rayon`,
`rustc-hash`, `smallvec`, `clap`, `tracing`; dev: `proptest`, `criterion`.

## Public documentation consulted (no code or data copied)

TypeSafe's public documentation (docs.typesafe.ai) was read to understand
the publicly documented System One request/response shape and product
philosophy. Sextant's API mirrors that public wire shape so that existing
clients can be pointed at a self-hosted engine; names, code and model are
original.
