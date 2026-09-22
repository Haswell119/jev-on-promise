# Latest evaluation report

Frozen engine commit `52940a8b976d` · weights `trained-1790088256` · calibration `calibrated-1790088258` · generated 2026-09-22T14:45:53Z

Every number below is **measured** (labels: internal / external dev / JevBench public / jev-bench / published reference). Nothing is estimated.

## Environment

| | |
|---|---|
| cpu | Intel(R) Xeon(R) Processor @ 2.80GHz |
| cores | 4 |
| ram | 15 GiB |
| kernel | 6.18.44-fc-v37 |
| rustc | rustc 1.94.1 (e408947bf 2026-03-25) |
| release_profile | opt-level=3 lto=thin codegen-units=16 |
| JevBench harness | fstandhartinger/jevbench @ 75e6224ed8103bbc3485ca74820a2eaf7ce8abe0 (MIT), public files sha256 231df3c2… / 5c2414ed… / 89e9e6be… |
| jev-bench | Praveenrajus/jev-bench @ 002ad22de8db2df5e0eb898b3da8072dbd4af4de; harness uspraveen/Jevify @ 2891025b8a4520d0eb639f2cd1b3fddaf4b922b3 (Apache-2.0) |

## Leakage check (`reports/leakage.json`)

Verdict **PASS** — training/calibration units 47991, evaluation units 46900; external state matches: exact 0, normalized 0, near-duplicate 0 (rate 0.0000); instruction matches for review 0; max 5-shingle Jaccard observed 0.846.

## JevBench public tiers (JevBench harness scoring, run 2 = final)

| tier | public / frozen | attempted | schema valid | strict | correct | accuracy | chance | intelligence | Brier | ECE | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **easy** | 48 / 72 | 48 | 100.0 % | 100.0 % | 39 | 81.2 % | 28.4 % | 73.8 | 0.231 | 0.059 | 1.3 ms | 2.1 ms |
| **standard** (original) | 72 / 96 | 72 | 100.0 % | 100.0 % | 32 | 44.4 % | 31.7 % | 18.7 | 0.871 | 0.346 | 1.4 ms | 1.9 ms |
| **hard** | 111 / 220 | 111 | 100.0 % | 100.0 % | 44 | 39.6 % | 33.6 % | 9.0 | 0.759 | 0.210 | 3.1 ms | 7.3 ms |
| judge | 0 / 146 | not public | – | – | – | – | 29.2 % | – | – | – | – | – |
| all public | 231 | 231 | 100.0 % | 100.0 % | 115 | 49.8 % | 31.8 % | – | 0.684 | 0.204 | 1.9 ms | 6.9 ms |

### Hard tier by family

| family | n | correct | accuracy | chance | Brier | ECE |
|---|---|---|---|---|---|---|
| `adversarial` | 6 | 2 | 33.3 % | 33.3 % | 0.686 | 0.283 |
| `ambiguous` | 7 | 2 | 28.6 % | 32.1 % | 1.064 | 0.422 |
| `judge_hard` | 17 | 11 | 64.7 % | 50.0 % | 0.612 | 0.356 |
| `long_policy` | 19 | 8 | 42.1 % | 30.1 % | 0.768 | 0.275 |
| `multi_hop` | 18 | 4 | 22.2 % | 25.2 % | 0.844 | 0.355 |
| `probability` | 10 | 1 | 10.0 % | 38.3 % | 0.941 | 0.624 |
| `routing_hard` | 5 | 5 | 100.0 % | 20.0 % | 0.238 | 0.386 |
| `temporal_numeric` | 15 | 4 | 26.7 % | 30.9 % | 0.736 | 0.311 |
| `tradeoff` | 6 | 3 | 50.0 % | 30.8 % | 1.077 | 0.764 |
| `trap` | 8 | 4 | 50.0 % | 37.5 % | 0.545 | 0.373 |

### By question type

| tier | type | n | correct | accuracy | Brier | ECE | ordinal MAE |
|---|---|---|---|---|---|---|---|
| easy | `choice` | 36 | 33 | 91.7 % | 0.112 | 0.081 | – |
| easy | `noul` | 12 | 6 | 50.0 % | 0.589 | 0.130 | – |
| standard | `choice` | 36 | 16 | 44.4 % | 0.870 | 0.335 | – |
| standard | `noul` | 24 | 12 | 50.0 % | 0.940 | 0.473 | – |
| standard | `score` | 12 | 4 | 33.3 % | 0.731 | 0.147 | 1.05 |
| hard | `choice` | 67 | 20 | 29.9 % | 0.808 | 0.207 | – |
| hard | `noul` | 38 | 22 | 57.9 % | 0.682 | 0.306 | – |
| hard | `score` | 6 | 2 | 33.3 % | 0.690 | 0.067 | 0.80 |

### Paraphrase pairs

| tier | pairs | both valid | same answer | agreement | both correct | both-correct rate (all pairs) |
|---|---|---|---|---|---|---|
| easy | 0 | 0 | 0 | – | 0 | – |
| standard | 36 | 36 | 30 | 83.3 % | 13 | 36.1 % |
| hard | 0 | 0 | 0 | – | 0 | – |

### Comparison with the published Jev 1.13 reference

| tier | Sextant (public subset, measured) | Jev 1.13 published (full tier) | Jev 1.13 public subset (independent audited run) | gap on public subset |
|---|---|---|---|---|
| easy | 39/48 = 0.812 | 1.000 | 48/48 = 1.000 | -0.188 |
| standard | 32/72 = 0.444 | 0.990 | 71/72 = 0.986 | -0.542 |
| hard | 44/111 = 0.396 | 0.741 | 81/111 = 0.730 | -0.333 |
| judge | not public (0/146) | 0.945 | – | – |

Reference targets set for this project (Stage A: easy ≥ 0.95, standard ≥ 0.80; Stage B: easy ≥ 0.98, standard ≥ 0.90, hard ≥ 0.50; Stage C: easy ≥ 0.99, standard ≥ 0.97, hard ≥ 0.70) are **not reached** on the standard and hard tiers; see docs/LIMITATIONS.md for the failure classes.

## jev-bench (22 configs, Jevify harness scoring, run 2 = final)

| configs | records scored | errors | macro acc | macro ECE | macro Brier | sel@90 | choice acc | score acc | noul acc | TVD→human | latency p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | 22773 | 0 | 0.415 | 0.192 | 0.646 | 0.425 | 0.329 | 0.302 | 0.623 | 0.486 | 6.4 ms | 12.3 ms |

Published Jev 1.13.0 reference on the same 22 configs: macro accuracy 0.733, macro ECE 0.113, macro Brier 0.349, TVD→human 0.432.

### Per config

| config | prim | K | scored / test | errors | acc | ECE | Brier | NLL | sel@90 | MAE | AUROC | TVD→human | p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | variable | 1000 / 1000 | 0 | 0.215 | 0.207 | 0.903 | 1.77 | 0.216 | – | – | – | 6.3 ms |
| `banking77` | choice | 77 | 1000 / 1000 | 0 | 0.465 | 0.376 | 0.853 | 3.84 | 0.494 | – | – | – | 8.1 ms |
| `boolq` | noul | variable | 1000 / 1000 | 0 | 0.528 | 0.121 | 0.280 | 0.77 | 0.517 | – | 0.536 | – | 6.2 ms |
| `chaosnli` | choice | 3 | 1599 / 1599 | 0 | 0.452 | 0.056 | 0.628 | 1.04 | 0.461 | – | – | 0.317 | 6.2 ms |
| `civil_comments` | noul | variable | 2000 / 2000 | 0 | 0.922 | 0.241 | 0.128 | 0.44 | 0.935 | – | 0.650 | 0.273 | 6.1 ms |
| `clinc150` | choice | 151 | 1000 / 1000 | 0 | 0.417 | 0.316 | 0.882 | 5.04 | 0.456 | – | – | – | 12.0 ms |
| `fever_evidence` | noul | variable | 1000 / 1000 | 0 | 0.617 | 0.056 | 0.236 | 0.67 | 0.632 | – | 0.668 | – | 6.1 ms |
| `go_emotions` | choice | 28 | 1000 / 1000 | 0 | 0.093 | 0.073 | 0.938 | 3.23 | 0.102 | – | – | 0.861 | 6.5 ms |
| `helpsteer2_helpfulness` | score | 5 | 1000 / 1000 | 0 | 0.291 | 0.121 | 0.797 | 1.58 | 0.302 | 1.15 | – | – | 6.9 ms |
| `helpsteer2_verbosity` | score | 5 | 1000 / 1000 | 0 | 0.480 | 0.219 | 0.751 | 1.48 | 0.498 | 0.59 | – | – | 7.6 ms |
| `ledgar` | choice | 100 | 1000 / 1000 | 0 | 0.318 | 0.236 | 0.912 | 3.23 | 0.344 | – | – | – | 11.6 ms |
| `massive` | choice | 60 | 1000 / 1000 | 0 | 0.358 | 0.109 | 0.779 | 2.78 | 0.388 | – | – | – | 7.3 ms |
| `measuring_hate_speech` | score | 3 | 1000 / 1000 | 0 | 0.423 | 0.131 | 0.666 | 1.13 | 0.433 | 0.87 | – | 0.495 | 6.3 ms |
| `mmlu` | choice | 4 | 1000 / 1000 | 0 | 0.277 | 0.194 | 0.863 | 1.67 | 0.281 | – | – | – | 6.3 ms |
| `mnli` | choice | 3 | 1000 / 1000 | 0 | 0.364 | 0.152 | 0.708 | 1.18 | 0.370 | – | – | – | 6.3 ms |
| `paws` | noul | variable | 1000 / 1000 | 0 | 0.432 | 0.565 | 0.565 | 3.61 | 0.433 | – | 0.534 | – | 6.1 ms |
| `sms_spam` | noul | variable | 800 / 800 | 0 | 0.855 | 0.343 | 0.236 | 0.67 | 0.856 | – | 0.493 | – | 5.9 ms |
| `sst5` | score | 5 | 1000 / 1000 | 0 | 0.170 | 0.211 | 0.835 | 1.65 | 0.170 | 1.07 | – | – | 6.2 ms |
| `strategyqa_closed` | noul | variable | 687 / 687 | 0 | 0.515 | 0.071 | 0.255 | 0.70 | 0.531 | – | 0.500 | – | 6.5 ms |
| `strategyqa_grounded` | noul | variable | 687 / 687 | 0 | 0.489 | 0.240 | 0.328 | 0.89 | 0.485 | – | 0.430 | – | 6.1 ms |
| `stsb` | score | 6 | 1000 / 1000 | 0 | 0.161 | 0.075 | 0.844 | 1.82 | 0.158 | 1.38 | – | – | 6.2 ms |
| `yelp5` | score | 5 | 1000 / 1000 | 0 | 0.286 | 0.111 | 0.814 | 1.62 | 0.282 | 1.08 | – | – | 6.5 ms |

## Internal development set (synthetic, by family)

| family | n | accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| adequacy | 8 | 1.000 | 0.132 | 0.037 | 0.121 |
| compatibility | 19 | 1.000 | 0.060 | 0.008 | 0.058 |
| enum_extraction | 38 | 1.000 | 0.041 | 0.002 | 0.040 |
| factual_yes_no | 150 | 0.927 | 0.385 | 0.254 | 0.215 |
| numeric | 169 | 0.929 | 0.287 | 0.149 | 0.118 |
| policy | 46 | 0.609 | 0.821 | 0.481 | 0.162 |
| relevance | 4 | 1.000 | 0.652 | 0.459 | 0.479 |
| request_detection | 33 | 0.848 | 0.479 | 0.256 | 0.162 |
| routing | 152 | 0.908 | 0.530 | 0.237 | 0.235 |
| sentiment | 24 | 0.958 | 0.649 | 0.370 | 0.421 |
| severity | 20 | 0.400 | 1.365 | 0.723 | 0.171 |
| temporal | 19 | 1.000 | 0.012 | 0.000 | 0.012 |
| **all** | 682 | 0.893 | 0.425 | 0.228 | 0.151 |
| kind: choice | 239 | 0.870 | 0.557 | 0.267 | 0.193 |
| kind: score | 94 | 0.904 | 0.556 | 0.257 | 0.269 |
| kind: noul | 349 | 0.905 | 0.300 | 0.194 | 0.122 |

Variant stability (paraphrase / option permutation / JSON wrapping / distractors of the same scenario): same-answer rate 0.951 over 102 groups, mean top-probability distance 0.067.

## External development sets (public permissive datasets, never trained on their dev splits; HWU64 never trained on at all)

| source | n | accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| BEE-spoke-data/consumer-finance-complaints | 400 | 0.562 | 1.499 | 0.614 | 0.076 |
| TimSchopf/medical_abstracts | 400 | 0.490 | 1.295 | 0.646 | 0.043 |
| allenai/prosocial-dialog | 400 | 0.255 | 1.586 | 0.791 | 0.016 |
| bitext/Bitext-customer-support-llm-chatbot-training-dataset | 800 | 0.664 | 1.106 | 0.445 | 0.077 |
| cardiffnlp/tweet_eval | 400 | 0.463 | 1.012 | 0.609 | 0.090 |
| fancyzhx/amazon_polarity | 800 | 0.774 | 0.461 | 0.299 | 0.038 |
| fancyzhx/dbpedia_14 | 400 | 0.718 | 0.910 | 0.374 | 0.067 |
| https://github.com/sonos/nlu-benchmark | 400 | 0.800 | 0.533 | 0.266 | 0.049 |
| https://github.com/xliuhw/NLU-Evaluation-Data | 800 | 0.611 | 1.565 | 0.528 | 0.111 |
| qiaojin/PubMedQA | 178 | 0.461 | 0.760 | 0.559 | 0.128 |
| rajpurkar/squad_v2 | 400 | 0.545 | 0.671 | 0.478 | 0.062 |
| stanfordnlp/snli | 800 | 0.536 | 0.833 | 0.521 | 0.049 |
| **all** | 6178 | 0.596 | 1.021 | 0.493 | 0.029 |
| kind: choice | 4400 | 0.606 | 1.107 | 0.495 | 0.035 |
| kind: score | 400 | 0.255 | 1.586 | 0.791 | 0.016 |
| kind: noul | 1378 | 0.664 | 0.583 | 0.400 | 0.023 |

## Latency and throughput (release build, embedded model, after warm-up, single process)

Threads: 4 · `sextant bench --iters 200` · Intel(R) Xeon(R) Processor @ 2.80GHz

| scenario | questions | state bytes | p50 ms | p95 ms | p99 ms | req/s |
|---|---|---|---|---|---|---|
| typical_8q_4k_tokens | 8 | 20635 | 5.85 | 6.65 | 8.89 | 167.0 |
| json_state_3q | 3 | 1033 | 1.21 | 1.58 | 1.64 | 784.7 |
| large_16q_16k_tokens_k64 | 16 | 81886 | 32.09 | 35.86 | 42.54 | 30.3 |
| choice_k2 | 1 | 3461 | 1.43 | 1.58 | 1.77 | 689.9 |
| choice_k5 | 1 | 3461 | 1.68 | 1.94 | 2.06 | 586.7 |
| choice_k10 | 1 | 3461 | 1.99 | 2.15 | 2.19 | 498.7 |
| choice_k32 | 1 | 3461 | 3.15 | 3.37 | 3.48 | 314.2 |
| choice_k64 | 1 | 3461 | 5.21 | 8.66 | 9.24 | 180.8 |
| choice_k128 | 1 | 3461 | 8.84 | 9.61 | 10.52 | 111.6 |
| choice_k255 | 1 | 3461 | 15.65 | 17.25 | 20.94 | 63.0 |
| questions_1 | 1 | 10304 | 3.07 | 3.27 | 3.35 | 324.4 |
| questions_8 | 8 | 10304 | 3.96 | 6.15 | 6.72 | 233.8 |
| questions_32 | 32 | 10304 | 5.78 | 8.64 | 15.12 | 160.0 |
| questions_128 | 128 | 10304 | 12.26 | 15.34 | 15.92 | 78.6 |
| questions_512 | 512 | 10304 | 39.51 | 45.26 | 52.82 | 24.6 |

Typical request (8 questions, ≈4k-token state, ≤10 criteria): p50 5.9 ms / p95 6.7 ms — target p50 ≤ 25 ms, p95 ≤ 75 ms: **met**. Published Jev round-trip: ≈100 ms typical.

