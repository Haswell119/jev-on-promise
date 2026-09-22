# jev-bench test splits — Sextant `sextant-1`

|  |  |
|---|---|
| Run | `c64b39be-20260922` finished 2026-09-22T14:37:04+00:00 |
| Sextant | commit `c64b39be2f20f189c21b71301cb7b0f0d88e2732`, `sextant 0.1.0` |
| Dataset | Praveenrajus/jev-bench @ `002ad22de8db2df5e0eb898b3da8072dbd4af4de` (test splits only) |
| Harness | uspraveen/Jevify @ `2891025b8a4520d0eb639f2cd1b3fddaf4b922b3` (Apache-2.0), `jevify-run api` + `jevify-run report` |
| Endpoint | `http://127.0.0.1:48755`, concurrency 4 |
| Hardware | Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores, 15.7 GiB RAM, rustc 1.94.1 (e408947bf 2026-03-25) |

## Leaderboard-style aggregates

| configs | records scored | errors | macro acc | macro ECE | macro Brier | sel@90 | choice acc | score acc | noul acc | TVD→human | latency p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | 22773 | 0 | 0.399 | 0.189 | 0.648 | 0.408 | 0.328 | 0.303 | 0.571 | 0.519 | 6.4 ms | 12.5 ms |

## Per config

| config | prim | K | scored / test | errors | acc | ECE | Brier | NLL | sel@90 | MAE | AUROC | TVD→human | p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | variable | 1000 / 1000 | 0 | 0.211 | 0.208 | 0.903 | 1.77 | 0.212 | – | – | – | 6.5 ms |
| `banking77` | choice | 77 | 1000 / 1000 | 0 | 0.465 | 0.376 | 0.853 | 3.84 | 0.494 | – | – | – | 8.9 ms |
| `boolq` | noul | variable | 1000 / 1000 | 0 | 0.449 | 0.168 | 0.294 | 0.80 | 0.439 | – | 0.539 | – | 6.3 ms |
| `chaosnli` | choice | 3 | 1599 / 1599 | 0 | 0.452 | 0.057 | 0.628 | 1.04 | 0.461 | – | – | 0.317 | 6.1 ms |
| `civil_comments` | noul | variable | 2000 / 2000 | 0 | 0.652 | 0.111 | 0.230 | 0.65 | 0.667 | – | 0.625 | 0.401 | 6.0 ms |
| `clinc150` | choice | 151 | 1000 / 1000 | 0 | 0.418 | 0.315 | 0.882 | 5.04 | 0.457 | – | – | – | 13.8 ms |
| `fever_evidence` | noul | variable | 1000 / 1000 | 0 | 0.608 | 0.062 | 0.231 | 0.65 | 0.617 | – | 0.666 | – | 6.3 ms |
| `go_emotions` | choice | 28 | 1000 / 1000 | 0 | 0.090 | 0.076 | 0.938 | 3.23 | 0.100 | – | – | 0.861 | 6.7 ms |
| `helpsteer2_helpfulness` | score | 5 | 1000 / 1000 | 0 | 0.291 | 0.133 | 0.797 | 1.58 | 0.302 | 1.13 | – | – | 8.0 ms |
| `helpsteer2_verbosity` | score | 5 | 1000 / 1000 | 0 | 0.481 | 0.208 | 0.739 | 1.45 | 0.500 | 0.60 | – | – | 8.3 ms |
| `ledgar` | choice | 100 | 1000 / 1000 | 0 | 0.318 | 0.236 | 0.912 | 3.23 | 0.344 | – | – | – | 9.2 ms |
| `massive` | choice | 60 | 1000 / 1000 | 0 | 0.357 | 0.108 | 0.779 | 2.78 | 0.388 | – | – | – | 7.5 ms |
| `measuring_hate_speech` | score | 3 | 1000 / 1000 | 0 | 0.423 | 0.162 | 0.681 | 1.18 | 0.430 | 0.86 | – | 0.497 | 6.3 ms |
| `mmlu` | choice | 4 | 1000 / 1000 | 0 | 0.279 | 0.191 | 0.863 | 1.67 | 0.283 | – | – | – | 6.3 ms |
| `mnli` | choice | 3 | 1000 / 1000 | 0 | 0.364 | 0.152 | 0.708 | 1.18 | 0.370 | – | – | – | 6.1 ms |
| `paws` | noul | variable | 1000 / 1000 | 0 | 0.432 | 0.563 | 0.563 | 3.18 | 0.434 | – | 0.532 | – | 6.2 ms |
| `sms_spam` | noul | variable | 800 / 800 | 0 | 0.861 | 0.285 | 0.200 | 0.59 | 0.861 | – | 0.494 | – | 5.9 ms |
| `sst5` | score | 5 | 1000 / 1000 | 0 | 0.170 | 0.241 | 0.847 | 1.67 | 0.169 | 1.06 | – | – | 6.1 ms |
| `strategyqa_closed` | noul | variable | 687 / 687 | 0 | 0.515 | 0.156 | 0.274 | 0.74 | 0.531 | – | 0.500 | – | 5.8 ms |
| `strategyqa_grounded` | noul | variable | 687 / 687 | 0 | 0.480 | 0.141 | 0.281 | 0.76 | 0.482 | – | 0.442 | – | 5.9 ms |
| `stsb` | score | 6 | 1000 / 1000 | 0 | 0.167 | 0.081 | 0.847 | 1.83 | 0.157 | 1.39 | – | – | 6.1 ms |
| `yelp5` | score | 5 | 1000 / 1000 | 0 | 0.288 | 0.123 | 0.811 | 1.60 | 0.283 | 1.06 | – | – | 6.7 ms |

## Definitions

- Every per-config number is the harness's own (jevify.bench.metrics at the pinned commit): accuracy = argmax (choice/score) or P(yes) >= 0.5 (noul); ECE = top-label, 15 equal-width bins; Brier = sum over options of (p - y)^2 for choice/score and (p_yes - y)^2 for noul; NLL = -log p(gold); sel@90 = accuracy on the 90 % most confident records (model `confidence` for choice/score, |2p-1| for noul); AURC = area under the risk-coverage curve; RPS/MAE = ordinal scores for score configs; AUROC for noul; TVD→human = total variation distance to the human label distribution on the four calibration-gold configs.
- macro rows are unweighted means over the configs that were scored; per-primitive rows average the configs of that primitive. With --limit, `n scored` is the number of records actually sent, not the full test split (`n test`).
- records with a runner error (transport failure or non-200 status) are excluded from scoring by the harness and counted under `errors`.
- latency = the runner's per-request wall time (httpx, concurrency as configured) in milliseconds.
