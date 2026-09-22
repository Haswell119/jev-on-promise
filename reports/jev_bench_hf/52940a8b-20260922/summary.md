# jev-bench test splits — Sextant `sextant-1`

|  |  |
|---|---|
| Run | `52940a8b-20260922` finished 2026-09-22T14:47:11+00:00 |
| Sextant | commit `52940a8b976d63f611be237d46c0f9b6474e811a` (dirty worktree), `sextant 0.1.0` |
| Dataset | Praveenrajus/jev-bench @ `002ad22de8db2df5e0eb898b3da8072dbd4af4de` (test splits only) |
| Harness | uspraveen/Jevify @ `2891025b8a4520d0eb639f2cd1b3fddaf4b922b3` (Apache-2.0), `jevify-run api` + `jevify-run report` |
| Endpoint | `http://127.0.0.1:37255`, concurrency 4 |
| Hardware | Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores, 15.7 GiB RAM, rustc 1.94.1 (e408947bf 2026-03-25) |

## Leaderboard-style aggregates

| configs | records scored | errors | macro acc | macro ECE | macro Brier | sel@90 | choice acc | score acc | noul acc | TVD→human | latency p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | 22773 | 0 | 0.415 | 0.192 | 0.646 | 0.425 | 0.329 | 0.302 | 0.623 | 0.486 | 6.4 ms | 12.3 ms |

## Per config

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

## Definitions

- Every per-config number is the harness's own (jevify.bench.metrics at the pinned commit): accuracy = argmax (choice/score) or P(yes) >= 0.5 (noul); ECE = top-label, 15 equal-width bins; Brier = sum over options of (p - y)^2 for choice/score and (p_yes - y)^2 for noul; NLL = -log p(gold); sel@90 = accuracy on the 90 % most confident records (model `confidence` for choice/score, |2p-1| for noul); AURC = area under the risk-coverage curve; RPS/MAE = ordinal scores for score configs; AUROC for noul; TVD→human = total variation distance to the human label distribution on the four calibration-gold configs.
- macro rows are unweighted means over the configs that were scored; per-primitive rows average the configs of that primitive. With --limit, `n scored` is the number of records actually sent, not the full test split (`n test`).
- records with a runner error (transport failure or non-200 status) are excluded from scoring by the harness and counted under `errors`.
- latency = the runner's per-request wall time (httpx, concurrency as configured) in milliseconds.
