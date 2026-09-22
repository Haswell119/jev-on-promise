# jev-bench test splits — Sextant `sextant-1`

> **smoke run, untrained/bootstrap model (hand-set bootstrap weights; first 20 test records per config, not the full 22,773)**

> Only the first 20 records of each config were sent (`--limit 20`); the full test set is 22,773 records.

|  |  |
|---|---|
| Run | `3643094f-20260922` finished 2026-09-22T14:21:14+00:00 |
| Sextant | commit `3643094fbfba9b74be9b1c4d66d3241acf797d98` (dirty worktree), `sextant 0.1.0` |
| Dataset | Praveenrajus/jev-bench @ `002ad22de8db2df5e0eb898b3da8072dbd4af4de` (test splits only) |
| Harness | uspraveen/Jevify @ `2891025b8a4520d0eb639f2cd1b3fddaf4b922b3` (Apache-2.0), `jevify-run api` + `jevify-run report` |
| Endpoint | `http://127.0.0.1:40591`, concurrency 4 |
| Hardware | Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores, 15.7 GiB RAM, rustc 1.94.1 (e408947bf 2026-03-25) |

## Leaderboard-style aggregates

| configs | records scored | errors | macro acc | macro ECE | macro Brier | sel@90 | choice acc | score acc | noul acc | TVD→human | latency p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | 440 | 0 | 0.391 | 0.241 | 0.647 | 0.404 | 0.333 | 0.233 | 0.600 | 0.550 | 6.4 ms | 10.6 ms |

## Per config

| config | prim | K | scored / test | errors | acc | ECE | Brier | NLL | sel@90 | MAE | AUROC | TVD→human | p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | variable | 20 / 1000 | 0 | 0.400 | 0.374 | 0.852 | 1.86 | 0.389 | – | – | – | 6.5 ms |
| `banking77` | choice | 77 | 20 / 1000 | 0 | 0.400 | 0.363 | 0.757 | 2.18 | 0.444 | – | – | – | 7.7 ms |
| `boolq` | noul | variable | 20 / 1000 | 0 | 0.700 | 0.200 | 0.250 | 0.69 | 0.722 | – | 0.500 | – | 6.0 ms |
| `chaosnli` | choice | 3 | 20 / 1599 | 0 | 0.350 | 0.101 | 0.648 | 1.05 | 0.333 | – | – | 0.368 | 6.0 ms |
| `civil_comments` | noul | variable | 20 / 2000 | 0 | 0.450 | 0.188 | 0.272 | 0.74 | 0.444 | – | 0.944 | 0.428 | 5.9 ms |
| `clinc150` | choice | 151 | 20 / 1000 | 0 | 0.450 | 0.279 | 0.748 | 2.96 | 0.500 | – | – | – | 10.1 ms |
| `fever_evidence` | noul | variable | 20 / 1000 | 0 | 0.450 | 0.203 | 0.278 | 0.76 | 0.500 | – | 0.520 | – | 6.5 ms |
| `go_emotions` | choice | 28 | 20 / 1000 | 0 | 0.100 | 0.071 | 0.915 | 3.15 | 0.111 | – | – | 0.845 | 6.9 ms |
| `helpsteer2_helpfulness` | score | 5 | 20 / 1000 | 0 | 0.350 | 0.298 | 0.842 | 1.62 | 0.333 | 0.92 | – | – | 6.5 ms |
| `helpsteer2_verbosity` | score | 5 | 20 / 1000 | 0 | 0.300 | 0.288 | 0.849 | 1.66 | 0.278 | 0.76 | – | – | 6.8 ms |
| `ledgar` | choice | 100 | 20 / 1000 | 0 | 0.400 | 0.282 | 0.887 | 3.28 | 0.444 | – | – | – | 7.8 ms |
| `massive` | choice | 60 | 20 / 1000 | 0 | 0.400 | 0.271 | 0.702 | 2.34 | 0.389 | – | – | – | 7.3 ms |
| `measuring_hate_speech` | score | 3 | 20 / 1000 | 0 | 0.200 | 0.263 | 0.800 | 1.35 | 0.222 | 1.01 | – | 0.557 | 7.2 ms |
| `mmlu` | choice | 4 | 20 / 1000 | 0 | 0.100 | 0.358 | 0.979 | 2.13 | 0.111 | – | – | – | 7.7 ms |
| `mnli` | choice | 3 | 20 / 1000 | 0 | 0.400 | 0.059 | 0.703 | 1.18 | 0.444 | – | – | – | 8.0 ms |
| `paws` | noul | variable | 20 / 1000 | 0 | 0.550 | 0.072 | 0.253 | 0.70 | 0.611 | – | 0.500 | – | 5.8 ms |
| `sms_spam` | noul | variable | 20 / 800 | 0 | 0.850 | 0.221 | 0.175 | 0.54 | 0.833 | – | 0.625 | – | 5.7 ms |
| `sst5` | score | 5 | 20 / 1000 | 0 | 0.250 | 0.166 | 0.717 | 1.39 | 0.278 | 1.01 | – | – | 5.9 ms |
| `strategyqa_closed` | noul | variable | 20 / 687 | 0 | 0.600 | 0.100 | 0.250 | 0.69 | 0.611 | – | 0.500 | – | 5.8 ms |
| `strategyqa_grounded` | noul | variable | 20 / 687 | 0 | 0.600 | 0.393 | 0.395 | 2.01 | 0.611 | – | 0.500 | – | 5.7 ms |
| `stsb` | score | 6 | 20 / 1000 | 0 | 0.050 | 0.355 | 1.004 | 2.29 | 0.056 | 1.48 | – | – | 6.0 ms |
| `yelp5` | score | 5 | 20 / 1000 | 0 | 0.250 | 0.392 | 0.957 | 1.98 | 0.222 | 1.29 | – | – | 5.8 ms |

## Definitions

- Every per-config number is the harness's own (jevify.bench.metrics at the pinned commit): accuracy = argmax (choice/score) or P(yes) >= 0.5 (noul); ECE = top-label, 15 equal-width bins; Brier = sum over options of (p - y)^2 for choice/score and (p_yes - y)^2 for noul; NLL = -log p(gold); sel@90 = accuracy on the 90 % most confident records (model `confidence` for choice/score, |2p-1| for noul); AURC = area under the risk-coverage curve; RPS/MAE = ordinal scores for score configs; AUROC for noul; TVD→human = total variation distance to the human label distribution on the four calibration-gold configs.
- macro rows are unweighted means over the configs that were scored; per-primitive rows average the configs of that primitive. With --limit, `n scored` is the number of records actually sent, not the full test split (`n test`).
- records with a runner error (transport failure or non-200 status) are excluded from scoring by the harness and counted under `errors`.
- latency = the runner's per-request wall time (httpx, concurrency as configured) in milliseconds.
