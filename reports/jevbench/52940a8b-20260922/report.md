# JevBench public tiers — Sextant `sextant-1`

> **run 2 (final): frozen engine 52940a8, model trained-1790088256**

> Public halves of the easy, standard and hard tiers, scored by the harness's own
> `typesafe` adapter and scoring at the pinned commit. Unattempted and schema-invalid
> answers count as wrong. Held-out items and the judge tier are not public, so these
> numbers are not comparable one-to-one with the published leaderboard.

|  |  |
|---|---|
| Run | `52940a8b-20260922` finished 2026-09-22T14:46:15+00:00 |
| Sextant | commit `52940a8b976d63f611be237d46c0f9b6474e811a` (dirty worktree), `sextant 0.1.0` |
| Harness | fstandhartinger/jevbench @ `75e6224ed8103bbc3485ca74820a2eaf7ce8abe0` (MIT), adapter `typesafe` |
| Endpoint | `http://127.0.0.1:33885` (serial requests, loopback) |
| Hardware | Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores, 15.7 GiB RAM, rustc 1.94.1 (e408947bf 2026-03-25) |
| Task files | `easy.jsonl` 231df3c2c8e8…, `hard.jsonl` 89e9e6becb33…, `original.jsonl` 5c2414edb300… |

## Tiers

| tier | public / frozen | attempted | schema valid | strict | correct | accuracy | chance | intelligence | Brier | ECE | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **easy** | 48 / 72 | 48 | 100.0 % | 100.0 % | 39 | 81.2 % | 28.4 % | 73.8 | 0.231 | 0.059 | 1.3 ms | 2.1 ms |
| **standard** (original) | 72 / 96 | 72 | 100.0 % | 100.0 % | 32 | 44.4 % | 31.7 % | 18.7 | 0.871 | 0.346 | 1.4 ms | 1.9 ms |
| **hard** | 111 / 220 | 111 | 100.0 % | 100.0 % | 44 | 39.6 % | 33.6 % | 9.0 | 0.759 | 0.210 | 3.1 ms | 7.3 ms |
| judge | 0 / 146 | not public | – | – | – | – | 29.2 % | – | – | – | – | – |
| all public | 231 | 231 | 100.0 % | 100.0 % | 115 | 49.8 % | 31.8 % | – | 0.684 | 0.204 | 1.9 ms | 6.9 ms |

## Hard tier by family

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

## By question type

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

## Paraphrase pairs (`group`)

| tier | pairs | both valid | same answer | agreement | both correct | both-correct rate (all pairs) |
|---|---|---|---|---|---|---|
| easy | 0 | 0 | 0 | – | 0 | – |
| standard | 36 | 36 | 30 | 83.3 % | 13 | 36.1 % |
| hard | 0 | 0 | 0 | – | 0 | – |

## Latency (caller wall time per decision, serial, loopback)

| tier | n | p50 | p95 | mean | max | first request | failed |
|---|---|---|---|---|---|---|---|
| easy | 48 | 1.3 ms | 2.1 ms | 1.5 ms | 10.7 ms | 10.7 ms | 0 |
| standard | 72 | 1.4 ms | 1.9 ms | 1.5 ms | 4.4 ms | 4.4 ms | 0 |
| hard | 111 | 3.1 ms | 7.3 ms | 3.8 ms | 13.1 ms | 13.1 ms | 0 |
| all public | 231 | 1.9 ms | 6.9 ms | 2.6 ms | 13.1 ms | 10.7 ms | 0 |

## Definitions

- accuracy = correct / all public items of the tier (harness `n_correct / n_scorable`); unattempted, failed and schema-invalid answers count as wrong.
- schema valid = share of attempted items whose distribution covers exactly the label set, lies in [0, 1] and sums to 1 within 2 % (renormalized); strict = within 0.001 (harness `schema_validity` / `schema_validity_strict`).
- Brier = mean over valid items of sum_k (p_k - y_k)^2 over the exact label set (binary questions use both labels, i.e. 2·(p_yes − y)^2).
- ECE = top-label expected calibration error, 10 equal-width confidence bins, over valid scorable items (harness `ece_top_label`).
- chance = mean of 1/|labels| over the tier's frozen items (harness option-count histograms, JevBench v1.3); intelligence = 100·(accuracy − chance)/(1 − chance), clipped to [0, 100].
- paraphrase pairs = `group`s of exactly two items; agreement = same predicted label among pairs with two valid answers; both-correct rate uses all pairs as denominator (harness `agreement`).
- latency = caller wall time per decision including HTTP, one request at a time (harness `latency_s`); p50/p95 use the harness's interpolating percentile.
