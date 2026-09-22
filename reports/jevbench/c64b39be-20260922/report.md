# JevBench public tiers — Sextant `sextant-1`

> **frozen engine c64b39b, model trained-1790087629**

> Public halves of the easy, standard and hard tiers, scored by the harness's own
> `typesafe` adapter and scoring at the pinned commit. Unattempted and schema-invalid
> answers count as wrong. Held-out items and the judge tier are not public, so these
> numbers are not comparable one-to-one with the published leaderboard.

|  |  |
|---|---|
| Run | `c64b39be-20260922` finished 2026-09-22T14:35:32+00:00 |
| Sextant | commit `c64b39be2f20f189c21b71301cb7b0f0d88e2732`, `sextant 0.1.0` |
| Harness | fstandhartinger/jevbench @ `75e6224ed8103bbc3485ca74820a2eaf7ce8abe0` (MIT), adapter `typesafe` |
| Endpoint | `http://127.0.0.1:58579` (serial requests, loopback) |
| Hardware | Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores, 15.7 GiB RAM, rustc 1.94.1 (e408947bf 2026-03-25) |
| Task files | `easy.jsonl` 231df3c2c8e8…, `hard.jsonl` 89e9e6becb33…, `original.jsonl` 5c2414edb300… |

## Tiers

| tier | public / frozen | attempted | schema valid | strict | correct | accuracy | chance | intelligence | Brier | ECE | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **easy** | 48 / 72 | 48 | 100.0 % | 100.0 % | 38 | 79.2 % | 28.4 % | 70.9 | 0.246 | 0.134 | 1.5 ms | 2.9 ms |
| **standard** (original) | 72 / 96 | 72 | 100.0 % | 100.0 % | 32 | 44.4 % | 31.7 % | 18.7 | 0.822 | 0.300 | 2.2 ms | 2.6 ms |
| **hard** | 111 / 220 | 111 | 100.0 % | 100.0 % | 42 | 37.8 % | 33.6 % | 6.3 | 0.763 | 0.232 | 3.1 ms | 7.9 ms |
| judge | 0 / 146 | not public | – | – | – | – | 29.2 % | – | – | – | – | – |
| all public | 231 | 231 | 100.0 % | 100.0 % | 112 | 48.5 % | 31.8 % | – | 0.674 | 0.216 | 2.4 ms | 7.1 ms |

## Hard tier by family

| family | n | correct | accuracy | chance | Brier | ECE |
|---|---|---|---|---|---|---|
| `adversarial` | 6 | 2 | 33.3 % | 33.3 % | 0.709 | 0.231 |
| `ambiguous` | 7 | 2 | 28.6 % | 32.1 % | 1.064 | 0.422 |
| `judge_hard` | 17 | 10 | 58.8 % | 50.0 % | 0.567 | 0.206 |
| `long_policy` | 19 | 7 | 36.8 % | 30.1 % | 0.854 | 0.363 |
| `multi_hop` | 18 | 4 | 22.2 % | 25.2 % | 0.788 | 0.336 |
| `probability` | 10 | 1 | 10.0 % | 38.3 % | 1.005 | 0.644 |
| `routing_hard` | 5 | 5 | 100.0 % | 20.0 % | 0.238 | 0.386 |
| `temporal_numeric` | 15 | 4 | 26.7 % | 30.9 % | 0.707 | 0.233 |
| `tradeoff` | 6 | 3 | 50.0 % | 30.8 % | 1.141 | 0.782 |
| `trap` | 8 | 4 | 50.0 % | 37.5 % | 0.528 | 0.293 |

## By question type

| tier | type | n | correct | accuracy | Brier | ECE | ordinal MAE |
|---|---|---|---|---|---|---|---|
| easy | `choice` | 36 | 33 | 91.7 % | 0.112 | 0.081 | – |
| easy | `noul` | 12 | 5 | 41.7 % | 0.647 | 0.371 | – |
| standard | `choice` | 36 | 16 | 44.4 % | 0.871 | 0.336 | – |
| standard | `noul` | 24 | 12 | 50.0 % | 0.790 | 0.388 | – |
| standard | `score` | 12 | 4 | 33.3 % | 0.737 | 0.161 | 1.05 |
| hard | `choice` | 67 | 20 | 29.9 % | 0.808 | 0.207 | – |
| hard | `noul` | 38 | 20 | 52.6 % | 0.694 | 0.306 | – |
| hard | `score` | 6 | 2 | 33.3 % | 0.694 | 0.124 | 0.79 |

## Paraphrase pairs (`group`)

| tier | pairs | both valid | same answer | agreement | both correct | both-correct rate (all pairs) |
|---|---|---|---|---|---|---|
| easy | 0 | 0 | 0 | – | 0 | – |
| standard | 36 | 36 | 30 | 83.3 % | 13 | 36.1 % |
| hard | 0 | 0 | 0 | – | 0 | – |

## Latency (caller wall time per decision, serial, loopback)

| tier | n | p50 | p95 | mean | max | first request | failed |
|---|---|---|---|---|---|---|---|
| easy | 48 | 1.5 ms | 2.9 ms | 2.0 ms | 11.6 ms | 11.6 ms | 0 |
| standard | 72 | 2.2 ms | 2.6 ms | 2.2 ms | 5.2 ms | 5.2 ms | 0 |
| hard | 111 | 3.1 ms | 7.9 ms | 4.1 ms | 13.9 ms | 13.9 ms | 0 |
| all public | 231 | 2.4 ms | 7.1 ms | 3.1 ms | 13.9 ms | 11.6 ms | 0 |

## Definitions

- accuracy = correct / all public items of the tier (harness `n_correct / n_scorable`); unattempted, failed and schema-invalid answers count as wrong.
- schema valid = share of attempted items whose distribution covers exactly the label set, lies in [0, 1] and sums to 1 within 2 % (renormalized); strict = within 0.001 (harness `schema_validity` / `schema_validity_strict`).
- Brier = mean over valid items of sum_k (p_k - y_k)^2 over the exact label set (binary questions use both labels, i.e. 2·(p_yes − y)^2).
- ECE = top-label expected calibration error, 10 equal-width confidence bins, over valid scorable items (harness `ece_top_label`).
- chance = mean of 1/|labels| over the tier's frozen items (harness option-count histograms, JevBench v1.3); intelligence = 100·(accuracy − chance)/(1 − chance), clipped to [0, 100].
- paraphrase pairs = `group`s of exactly two items; agreement = same predicted label among pairs with two valid answers; both-correct rate uses all pairs as denominator (harness `agreement`).
- latency = caller wall time per decision including HTTP, one request at a time (harness `latency_s`); p50/p95 use the harness's interpolating percentile.
