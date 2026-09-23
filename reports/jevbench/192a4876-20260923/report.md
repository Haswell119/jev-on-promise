# JevBench public tiers — Sextant `sextant-1`

> **hybrid champion E1f (MiniLM-L6 cross-encoder + symbolic), Noul orientation fixed**

> Public halves of the easy, standard and hard tiers, scored by the harness's own
> `typesafe` adapter and scoring at the pinned commit. Unattempted and schema-invalid
> answers count as wrong. Held-out items and the judge tier are not public, so these
> numbers are not comparable one-to-one with the published leaderboard.

|  |  |
|---|---|
| Run | `192a4876-20260923` finished 2026-09-23T07:11:37+00:00 |
| Sextant | commit `192a4876db2b411bb6a1e1da250e19ff69708b78`, `sextant 0.1.0` |
| Harness | fstandhartinger/jevbench @ `75e6224ed8103bbc3485ca74820a2eaf7ce8abe0` (MIT), adapter `typesafe` |
| Endpoint | `http://127.0.0.1:60403` (serial requests, loopback) |
| Hardware | Intel(R) Xeon(R) Processor @ 2.10GHz, 4 cores, 15.7 GiB RAM, rustc 1.94.1 (e408947bf 2026-03-25) |
| Task files | `easy.jsonl` 231df3c2c8e8…, `hard.jsonl` 89e9e6becb33…, `original.jsonl` 5c2414edb300… |

## Tiers

| tier | public / frozen | attempted | schema valid | strict | correct | accuracy | chance | intelligence | Brier | ECE | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **easy** | 48 / 72 | 48 | 100.0 % | 100.0 % | 40 | 83.3 % | 28.4 % | 76.7 | 0.236 | 0.059 | 56.1 ms | 75.5 ms |
| **standard** (original) | 72 / 96 | 72 | 100.0 % | 100.0 % | 34 | 47.2 % | 31.7 % | 22.8 | 0.628 | 0.192 | 65.6 ms | 90.6 ms |
| **hard** | 111 / 220 | 111 | 100.0 % | 100.0 % | 45 | 40.5 % | 33.6 % | 10.4 | 0.655 | 0.057 | 304.8 ms | 472.1 ms |
| judge | 0 / 146 | not public | – | – | – | – | 29.2 % | – | – | – | – | – |
| all public | 231 | 231 | 100.0 % | 100.0 % | 119 | 51.5 % | 31.8 % | – | 0.559 | 0.072 | 87.1 ms | 445.7 ms |

## Hard tier by family

| family | n | correct | accuracy | chance | Brier | ECE |
|---|---|---|---|---|---|---|
| `adversarial` | 6 | 3 | 50.0 % | 33.3 % | 0.609 | 0.360 |
| `ambiguous` | 7 | 2 | 28.6 % | 32.1 % | 0.745 | 0.309 |
| `judge_hard` | 17 | 10 | 58.8 % | 50.0 % | 0.489 | 0.066 |
| `long_policy` | 19 | 8 | 42.1 % | 30.1 % | 0.663 | 0.215 |
| `multi_hop` | 18 | 5 | 27.8 % | 25.2 % | 0.735 | 0.151 |
| `probability` | 10 | 5 | 50.0 % | 38.3 % | 0.725 | 0.219 |
| `routing_hard` | 5 | 3 | 60.0 % | 20.0 % | 0.681 | 0.427 |
| `temporal_numeric` | 15 | 3 | 20.0 % | 30.9 % | 0.694 | 0.274 |
| `tradeoff` | 6 | 2 | 33.3 % | 30.8 % | 0.742 | 0.351 |
| `trap` | 8 | 4 | 50.0 % | 37.5 % | 0.525 | 0.352 |

## By question type

| tier | type | n | correct | accuracy | Brier | ECE | ordinal MAE |
|---|---|---|---|---|---|---|---|
| easy | `choice` | 36 | 34 | 94.4 % | 0.128 | 0.078 | – |
| easy | `noul` | 12 | 6 | 50.0 % | 0.559 | 0.148 | – |
| standard | `choice` | 36 | 17 | 47.2 % | 0.645 | 0.208 | – |
| standard | `noul` | 24 | 13 | 54.2 % | 0.540 | 0.169 | – |
| standard | `score` | 12 | 4 | 33.3 % | 0.751 | 0.301 | 0.90 |
| hard | `choice` | 67 | 23 | 34.3 % | 0.723 | 0.061 | – |
| hard | `noul` | 38 | 21 | 55.3 % | 0.523 | 0.078 | – |
| hard | `score` | 6 | 1 | 16.7 % | 0.732 | 0.115 | 0.69 |

## Paraphrase pairs (`group`)

| tier | pairs | both valid | same answer | agreement | both correct | both-correct rate (all pairs) |
|---|---|---|---|---|---|---|
| easy | 0 | 0 | 0 | – | 0 | – |
| standard | 36 | 36 | 26 | 72.2 % | 13 | 36.1 % |
| hard | 0 | 0 | 0 | – | 0 | – |

## Latency (caller wall time per decision, serial, loopback)

| tier | n | p50 | p95 | mean | max | first request | failed |
|---|---|---|---|---|---|---|---|
| easy | 48 | 56.1 ms | 75.5 ms | 50.2 ms | 78.9 ms | 78.9 ms | 0 |
| standard | 72 | 65.6 ms | 90.6 ms | 65.4 ms | 111.8 ms | 65.6 ms | 0 |
| hard | 111 | 304.8 ms | 472.1 ms | 290.0 ms | 559.8 ms | 483.4 ms | 0 |
| all public | 231 | 87.1 ms | 445.7 ms | 170.2 ms | 559.8 ms | 78.9 ms | 0 |

## Definitions

- accuracy = correct / all public items of the tier (harness `n_correct / n_scorable`); unattempted, failed and schema-invalid answers count as wrong.
- schema valid = share of attempted items whose distribution covers exactly the label set, lies in [0, 1] and sums to 1 within 2 % (renormalized); strict = within 0.001 (harness `schema_validity` / `schema_validity_strict`).
- Brier = mean over valid items of sum_k (p_k - y_k)^2 over the exact label set (binary questions use both labels, i.e. 2·(p_yes − y)^2).
- ECE = top-label expected calibration error, 10 equal-width confidence bins, over valid scorable items (harness `ece_top_label`).
- chance = mean of 1/|labels| over the tier's frozen items (harness option-count histograms, JevBench v1.3); intelligence = 100·(accuracy − chance)/(1 − chance), clipped to [0, 100].
- paraphrase pairs = `group`s of exactly two items; agreement = same predicted label among pairs with two valid answers; both-correct rate uses all pairs as denominator (harness `agreement`).
- latency = caller wall time per decision including HTTP, one request at a time (harness `latency_s`); p50/p95 use the harness's interpolating percentile.
