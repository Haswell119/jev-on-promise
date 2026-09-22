# JevBench public tiers — Sextant `sextant-1`

> **smoke run, untrained/bootstrap model (hand-set bootstrap weights, no training on any JevBench item)**

> Public halves of the easy, standard and hard tiers, scored by the harness's own
> `typesafe` adapter and scoring at the pinned commit. Unattempted and schema-invalid
> answers count as wrong. Held-out items and the judge tier are not public, so these
> numbers are not comparable one-to-one with the published leaderboard.

|  |  |
|---|---|
| Run | `6c8430b1-20260922` finished 2026-09-22T14:00:05+00:00 |
| Sextant | commit `6c8430b13d9e0b6a5c679f0244f3b0e721fb4797` (dirty worktree), `sextant 0.1.0` |
| Harness | fstandhartinger/jevbench @ `75e6224ed8103bbc3485ca74820a2eaf7ce8abe0` (MIT), adapter `typesafe` |
| Endpoint | `http://127.0.0.1:52713` (serial requests, loopback) |
| Hardware | Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores, 15.7 GiB RAM, rustc 1.94.1 (e408947bf 2026-03-25) |
| Task files | `easy.jsonl` 231df3c2c8e8…, `hard.jsonl` 89e9e6becb33…, `original.jsonl` 5c2414edb300… |

## Tiers

| tier | public / frozen | attempted | schema valid | strict | correct | accuracy | chance | intelligence | Brier | ECE | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **easy** | 48 / 72 | 48 | 100.0 % | 100.0 % | 39 | 81.2 % | 28.4 % | 73.8 | 0.252 | 0.094 | 1.9 ms | 3.5 ms |
| **standard** (original) | 72 / 96 | 72 | 100.0 % | 100.0 % | 32 | 44.4 % | 31.7 % | 18.7 | 0.750 | 0.258 | 1.8 ms | 2.5 ms |
| **hard** | 111 / 220 | 111 | 100.0 % | 100.0 % | 35 | 31.5 % | 33.6 % | 0.0 | 1.007 | 0.400 | 2.8 ms | 7.1 ms |
| judge | 0 / 146 | not public | – | – | – | – | 29.2 % | – | – | – | – | – |
| all public | 231 | 231 | 100.0 % | 100.0 % | 106 | 45.9 % | 31.8 % | – | 0.770 | 0.283 | 2.1 ms | 6.6 ms |

## Hard tier by family

| family | n | correct | accuracy | chance | Brier | ECE |
|---|---|---|---|---|---|---|
| `adversarial` | 6 | 2 | 33.3 % | 33.3 % | 0.794 | 0.470 |
| `ambiguous` | 7 | 2 | 28.6 % | 32.1 % | 1.215 | 0.629 |
| `judge_hard` | 17 | 7 | 41.2 % | 50.0 % | 0.793 | 0.382 |
| `long_policy` | 19 | 5 | 26.3 % | 30.1 % | 1.047 | 0.492 |
| `multi_hop` | 18 | 5 | 27.8 % | 25.2 % | 1.058 | 0.502 |
| `probability` | 10 | 3 | 30.0 % | 38.3 % | 1.098 | 0.645 |
| `routing_hard` | 5 | 4 | 80.0 % | 20.0 % | 0.612 | 0.477 |
| `temporal_numeric` | 15 | 4 | 26.7 % | 30.9 % | 0.994 | 0.500 |
| `tradeoff` | 6 | 2 | 33.3 % | 30.8 % | 0.817 | 0.412 |
| `trap` | 8 | 1 | 12.5 % | 37.5 % | 1.533 | 0.831 |

## By question type

| tier | type | n | correct | accuracy | Brier | ECE | ordinal MAE |
|---|---|---|---|---|---|---|---|
| easy | `choice` | 36 | 32 | 88.9 % | 0.130 | 0.087 | – |
| easy | `noul` | 12 | 7 | 58.3 % | 0.618 | 0.292 | – |
| standard | `choice` | 36 | 15 | 41.7 % | 0.903 | 0.419 | – |
| standard | `noul` | 24 | 10 | 41.7 % | 0.597 | 0.190 | – |
| standard | `score` | 12 | 7 | 58.3 % | 0.598 | 0.328 | 0.84 |
| hard | `choice` | 67 | 16 | 23.9 % | 1.097 | 0.412 | – |
| hard | `noul` | 38 | 17 | 44.7 % | 0.892 | 0.416 | – |
| hard | `score` | 6 | 2 | 33.3 % | 0.736 | 0.445 | 0.63 |

## Paraphrase pairs (`group`)

| tier | pairs | both valid | same answer | agreement | both correct | both-correct rate (all pairs) |
|---|---|---|---|---|---|---|
| easy | 0 | 0 | 0 | – | 0 | – |
| standard | 36 | 36 | 19 | 52.8 % | 9 | 25.0 % |
| hard | 0 | 0 | 0 | – | 0 | – |

## Latency (caller wall time per decision, serial, loopback)

| tier | n | p50 | p95 | mean | max | first request | failed |
|---|---|---|---|---|---|---|---|
| easy | 48 | 1.9 ms | 3.5 ms | 2.0 ms | 10.5 ms | 10.5 ms | 0 |
| standard | 72 | 1.8 ms | 2.5 ms | 1.9 ms | 6.8 ms | 6.8 ms | 0 |
| hard | 111 | 2.8 ms | 7.1 ms | 3.8 ms | 13.1 ms | 13.1 ms | 0 |
| all public | 231 | 2.1 ms | 6.6 ms | 2.8 ms | 13.1 ms | 10.5 ms | 0 |

## Definitions

- accuracy = correct / all public items of the tier (harness `n_correct / n_scorable`); unattempted, failed and schema-invalid answers count as wrong.
- schema valid = share of attempted items whose distribution covers exactly the label set, lies in [0, 1] and sums to 1 within 2 % (renormalized); strict = within 0.001 (harness `schema_validity` / `schema_validity_strict`).
- Brier = mean over valid items of sum_k (p_k - y_k)^2 over the exact label set (binary questions use both labels, i.e. 2·(p_yes − y)^2).
- ECE = top-label expected calibration error, 10 equal-width confidence bins, over valid scorable items (harness `ece_top_label`).
- chance = mean of 1/|labels| over the tier's frozen items (harness option-count histograms, JevBench v1.3); intelligence = 100·(accuracy − chance)/(1 − chance), clipped to [0, 100].
- paraphrase pairs = `group`s of exactly two items; agreement = same predicted label among pairs with two valid answers; both-correct rate uses all pairs as denominator (harness `agreement`).
- latency = caller wall time per decision including HTTP, one request at a time (harness `latency_s`); p50/p95 use the harness's interpolating percentile.
