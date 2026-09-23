# Champion progression

Generated from `experiments/index.jsonl`. Every figure is a recorded measurement; a metric a run never measured is left blank rather than filled in.

Generated 2026-09-23T08:50:26+00:00

## Champions, in order

| id | dev score | overall | easy | standard | hard | ECE | p50 ms | architecture |
|---|---|---|---|---|---|---|---|---|
| C0 | 0.5847 | 0.5351 | 0.6791 | 0.5351 | 0.5009 | 0.1071 | 5.9 | symbolic: BM25 + tf-idf + char n-grams + WordNet graph + neg |
| E1 | 0.5936 | 0.5598 | 0.6604 | 0.5340 | 0.5544 | 0.0880 | 60.7 | MiniLM-L6 cross-encoder (22.9M params), mean pooling, listwi |
| E1f | 0.7048 | 0.6790 | 0.7850 | 0.6340 | 0.7117 | 0.0578 | 59.0 | E1's MiniLM-L6 cross-encoder unchanged; only the engine's No |

Current champion: **E1f**, dev score 0.7048, promoted 2026-09-23T07:05:56+00:00.


## Adopted without becoming a champion

Changes to retrieval or measurement, and benchmark milestones. They carry no dev score of their own because they change what a model is given or how it is judged, not the model.

| id | what it changed |
|---|---|
| R2 | Both switches improve evidence recall monotonically with no regression in any length bucket; gains concentrate in the long buckets (1k +4.2pts, 4k +3. |
| R8 | The largest retrieval gain measured so far, and it costs nothing at inference |
| M1 | The honest number, and it is far below both the targets and what the internal benchmark implied |

## Every experiment, including the ones that failed

A rejected or aborted run is kept because the reason it failed is the result. Three learned rankers were rejected before the answer turned out to be contiguity, and two measurements were aborted because they could not have failed.

| id | decision | what it establishes |
|---|---|---|
| C0 | BASELINE | pre-neural baseline measured on the frozen internal benchmark (3000 dev records) |
| R1 | REJECT | per-candidate quotas at a 140-word budget are worse than pooled (0.145 vs 0.283 full recall); wider budgets help sub-linearly (0.28/0.44/0.57 at 140/3 |
| PILOT | REJECT | toy model, as expected below the champion (dev_score 0.585 -> 0.488) |
| R2 | PROMOTE | Both switches improve evidence recall monotonically with no regression in any length bucket; gains concentrate in the long buckets (1k +4.2pts, 4k +3. |
| R3a | ABORT | Measurement invalid, no conclusion drawn about the hypothesis |
| R3 | REJECT | Wins on the metric it was fitted to and loses on the one that matters |
| R4 | REJECT | Both corrections worked and the ranker still does not beat the heuristic |
| R5 | ABORT | The ablation does not test what it was meant to test, and saying so is the result |
| R8 | PROMOTE | The largest retrieval gain measured so far, and it costs nothing at inference |
| E1 | PROMOTE | Clears the bar, and by far less than the model is worth |
| E1f | PROMOTE | The same weights, read the right way round |
| M1 | BASELINE | The honest number, and it is far below both the targets and what the internal benchmark implied |

2 abort, 2 baseline, 4 promote, 4 reject.

