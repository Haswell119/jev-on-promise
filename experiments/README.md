# Experiment framework

`experiments/index.jsonl` — one JSON object per experiment (append-only).
`experiments/CURRENT_CHAMPION.json` — machine-readable pointer to the
champion: experiment id, artifact paths, metrics, promotion timestamp.
`experiments/runs/<experiment_id>/` — config, logs, checkpoints, metrics
(gitignored except `config.json` and `metrics.json`).

Every record carries: experiment_id, parent_champion, hypothesis,
architecture, dataset_version, model_init, hyperparameters, seed,
training_time_s, hardware, parameter_count, artifact_bytes, dev_metrics,
calibration_metrics, latency_ms, memory_mb, decision (PROMOTE | REJECT |
ABORT), reason.

## Promotion rule

A challenger replaces the champion only if it improves the weighted
development score computed by `scripts/dev_score.py`:

```
score = 0.34 * standard_like_accuracy
      + 0.30 * hard_like_accuracy
      + 0.10 * easy_like_accuracy
      + 0.10 * (1 - ECE)
      + 0.06 * paraphrase_stability
      + 0.05 * option_order_stability
      + 0.03 * latency_score      # 1 at <=50ms p50, 0 at >=1000ms (log scale)
      + 0.02 * memory_score       # 1 at <=500MB, 0 at >=4GB
```

`standard_like` / `hard_like` / `easy_like` are tiers of the INTERNAL frozen
benchmark (`data/bench_internal/`), never JevBench. A promotion also requires
schema validity = 100 % and no calibration collapse (ECE <= champion + 0.05).
Public JevBench is run only at milestones (internal +5 points, architecture
change, long-context breakthrough) and never used to choose between
challengers.
