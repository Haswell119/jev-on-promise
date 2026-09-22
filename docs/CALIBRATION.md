# Calibration and confidence

Calibration is a first-class component. Its parameters live in
`model/calibration.json`, separate from the semantic weights, and are fitted
by `sextant calibrate` on a split that is disjoint from training and from
every benchmark.

## What is calibrated

| Component | Method | Granularity |
|---|---|---|
| Choice / Score probabilities | temperature scaling `softmax(z / T)` | global per primitive; per question family (≥ 40 examples); cardinality-bucket multiplier (2, 3–4, 5–8, 9–16, 17–64, 65+) |
| Symbolic (resolver) answers | temperature 1 on the fixed resolver scale + Laplace-smoothed uniform mixing `p' = (1−ε)p + ε/K`, `ε = (errors + 1)/(n + 2)` | per primitive |
| Noul | Platt scaling `σ(a·logit + b)` | global; per family (≥ 40 examples); symbolic ε-mixing with 0.5 |
| Score shape | ordinal adjacency smoothing λ (grid search on NLL) | global |
| Confidence | logistic map over distribution shape (see below) | per primitive |

Temperature and λ are fitted by golden-section search on the mean negative
log-likelihood of the held-out calibration examples; Platt parameters by
gradient descent on binary cross-entropy. Everything is deterministic.

## Why temperature + Platt (and not isotonic) in the artifact

`sextant calibrate --compare-isotonic` performs a 2-fold comparison of Platt
scaling against pool-adjacent-violators isotonic regression for Noul. On the
internal calibration split the two were within noise (isotonic NLL 0.424 /
ECE 0.061 vs Platt NLL 0.472 / ECE 0.067 on 223 examples). Platt is kept in
the artifact because it is monotone, has two parameters, cannot produce
step artefacts and extrapolates sensibly; the comparison is re-run and
reported whenever the artifact is refitted.

## Confidence

Choice and Score answers carry a scalar confidence that is *not* `max(p)`.
It is built from the shape of the distribution and evidence signals:

```
concentration = 1 − H(p) / log K            (H = Shannon entropy)
margin        = p₁ − p₂                    (winner minus runner-up)
evidence      = max_k cov_w(k)             (best idf-weighted term coverage)
ood           = min_k ood(k)               (share of rare criterion terms absent from the state)

confidence = σ(a·concentration + b·margin + c·evidence + d·ood + e)
```

The five coefficients are fitted per primitive by logistic regression of
*empirical correctness* on those four statistics over the calibration split,
so `confidence ≈ P(argmax is correct)`. `explain=true` shows the calibration
parameters that were applied. Noul exposes its probability directly.

## Metrics reported

`sextant eval` reports accuracy, NLL, Brier (multi-class sum), ECE (10-bin
and 15-bin top-label), schema validity (loose 2e-2 and strict 1e-3 sum
tolerance), selective accuracy at 50 % coverage by confidence, and the AUROC
of confidence as a predictor of correctness. Reliability data for curves is
in the eval JSON report.

## Rules

* Never calibrate on final benchmark data.
* Keep calibration artifacts separate from semantic weights.
* A temperature must be positive and finite, λ in [0, 1); `Model::load_dir`
  rejects invalid artifacts.
