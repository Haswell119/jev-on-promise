# Training and calibration

Sextant learns a *small decision surface over deterministic features*, not a
model of language. Every learned number lives in two human-readable JSON
files:

| Artifact | Contents | Fitted by |
|---|---|---|
| `model/weights.json` | feature-named weights for the Choice head, the Score head (each: base expert + per-family deltas) and the Noul head (`diff`, `yes`, `bias`, family deltas) | `sextant train` |
| `model/calibration.json` | temperatures (global / family / bucket), symbolic ε, Platt parameters, ordinal λ, confidence coefficients | `sextant calibrate` |

Both files are embedded into the binary at compile time
(`crates/core/src/model/mod.rs`) and can be overridden at runtime with
`--model-dir`.

## One command

```bash
scripts/train.sh            # builds, generates synthetic data, downloads datasets,
                            # trains, calibrates, runs the leakage check and evals
cargo build --release       # re-embed the new artifact
```

`SEXTANT_SKIP_DATA=1 scripts/train.sh` skips the dataset download step.

## Data

1. **Synthetic recipes** (`scripts/synth/generate.py`, seed 20260922) —
   programmatically labelled scenarios: routing (with paraphrase, distractor,
   JSON wrapping, option permutation, injection and negated-contrast
   transformations), long-context retrieval, enum extraction, numeric levels
   and comparisons, dates, request detection with negation/inquiry
   contrasts, policy eligibility, sentiment levels, severity levels, JSON
   facts, answer adequacy and NLI-style compatibility. Records are split
   train / calib / dev by a hash of their scenario group, so variants of one
   scenario never straddle splits.
2. **Permissive public datasets** (`scripts/prepare_data.py`) — see
   `data/README.md` and `data/provenance.jsonl` for the exact list, licenses,
   revisions and transformations. Every dataset is disjoint from the JevBench
   and jev-bench sources.

Every record keeps `source`, `license`, `split`, `transformation`,
`synthetic` and `group`.

## Training (`sextant train`)

* Extracts features with the exact runtime pipeline (`StateIndex` →
  `QuestionView` → `extract_features`); examples answered by a symbolic
  resolver are excluded from the semantic fit.
* Choice / Score: conditional logit, full-batch Adam (lr 0.05, 600 epochs),
  L2 0.001 on the base expert, 0.01 shrinkage on per-family deltas, families
  with fewer than 40 examples share the base. Score targets are smoothed
  toward adjacent levels (0.08); annotator distributions are used as soft
  targets when present.
* Noul: logistic regression over `[f_yes − f_no, f_yes]` with a bias.
* Deterministic: no sampling, fixed iteration order; `--seed` is recorded
  for provenance only.
* Ablations: `--drop-features a,b` zeroes features during training (the
  resulting weights then ignore them at inference); `--no-families` disables
  the mixture of experts.

## Calibration (`sextant calibrate`)

Uses the trained weights and a **separate** split (never training data,
never benchmark data). Fits temperatures by golden-section search on NLL,
Platt parameters by gradient descent, ordinal λ by grid search, symbolic ε as
the Laplace-smoothed resolver error rate, and the confidence map by logistic
regression of correctness on distribution-shape statistics.
`--compare-isotonic` reports a 2-fold Platt-vs-isotonic comparison.

## Reproducibility checklist

* `python3 scripts/synth/generate.py` → byte-identical JSONL for the same seed.
* `sextant train` twice on the same inputs → identical `weights.json`.
* `reports/leakage.json` must report `PASS` before any benchmark run.
* Record the git commit and the artifact `version` fields in reports.
