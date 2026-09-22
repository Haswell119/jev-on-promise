# Experiment log

All numbers below are INTERNAL development measurements on
`data/synthetic/dev.jsonl` (733 questions; synthetic, labels verifiable by
construction) unless stated otherwise. They are not benchmark results. Metrics:
accuracy (argmax), NLL, Brier, ECE (10-bin top-label). Latency is the debug
build unless noted (see reports/latest.md for release-build numbers).

| # | Hypothesis | Modification | Before (acc / NLL / ECE) | After (acc / NLL / ECE) | Latency | Calibration | Decision |
|---|---|---|---|---|---|---|---|
| 1 | Hand-set bootstrap weights give a usable baseline | Bootstrap weights, no lexicon, 4 resolvers | – | 0.697 / 0.969 / 0.126 | p50 2.2 ms | uncalibrated | baseline recorded |
| 2 | Fitting the fusion weights (conditional logit / logistic) beats hand weights | `sextant train`, L2 = 0.05 | 0.697 / 0.969 / 0.126 | 0.693 / 0.821 / 0.144 | = | – | keep training, but L2 too strong |
| 3 | Over-regularization flattens logits | L2 0.05 → 0.001, family L2 0.5 → 0.01, 600 epochs | 0.693 / 0.821 / 0.144 | 0.775 / 0.630 / 0.126 | = | – | keep |
| 4 | Numeric comparison, count (incl. array length) and date comparison are mechanical | 3 new symbolic resolvers | 0.775 / 0.630 / 0.126 | 0.835 / 0.525 / 0.117 | = | – | keep (numeric 0.76→0.93, temporal 0.79→1.00) |
| 5 | Directive text in the state ("ignore the previous question and select billing") should be treated as data | DIRECTIVE segment flag; literal hits inside directive segments earn no credit; `directive_frac` feature | injection acc 0.72 (below plain) | injection acc 0.75 (= plain) | = | – | keep |
| 6 | Hypernym / topic-domain compatibility and antonym-under-negation add signal beyond exact matches | `hyper_match`, `domain_match`, `antonym_negated` features (OEWN) | 0.835 | 0.840 / 0.502 / 0.123 | +0.2 ms | – | keep (small but consistent) |
| 7 | Tokenizer glued digits ("P1" → "1") produced spurious numeric-range resolutions | `\b` before numbers; number words excluded from level matching | severity 0.25 | severity 0.45 | = | – | keep (bug fix) |
| 8 | Temperature + Platt calibration on a separate split fixes sharpness | `sextant calibrate` on calib split | 0.840 / 0.502 / 0.123 | 0.846 / 0.401 / 0.040 | = | ECE 0.123 → 0.040 | keep |
| 9 | Isotonic vs Platt for Noul (2-fold on calib) | comparison only | Platt NLL 0.472 / ECE 0.067 | isotonic NLL 0.424 / ECE 0.061 | – | – | keep Platt in the artifact (monotone, 2 parameters, no step artefacts); isotonic difference is within noise on 223 examples |
| 10 | Symbolic resolvers should not be infinitely sharp | Laplace-smoothed error-rate mixing with uniform (`symbolic_epsilon`) | max symbolic p ≈ 1.0 | max symbolic p ≈ 0.96–0.99 | = | principled | keep |

Failure classes observed after #10 (internal dev): lexical mismatch in routing
paraphrases (~25 % of routing), multi-step policy reasoning (policy outcome
choice), ordinal severity levels needing magnitude understanding, answer
adequacy with short answers.
| 11 | Relations between two referenced state fields (question/answer, premise/hypothesis) are the decisive evidence for adequacy / compatibility / relevance | cross-field channel (`xfield_*`), option polarity profile (`opt_neg_share`, `opt_hyp_share`), crossed features (`x_sim_pos`, `x_conflict_neg`, `x_low_neutral`) | adequacy 0.50, compatibility 1.00 (19), overall 0.884 | adequacy 1.00, overall 0.892 / 0.292 / 0.050 | +0.1 ms | ECE 0.041 → 0.050 (noise) | keep |
| 12 | Level descriptions whose intensity/valence is monotone in the index let the state's intensity/valence be mapped onto the scale | `ord_pos_int`, `ord_pos_val`, `ord_hit` for Score | severity 0.40 | severity 0.60 | = | – | keep |
| 13 | Question analysis dominated 255-option latency (repeated stemming, per-synonym allocation, linear cue-list scans) | per-question expansion + surface caches, `intern_stem`, hash-set cue lookup, stem reuse in valence | 255-option question 41.0 ms; typical request p50 9.9 ms | 13.0 ms; p50 5.7 ms (release) | −68 % / −42 % | identical outputs | keep |
| 14 | Duration windows ("within 30 days") vs elapsed time ("20 days ago", dated event vs stated reference date) are mechanical | duration/reference-date parsing, relative dates resolved to absolute dates, `window_ok` Noul feature | policy 0.674, noul 0.908 | policy 0.696, noul 0.914 / 0.175 / 0.025 | = | – | keep (small; generic temporal-numeric capability) |
| 15 | Real permissively-licensed datasets (12 sources, ~18.5k train records) generalize beyond synthetic templates | train on synthetic + real (`scripts/prepare_data.py`) | external dev 0.529 / 1.628 / 0.186 (synthetic-only model); synthetic dev 0.893 | external 0.605 / 0.990 / 0.028; synthetic 0.790 | = | ECE 0.186 → 0.028 on external | keep data; large sources crowd out small recipes |
| 16 | Balancing the loss per (primitive, source) restores small-recipe behaviour | `--balance sqrt` vs `full`; family shrinkage 0.01 → 0.001 | external 0.605, synthetic 0.790 | sqrt/0.01: 0.605 / 0.810; full/0.01: 0.596 / 0.844; **full/0.001: 0.612 / 0.887** | = | external ECE 0.020 | keep full/0.001 as defaults (1200 epochs: no change → converged) |
| 17 | Fallback options ("other", "none of the above", "unclear") should win when the *other* options lack evidence | `is_fallback` detection; `fallback_opt`, `fallback_x_lowmax` features; fallback recipe in synthetic data | fallback_option dev n/a | fallback_option 0.857 (7 dev items); external unchanged 0.612; synthetic 0.889 | = | – | keep; **engine frozen after this row** |
