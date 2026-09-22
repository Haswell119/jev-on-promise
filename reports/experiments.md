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
