# Limitations

Sextant is a non-neural engine. It is fast, deterministic and calibrated, but
it does not understand language the way a large model does. Measured
consequences are in `docs/BENCHMARKS.md` and `reports/latest.md`; the
structural limitations are listed here so nobody is surprised.

## Semantic gaps (by design of a lexical/statistical engine)

* **World knowledge.** Questions whose answers require facts not present in
  the state (multiple-choice science questions, commonsense yes/no) are
  answered near chance, with appropriately low confidence when the criteria
  share no vocabulary with the state.
* **Paraphrase without lexical overlap.** WordNet synonyms, hypernyms,
  derivational links, topic domains and character n-grams recover part of the
  gap, but a message like "the device will not turn on after the update"
  matched against "product defects, crashes, bugs" can still be missed. Richer
  option descriptions (`what` / `not_for` / `examples`) help substantially.
* **Multi-hop and policy reasoning.** Applying a rule with several conditions
  and exceptions ("returns within 30 days unless perishable, opened items
  only if defective") is only handled when each condition maps to a symbolic
  comparison; free-form rule application is a known weak class.
* **Ordinal judgement.** Score levels are placed using lexical overlap, a
  magnitude/intensity lexicon and valence; nuanced rubrics ("mostly
  equivalent but unimportant details differ") are hard.
* **Sarcasm, irony, implicit sentiment.** VADER-style valence with negation
  handling covers explicit sentiment only.
* **Language.** English only (stemmer, lexicons, cue lists). Other languages
  degrade to character n-gram and literal matching.

## Robustness boundaries

* Negation is scoped with bounded windows; long-distance negation ("I would
  not, under any circumstances that I can imagine, want a refund") can escape
  the window.
* Directive detection treats imperative meta-instructions in the state as
  data. Adversarial text that *describes* a false situation (rather than
  instructing) is indistinguishable from evidence by design — as it should be.
* Option-order invariance holds for probabilities up to floating-point
  determinism; ties are broken lexicographically.

## Calibration boundaries

* Calibration is fitted on internal data. Out-of-distribution workloads may
  be over- or under-confident; use the `explain` block and re-run
  `sextant calibrate` on a representative labelled split of your own data.
* Confidence models the probability that the argmax is correct given the
  distribution shape and evidence; it is not a probability that the caller's
  rubric was interpreted correctly.

## Operational limits

* Defaults: 4 MiB body, 2 MiB state, depth 64, 255 options, 10 levels, 1024
  questions. Latency grows roughly linearly with options × segments.
* Start-up parses ~8 MB of embedded lexicon (~0.5 s).
