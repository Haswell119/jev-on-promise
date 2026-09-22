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

## Measured gap to the published Jev 1.13 reference (frozen engine, run 2)

| Benchmark | Sextant (measured) | Jev 1.13 (published) | Gap |
|---|---|---|---|
| JevBench public easy | 39/48 = 81.2 % | 48/48 = 100 % | −18.8 pts |
| JevBench public standard | 32/72 = 44.4 % | 71/72 = 98.6 % | −54.2 pts |
| JevBench public hard | 44/111 = 39.6 % | 81/111 = 73.0 % | −33.4 pts |
| JevBench judge tier | not public | 94.5 % | n/a |
| jev-bench macro accuracy | 0.415 | 0.733 | −0.318 |
| jev-bench macro ECE | 0.192 | 0.113 | +0.079 (worse) |
| Schema validity | 100 % | 100 % | = |
| Latency, typical request p50 | ≈6 ms (local) | ≈100 ms (published round trip) | ≈16× faster |

None of the staged accuracy goals (Stage A: easy ≥ 95 %, standard ≥ 80 %)
is met on the standard or hard tiers; the easy tier misses Stage A by 14
points. Schema validity, probability validity, determinism, order robustness
and latency goals are met.

Failure classes behind the gap (from aggregate diagnostics and internal
probes; no benchmark item was inspected):

* **Judgement without lexical anchors** — the option or proposition that is
  right shares no vocabulary with the state ("screen is cracked" vs
  "broken or damaged"); the lexical graph closes only part of this gap.
* **Long states** — accuracy drops from 59 % (< 64 tokens) to 21–31 %
  (> 256 tokens) on the public items: retrieval brings the right fragment
  but the fusion cannot weigh it against many partially matching ones.
* **Multi-hop, probability and trade-off reasoning** — near or below chance
  (hard families `multi_hop` 22 %, `probability` 10 %); these need
  arithmetic or chained inference the engine does not perform.
* **Calibration under distribution shift** — the confidence and temperature
  models are fitted on our data; on the standard tier ECE is 0.35 (wrong
  answers stay confident when a distractor option matches one word).
* **Noul on short factual records** improved with class balancing but still
  hovers near 0.5 when the proposition's subject noun is absent from the
  record ("Has the parcel been delivered?" vs a record that says only
  "Status: delivered").

Where the engine *is* strong: literal extraction and tool/route selection
with descriptive options (easy Choice 91.7 %), numeric/date questions that
resolve symbolically, injection resistance, and cost/latency.
