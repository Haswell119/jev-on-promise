# Research notes (Phase 1)

Clean-room research conducted from **public** sources only. No TypeSafe/Jev API
calls were made, no Jev outputs were collected, nothing was distilled or
reverse-engineered. Dates: 2026-09-22.

## 1. The public System One contract (docs.typesafe.ai)

* `POST /v1/systemone` with `{state, model, questions}`; `state` is a string,
  object or array; `questions` is a caller-keyed map; keys never reach the
  model. Answers come back under the same keys plus `usage`.
* Three primitives: **Choice** (`criteria`: option → string | object | array |
  null, max 255 options; answer `choice`, `probabilities`, `confidence`),
  **Score** (`criteria`: ordered array of 2–10 level descriptions; answer
  `score` = Σ level×P(level), `legend`, `probabilities` keyed `"0".."K-1"`,
  `confidence`), **Noul** (optional `criteria.true/false`; answer `noul` =
  P(yes)).
* Instructions and criteria accept JSON structure; backticked names refer to
  state paths or to sibling fields of the instruction object ("Does
  `extracted_value` match the `field` in `source_text`?").
* Documented limits: 64k tokens per request, 32k for state + longest question;
  errors 401/422/429/529; "reliably up to roughly 240 options".
* Confidence is "derived from the probability distribution"; the exact formula
  is unpublished (the demo approximates `(n·max−1)/(n−1)`, the preview API used
  `1 − normalized entropy`).
* Published Jev 1.13 behaviour/latency: ~100 ms typical round-trip;
  batching 13 questions over a 54k-char document costs 0.27 s.
* Documented jagged edges of jev-1.13 (each is a design target for us):
  literal reading; arithmetic/counting/numeric encodings; date/time comparison;
  indirection and double negatives; large irrelevant state (context rot);
  adversarial content not treated as hostile; contradictory instructions vs
  criteria; no structural invariants across questions; no generation.
* Design lessons stated publicly: do deterministic work in code; ask atomic
  questions; fan out speculatively; describe situations, not degrees; use
  contrastive `what` / `not_for` / `examples` objects; include `other`
  options; keep code in control.

## 2. Benchmarks

Two different public artifacts carry the name "jev bench":

| | JevBench (Benchmark Heaven) | jev-bench (Hugging Face, Praveenrajus) |
|---|---|---|
| Repo | `fstandhartinger/jevbench` @ `75e6224ed8103bbc3485ca74820a2eaf7ce8abe0` (MIT) | dataset `Praveenrajus/jev-bench` @ `002ad22de8db2df5e0eb898b3da8072dbd4af4de`; harness `uspraveen/Jevify` @ `2891025b8a4520d0eb639f2cd1b3fddaf4b922b3` (Apache-2.0) |
| Content | 534 authored/imported typed decisions in tiers easy 72 / standard 96 / judge 146 / hard 220 | 22 configs (166k rows, 22,773 test) recast from public human-labelled datasets |
| Public part | 231 items: easy 48, standard ("original") 72, hard 111; judge tier and held-out halves are private | everything |
| Jev 1.13 reference | easy 1.000, standard 0.990, judge 0.945, hard 0.741, schema validity 100 %, hard ECE 0.061 (public subset: 48/48, 71/72, 81/111) | macro accuracy 0.733, macro ECE 0.113, Brier 0.349 |

The reference profile in the task statement (~100 / ~99 / ~94.5 / ~74) is
JevBench. Both harnesses drive any `POST /v1/systemone` endpoint; both are used
unmodified at the pinned commits (see `benchmarks/`). Scoring rules that matter
for us: probabilities must cover exactly the label set and sum to 1 within 1e-3
(strict) or 2e-2 (renormalised), argmax with lexicographic tie-break, failed
answers count as wrong, ECE = 10-bin top-label.

**Contamination policy.** Neither benchmark's data is used for training,
coefficient fitting, prompt/criteria development, calibration or optimisation.
All 21 upstream corpora of jev-bench (banking77, clinc150, massive, ledgar,
go_emotions, mmlu, ai2_arc, mnli/chaosnli, sst, yelp, HelpSteer2, stsb,
measuring-hate-speech, boolq, fever, paws, civil_comments, sms_spam,
StrategyQA) and all mirrors/recasts are excluded from our data pipeline, as is
anything derived from `auto-model-router`. `sextant leakage` enforces this
mechanically (exact / normalized / 5-shingle MinHash) and its report is
`reports/leakage.json`.

## 3. Lexical resources (licenses verified from the actual files)

| Resource | Use | License |
|---|---|---|
| Open English WordNet 2024 (WNDB) | synonyms, hypernyms, similar/derivational links, antonyms, topic domains | CC BY 4.0 (attribution to Princeton WordNet and the OEWN team) |
| VADER lexicon | valence in [-4, 4] → sentiment channel | MIT |
| Google Books Ngram derived 1-gram list (orgtre) | background IDF prior (10k words) | CC BY 3.0 |
| rust-stemmers (Snowball English) | stemming | MIT / BSD-3 |

Rejected (license): NRC EmoLex/VAD, SO-CAL, labMT, DepecheMood, BioScope,
General Inquirer, MPQA, Norvig `count_1w`, SUBTLEX, NLTK stopwords, PPDB
(unverifiable), ConceptNet (CC BY-SA, size). Own-authored: negation / modality /
request / exception cue lists, magnitude lexicon, stopwords.

## 4. Training data (permissive, benchmark-disjoint)

See `data/README.md` and `data/provenance.jsonl`. Synthetic development
recipes (`scripts/synth/generate.py`) cover routing, enum extraction, numeric
and temporal comparison, negation/request detection, policy applicability,
sentiment, severity, JSON facts, adequacy and NLI-style compatibility, with
paraphrase / distractor / JSON-wrap / permutation / injection transformations.

## 5. Classical techniques adopted

BM25 (with a background IDF prior), TF-IDF cosine, token Jaccard, character
4-gram Dice/coverage, phrase bigrams, verbatim literal matching with word
boundaries, NegEx/ConText-style bounded scope for negation, hypotheticals,
requests and exceptions, WordNet synonym/hypernym/antonym/domain expansion,
VADER-style valence with negation flip and intensity, symbolic resolvers for
enum extraction, numeric ranges/comparisons (incl. array counts), date
comparisons, reference equality and boolean fields, conditional-logit fusion
with per-family linear experts, temperature/Platt calibration, ordinal
adjacency smoothing and a logistic confidence map over distribution shape.
