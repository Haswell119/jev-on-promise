# Research backlog

Ranked hypotheses for improving Sextant's semantic capability. Columns:
expected gain (on internal dev / JevBench-like capabilities), evidence,
implementation cost, compute cost, risk. Sub-agents may append; deduplicate
before starting work. One experiment at a time challenges the champion.

| id | hypothesis | gain | evidence | impl | compute | risk | status |
|---|---|---|---|---|---|---|---|
| H1 | A compact pretrained encoder scoring (question, criterion, retrieved evidence) beats the lexical fusion on paraphrase/implicit-intent items | very high | champion fails when option vocabulary ≠ state vocabulary (routing paraphrases, easy-tier Noul 50 %) | medium | ~6 h CPU | medium | in progress (E1) |
| H2 | Keeping symbolic features alongside the neural logit (hybrid fusion) beats pure neural | high | symbolic resolvers are exact on numeric/date/count items where neural models are weak | low | low | low | pending |
| H3 | Symbolic retrieval of top-K evidence lets a short-sequence encoder handle 2k–16k states | very high | accuracy drops 59 %→21 % above 256 tokens | medium | medium | medium | pending |
| H4 | Listwise (softmax over candidates) training beats pointwise binary training for Choice | medium | native distributions are the product requirement; pointwise needs post-hoc normalisation | low | none extra | low | pending |
| H5 | Multi-hop synthetic curriculum (rule + attribute + exception) lifts hard-tier composition | high | hard families multi_hop 22 %, probability 10 % | medium | medium | medium | pending |
| H6 | Joint CE + Brier / label-smoothing objective improves ECE without hurting accuracy | medium | post-hoc temperature only partially fixes ECE 0.35 on standard tier | low | none extra | low | pending |
| H7 | Shared-state encoding (encode state once, read out many question/criterion heads) cuts latency for many-question requests | medium (latency) | 512-question requests re-encode the state per question | high | high | high | pending |
| H8 | NLI/entailment training data (WANLI, SciTail, HotpotQA-derived) teaches contradiction and adequacy | high | compatibility/adequacy are the weakest non-numeric families | medium | medium | low | pending |
| H9 | Option-order invariance via permutation augmentation + invariance loss | low-medium | current stability already 95 % same-answer; neural models are more order-sensitive | low | low | low | pending |
| H10 | Larger encoder (33M L12 or 100M+) buys accuracy worth the CPU latency | unknown | Pareto search required | low | high | medium | pending |
| H11 | Distilling the trained cross-encoder into a bi-encoder for first-stage candidate pruning at K>32 | medium (latency) | 255-option Choice needs 255 forwards | high | medium | medium | pending |
| H12 | Sextant-native encoder pretraining (MLM + contrastive on decision-shaped data) vs pretrained init | unknown | ideological interest only if it wins | high | very high | high | deferred (CPU-bound) |

## Added after the retrieval diagnostic (R1)

| id | hypothesis | gain | evidence | impl | compute | risk | status |
|---|---|---|---|---|---|---|---|
| H13 | A learned bi-encoder retriever (question → chunk) trained on the 72k gold-evidence annotations lifts long-context recall far above BM25 | very high | R1: BM25 full recall 0.04–0.09 above 1k tokens; gold evidence spans are available for free in the training pool | medium | ~1.5 h CPU | medium | pending |
| H14 | Encoding state chunks ONCE per request and reusing them across questions makes hierarchical retrieval affordable (Direction B+D) | high (latency) | a request with many questions currently re-reads the state per question | high | medium | medium | pending |
| H15 | Adaptive evidence budget (grow with state length, cap 400 words) is worth its latency | medium | recall 0.283 → 0.342 at equal short-state cost | done | low | low | testing in E2 |

## Added after the retrieval scoring experiment (R2)

| id | hypothesis | gain | evidence | impl | compute | risk | status |
|---|---|---|---|---|---|---|---|
| H16 | BM25 retrieval is missing a state-local document-frequency factor, so long states are scored almost uniformly | medium | R2: full evidence recall 0.288 → 0.306 overall, +4 points in every bucket above 1k tokens, no bucket regressed | low | none | low | adopted (R2) |
| H17 | Expanding the question query with the criteria synonym sets reaches needles that paraphrase the question | low-medium | R2: +0.8 points alone, additive with H16 | low | none | low | adopted (R2) |
| H18 | Boosting segments that reproduce a candidate phrase near-verbatim (char 4-gram containment ≥ 0.7) | none | R2: 0.000 change in every length bucket; code removed | low | none | low | rejected (R2) |
| H19 | Oracle ranking is the whole game: the gold span is a median 38 words against a 139-word budget, so a perfect ranker would reach ~1.0 recall where BM25 reaches 0.31 | very high | R2 span measurement; bounds the ceiling for H13 | — | — | — | motivates H13 |

## Added after the first ranker fit (R3a, aborted)

| id | hypothesis | gain | evidence | impl | compute | risk | status |
|---|---|---|---|---|---|---|---|
| H20 | Any retrieval metric must be computed only over states that exceed the budget; including states that fit reports success regardless of the model | — | R3a: budget recall 0.9946 for the ranker vs 0.9929 for the BM25 baseline, because most training records are under 128 tokens and fit the 140-word budget whole | done | none | — | adopted (`export-retrieval --budget-words`) |
| H21 | Negative subsampling has to preserve the real competition: capping at 60 segments per list hides the hundreds a 16k state contains | — | R3a, same run | done | none | — | adopted (default `--max-negatives 0`) |
| H22 | The training pool's long-context split stopped at 1024 tokens while evaluation runs to 16384, so every model extrapolated 16x on the worst axis | high | length histogram of `bench_train.jsonl`: 53128 records at 64 tokens, 790 at 2048, none above | done | low | low | data built and verified (C1); awaits a run that trains on it |
| H23 | Fill the budget by score per word rather than by score: a long high-scoring segment can consume the budget that two short decisive ones needed | medium | the ranker sees `len_ratio` but can only weigh it linearly, and the greedy fill itself is length-blind | low | none | low | pending |

## Added after the ranker was rejected (R3)

| id | hypothesis | gain | evidence | impl | compute | risk | status |
|---|---|---|---|---|---|---|---|
| H24 | Fit the ranker against token-level evidence recall, not the count of annotated segments retrieved: the two come apart, and only the first is what the encoder experiences | high | R3: segment budget recall rose 0.592 to 0.731 while token recall on the frozen suite fell 0.306 to 0.238 | medium | low | low | pending |
| H25 | Hold the ranker's selection set out by TEMPLATE POOL, not by record: a record-level split cannot detect a feature that is an artefact of the training templates | high | R3: a weight of +4.03 on a binary negation flag survived record-level selection and failed across the pool boundary | low | none | low | pending |
| H26 | Much heavier regularisation, or dropping the binary flags entirely, keeps the ranker to signals that generalise | medium | R3 weight profile: the three largest weights are two binary flags and a length ratio | low | none | low | pending |
| H27 | Part of the heuristic's advantage at evidence-at-start is its document-order fallback, not its ranking; measure the heuristic with the fallback disabled to size the real gap | medium (diagnostic) | R3: the ranker's loss is concentrated at start (-0.192) and split (-0.107), the two positions the fallback favours | low | low | low | pending |
| H28 | Retrieve contiguous neighbourhoods, not independent segments: an annotated span runs across several segments, and taking the individually best ones returns it in pieces | high | R4: the heuristic's document-order fill is contiguous by construction and beats the ranker most where scattering hurts most; the recall metric scores the longest contiguous run of gold tokens, so a split span scores as its largest fragment | low | low | low | testing (R6) |

## Raised by the E5 comparison

| id | hypothesis | gain | evidence | impl | compute | risk | status |
|---|---|---|---|---|---|---|---|
| H29 | The dev score has no long-context component, so a model cannot be promoted for the axis the brief calls a primary research target | — | E5f: 0.473 against the champion's 0.405 on the long-context suite and 0.301 against 0.216 on long-context Choice, earning nothing, while losing 0.015 of a score built from easy, standard, hard, calibration, paraphrase, order, latency and memory | low | none | **changing it with a result in hand fits the metric to the answer** | raised, deliberately not acted on |
| H30 | E5 and the champion are complementary rather than ranked: E5 wins long context, Noul and calibration, the champion wins Choice, Score and the judge tier | medium | E5f per-primitive split | medium | medium | medium | pending |
| H31 | Training on more long states cost short-context sharpness; a curriculum weighted toward long context should be measured against one that is not, holding retrieval fixed | medium | E5 vs E1 differ in retrieval AND curriculum AND labels; nothing isolates the curriculum | low | ~2.5 h | low | pending |
