# The neural decision scorer

Sextant is a **hybrid** engine: deterministic symbolic reasoning for
mechanical questions, a locally hosted neural encoder for semantic
judgement, and one calibrated probability layer on top. Nothing leaves the
machine: no API, no LLM, no network at inference.

## What the neural component is (and is not)

It is a **compact encoder that scores candidates**, not a text generator.
For one question it computes

```
compatibility(state, question, criterion) -> scalar logit
```

for every caller-supplied criterion, and the engine turns the group of
logits into a probability distribution. There is no fixed label vocabulary:
a brand-new option description supplied at request time is scored the same
way as any other, which is what keeps the Choice / Score / Noul abstraction
intact. No text is decoded at any point.

## Inference path

```
state ──► symbolic index (once per request)
           │
           ├─► symbolic resolvers ──────────────► exact answer (numbers, dates,
           │      (numeric, date, count, literal,   counts, literals, booleans)
           │       reference equality, boolean)
           │
           └─► evidence selection (BM25 + focus fields, word budget)
                      │
question + criteria ──┴─► encoder: [CLS] question | criterion [SEP] evidence
                              │
                              ▼
                        one logit per criterion
                              │
             fused: w_n · neural + w_s · symbolic   (weights fitted on a
                              │                      held-out split)
                              ▼
              softmax / ordinal smoothing → calibration → confidence
```

`resolver_priority` (default true) keeps the exact resolvers in charge when
they fire: arithmetic, date comparison and literal extraction stay
deterministic, and the encoder is not consulted at all. That is a deliberate
design choice — code is more reliable than a model wherever code applies.

## Artifact layout

`model/neural/` (or any directory passed to `--model-dir`, which the engine
loads as `<model-dir>/neural`):

| File | Contents |
|---|---|
| `config.json` | encoder architecture (HuggingFace BERT layout) |
| `model.safetensors` | encoder weights |
| `tokenizer.json` | the exact tokenizer used at training time |
| `head.safetensors` | pooling head (`l1`, `l2`, optional `feat_norm`) |
| `scorer.json` | max length, pooling, evidence budget/strategy, feature usage, provenance |

The Rust runtime loads it with [candle](https://github.com/huggingface/candle)
and the `tokenizers` crate. The core crate keeps `#![forbid(unsafe_code)]`,
so weights are read through the safe (non-mmap) safetensors loader. Building
with `--no-default-features` yields a pure symbolic binary with no tensor
dependencies.

## Training

```bash
# 1. export training inputs through the production pipeline (this is what
#    guarantees that training and inference see identical text)
sextant export-pairs data/synthetic/bench_train.jsonl data/processed/*/train.jsonl \
    --out data/neural/train.jsonl --strategy quota --budget-words 140

# 2. train (CPU-friendly, resumable, checkpoints optimizer + RNG + cursor)
python3 scripts/neural/train.py --train data/neural/train.jsonl \
    --dev data/neural/bench_dev.jsonl --out experiments/runs/E1 \
    --encoder sentence-transformers/all-MiniLM-L6-v2 --pooling mean \
    --max-len 192 --group-batch 4 --max-candidates 6 --balance sqrt

# 3. export to the Rust layout, fit the fusion block, evaluate, get a verdict
scripts/evaluate_challenger.sh E1
```

Training details that matter:

* **Listwise objective.** The loss is a softmax over the candidates of one
  question (cross-entropy, optionally plus a Brier term), so the model
  learns calibrated *relative* compatibility rather than per-label
  classification. Score questions use ordinal label smoothing; annotator
  distributions are used as soft targets when a dataset provides them.
* **No position leakage.** Candidates are reshuffled on every training step
  and negatives are subsampled, so candidate order cannot become a shortcut.
  Option-order stability is measured explicitly at evaluation time.
* **Per-source balancing** (`--balance sqrt`) prevents a large synthetic
  pool from drowning the smaller real, human-labelled corpora.
* **Train/inference parity is tested**: `sextant neural-probe` scores the
  same rows in Rust and `scripts/neural/score_pairs.py` in Python; the
  measured maximum logit difference is 1.2e-5 (float32 rounding).

## Evidence selection

Long states are not truncated blindly. The symbolic index scores every
segment (BM25 over the question terms, the criteria terms and their WordNet
expansions, plus a prior on fields the question references by path) and the
best fragments are handed to the encoder in document order, within a word
budget that can grow with the state length (`evidence_adaptive_cap`).

`reports/evidence_recall_study.json` measures how often the decisive
sentence actually survives this step. It is the dominant long-context
failure mode and is tracked as a first-class metric.

Two scoring switches are recorded per model in `scorer.json`, because
retrieval must be identical in training and inference:

- `evidence_local_idf` weights each query term by its document frequency
  *inside this state*, `ln(1 + (N - df + 0.5) / (df + 0.5))` over segments.
  The query-side weight already carries a corpus-level idf, but within one
  long document the discriminating signal is how many of its own segments
  contain the term. Without it a state about refunds scores almost
  uniformly on the term "refund" and ranking degenerates to document order.
- `evidence_q_expand` adds the criteria synonym sets to the question-only
  query at 0.35 weight, so a fragment that paraphrases the question rather
  than an option is still reachable.

Measured on the internal long-context suite (1548 questions, 140-word
budget, `reports/retrieval/R2_local_idf.json`), full recall of the gold
span rises from 0.288 to 0.306 and the fraction of questions where the span
is lost entirely falls from 0.525 to 0.504. No length bucket regressed and
the gain concentrates where retrieval is worst: +4.2 points at 1k tokens,
+3.9 at 4k, +4.0 at 16k. Both switches default to off so that models
trained before this change keep their parity.

A third idea measured in the same pass, boosting segments that reproduce a
candidate phrase near-verbatim (char 4-gram containment at least 0.7),
changed recall by 0.000 in every bucket; the code was removed rather than
left as a dead switch.

The ceiling here is high and unclaimed. The gold span is a median 38 words
against a 139-word block, so a perfect ranker would retrieve nearly all of
them where BM25 retrieves 0.31. That gap, not the encoder, is the main
long-context target (backlog H13, a learned bi-encoder retriever).

## Reproducibility

Every experiment is recorded in `experiments/index.jsonl` with its
hypothesis, architecture, dataset version, hyperparameters, seed, hardware,
parameter count, metrics and decision; `experiments/CURRENT_CHAMPION.json`
names the champion. `scripts/research_loop.sh status|next|run|resume` drives
the queue and resumes interrupted training from the checkpoint (model,
optimizer, scheduler, RNG and data cursor).

## Promotion checklist (documentation), completed

E1 was promoted on 2026-09-23 and the prose claims that the engine has no
neural network were corrected in the same commit: `README.md`,
`crates/core/src/lib.rs`, `crates/core/Cargo.toml`,
`crates/cli/src/main.rs`, `docs/LIMITATIONS.md`, `docs/BENCHMARKS.md` and
`scripts/prepare_data.py`. The `/v1/models` card was already computed from
the loaded artifact via `Model::has_neural`, so it needed no edit.

`scripts/experiment.py promote` prints this list whenever a neural
artifact becomes the champion, so a future promotion cannot quietly leave
the claim stale again.

## Learned retrieval ranker

Retrieval, not the encoder, is the dominant long-context failure, and the
budget is not what binds: the annotated span is a median 38 words against a
139-word block, so a perfect ranker would retrieve nearly all of them where
BM25 retrieves about a third.

`model/retrieval.json` holds a linear ranker over 16 features that
retrieval already computes for every segment: z-scores and rank
percentiles of the question, pooled and best-candidate BM25 vectors, the
question and candidate term coverage, candidate char-4-gram containment,
length ratio, document position, the focus-field flag, number, date and
negation flags, and the log segment count. Scores and ranks are normalised
within the state, because raw BM25 magnitudes are not comparable across
documents and a ranker trained on raw values learns the document rather
than the segment.

It is linear on purpose. The ranker runs on every segment of every state
on the inference path, so it has to cost one dot product; a second encoder
pass over hundreds of segments would blow the CPU latency budget on its
own. Capacity belongs in the cross-encoder, which only ever sees the
retrieved block.

Training is listwise: each state's segments form one list and the loss is
the negative log of the probability mass the softmax puts on the annotated
evidence segments. Per-segment binary classification would spend its
capacity on the hundreds of easy negatives a long state contains. Model
selection uses budget recall, the fraction of annotated segments that
survive a greedy fill of the word budget, because a ranking metric would
hide a long winning segment crowding the needle out.

    sextant export-retrieval data/synthetic/bench_train.jsonl \
        --out data/retrieval/train.jsonl --local-idf --q-expand
    python3 scripts/neural/train_retrieval.py \
        --train data/retrieval/train.jsonl --dev data/retrieval/dev.jsonl \
        --out model/retrieval.json --version R3

**No ranker is currently adopted.** The first one fitted (experiment R3)
was rejected: see the measurement below. With no `retrieval.json` present
the engine keeps the heuristic pooled ordering, so the ranker is strictly
additive and production behaviour is unchanged. When one is loaded the
document-order fallback disappears: every segment is a candidate and the
budget is filled greedily by learned score, which also removes the bias
toward the start of long states that the fallback introduced.

The ranker is fitted on the training pool only and selected on a held-out
slice of it, never on the internal dev set or the long-context suite, so
the recall reported on those suites remains a clean measurement.

## Long-context curriculum

The training pool's long-context split originally stopped at 1024 tokens
while the evaluation suite runs to 16384, so every model had to
extrapolate sixteen-fold on the axis that already fails hardest.
`data/synthetic/bench_train_lc.jsonl` closes that gap: 3600 records over
256 to 16384 tokens with the evidence at the start, middle, end and split
across the state, drawn from the same disjoint train-side template pool.

    python3 scripts/synth/bench/build.py --long-train-only
    python3 scripts/synth/bench/verify.py

The verifier re-derives every gold label from `meta.checks` and confirms
that the bench and train template pools stay disjoint with no state
leakage in either direction. It is written as a separate file rather than
folded into `bench_train.jsonl` so that training exports made before it
existed stay reproducible.

### A saturated metric, and how the ranker avoids it

The first fit of the ranker (experiment R3a) reported budget recall 0.9946
against a pooled-BM25 baseline of 0.9929 on the held-out slice. That is not
a small win, it is a broken measurement: most records in the training pool
are under 128 tokens, so the whole state fits in a 140-word budget and
retrieval never has to choose. Capping negatives at 60 per list compounded
it, shortening every list far below the hundreds of segments a real long
state contains. The run was recorded as ABORT and its weights discarded.

`sextant export-retrieval` therefore takes `--budget-words` and skips any
question whose entire state fits inside it, and defaults to keeping every
negative. What remains is exactly the population the ranker exists to
serve: states where something has to be left out.

The general point is worth stating because it applies to the whole
research loop. A metric computed over a population that does not contain
the failure mode will report success no matter what the model does.

### R3: the ranker was rejected

The first fit on the corrected population beat the pooled-BM25 baseline
clearly on the metric it was trained for. Budget recall on the held-out
slice rose from 0.592 to 0.731 and top-1 from 0.601 to 0.712. On the
frozen long-context suite it was worse:

| | heuristic | ranker |
|---|---|---|
| gold span fully retrieved | 0.306 | 0.238 |
| gold span lost entirely | 0.504 | 0.625 |

Every length bucket from 256 tokens upward regressed, worst at 512 tokens
(0.558 to 0.342). By evidence position the damage concentrates at the
start (0.644 to 0.452) and split (0.605 to 0.499), the two positions the
heuristic's document-order fallback happens to favour.

Two causes, both actionable, both now in the backlog:

- **Metric mismatch.** The fit maximises the fraction of annotated
  *segments* retrieved; the suite measures token-level containment of the
  gold *strings*. A ranker can retrieve the right number of segments and
  the wrong ones. The next attempt optimises the suite's metric directly.
- **Template-pool shift.** The bench and train template pools are disjoint
  by construction. A weight of +4.03 on a binary negation flag is exactly
  the sort of generator artefact that correlates with evidence inside one
  pool and not the other, and a record-level selection split cannot see
  it. The next attempt holds out by template pool.

The negative result is worth as much as a positive one here: it says the
gap between BM25 and a perfect ranker is not closed by better weights over
these features, and it caught a selection protocol that would have passed
a bad model through.

### Selecting the ranker across a template boundary

`split_retrieval.py --by template` keys the selection split on a record's
**structural signature**: its non-slot template families, deduplicated,
instance numbers stripped, sorted and joined. Slot templates are the value
fillers (names, cities, products) and appear in nearly every record, so
they cannot partition anything; the structural families decide what a
record looks like.

Requiring every individual template to land on one side would drop almost
every record, since each carries five to twelve of them. Keying on the
whole combination holds out structures instead: 644 distinct signatures
over the training pool, which partitions cleanly (518 signatures for
fitting, 126 for selection, none on both sides).

This is weaker than the bench/train pool boundary, because two signatures
can share a family, and stronger than a record-level split, which puts the
same structure on both sides by construction. It exists so that a weight
learned from a generator artefact fails during selection rather than after
adoption.

## Contiguity: the largest retrieval gain measured

The annotated span is a median 38 words and routinely runs across several
segments. The recall metric scores the longest unbroken run of gold
tokens, and a reader needs a whole sentence for the same reason, so a
span retrieved in pieces counts only as its largest fragment. Selecting
the individually best segments scatters it.

`neighbour_glue` takes N segments either side of each chosen segment
while the budget allows. On the frozen long-context suite, at a 140-word
budget:

| glue | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|
| span fully retrieved | 0.306 | 0.403 | 0.446 | 0.462 | 0.463 | 0.452 | 0.457 | 0.457 |
| span lost entirely | 0.504 | 0.339 | 0.315 | 0.316 | 0.316 | 0.317 | 0.317 | 0.318 |

Monotone to 3 and flat beyond, so 3 is the knee and the default. Every
length bucket improves and the long ones most: 2048 tokens 0.356 to
0.493, 4096 tokens 0.353 to 0.584, 8192 tokens 0.327 to 0.523, 16384
tokens 0.366 to 0.581. Every evidence position improves too.

This also settles the ranker question. Applying the same contiguity to
the learned R4 ranker gives 0.444, against 0.446 for the heuristic
ordering at the same width. The contiguity is the whole gain; the
learned ranking adds nothing on top of it. R3 and R4 remain rejected and
no `retrieval.json` is shipped.

It changes retrieval only, so the symbolic champion is unaffected. The
accuracy benefit arrives when a model is trained against the improved
evidence, which is why the switch is recorded per model in `scorer.json`
alongside the others.

## Where the numbers live

`reports/champions.md`, regenerated by `scripts/champion_progression.py`,
carries the champion progression with per-tier accuracy, the retrieval and
milestone entries that were adopted without becoming champions, and every
experiment including the rejected ones. `experiments/index.jsonl` is the
machine-readable source it reads, and `research/FINDINGS.md` is the
argument behind the numbers.

The one figure to read first is M1, the JevBench milestone. Gains on the
internal suite have run about ten times larger than the same change is
worth on JevBench, so an internal result is a weak signal until a
milestone confirms it.
