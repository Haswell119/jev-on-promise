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

## Promotion checklist (documentation)

The repository still describes Sextant as a non-neural engine, which is true
of the current champion `C0`. The `/v1/models` card is already computed from
the loaded artifact (`Model::has_neural`), so it needs no edit. The following
prose claims are stale the moment a neural artifact becomes the champion and
must be updated in the same commit as the promotion:

- `README.md` — headline "A non-neural, deterministic, calibrated probabilistic decision engine."
- `crates/core/src/lib.rs` — crate doc comment.
- `crates/cli/src/main.rs` — the clap `about` string.
- `docs/LIMITATIONS.md` — opening paragraph and the capability limits that follow from having no learned semantics.
- `docs/BENCHMARKS.md` — the paragraph explaining the expected ceiling of a non-neural engine.
- `scripts/prepare_data.py` — the dataset card description.

A promotion is not complete while any of these still claims the engine has no
neural network. `scripts/experiment.py promote` prints this list as a reminder.
