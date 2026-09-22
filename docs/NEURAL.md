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

## Reproducibility

Every experiment is recorded in `experiments/index.jsonl` with its
hypothesis, architecture, dataset version, hyperparameters, seed, hardware,
parameter count, metrics and decision; `experiments/CURRENT_CHAMPION.json`
names the champion. `scripts/research_loop.sh status|next|run|resume` drives
the queue and resumes interrupted training from the checkpoint (model,
optimizer, scheduler, RNG and data cursor).
