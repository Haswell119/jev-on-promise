# Findings

What the experiments established, in the order they established it.
`experiments/index.jsonl` is the machine-readable record and
`research/BACKLOG.md` the ranked list of what is still open. This file is
the argument: what was believed, what was measured, and what changed.

## Retrieval is the long-context bottleneck, not reasoning

The symbolic champion answers 53.5% of the internal dev set correctly and
falls off sharply with state length. The first question was whether that
is a reasoning failure or a retrieval failure, and it is a retrieval
failure: at a 140-word evidence budget the decisive sentence reached the
scorer in roughly a third of long states. A model cannot be right about
evidence it never sees, so everything below is about the selector.

One number bounds the opportunity. The annotated span is a median 38
words against a 139-word block, so the budget is not what binds. A
perfect selector would retrieve nearly every span where the original
retrieved 0.31.

## R1: per-candidate quotas made it worse

Reserving slots so each option contributes its own supporting fragment
dropped recall from 0.283 to 0.145. The reserved slots were spent on
decoy paragraphs that name the wrong option. Rejected.

## R2: BM25 was missing its document-frequency term

The scorer weighted query terms by a corpus-level idf but never by how
many of *this state's* segments contain the term. Inside one long
document that is the discriminating signal: a state about refunds
mentions "refund" everywhere, so the term localises nothing, and long
states were scored almost uniformly. Adding a state-local idf, and
expanding the question query with the criteria synonym sets, raised full
recall from 0.288 to 0.306 with no length bucket regressing and the gain
concentrated above 1k tokens. Adopted.

A third idea measured in the same pass, boosting segments that reproduce
a candidate phrase near-verbatim, moved recall by 0.000 in every bucket.
Removed rather than left as a dead switch.

## R3a: a metric that could not fail

The first learned ranker scored 0.9946 against a 0.9929 baseline. That is
not a narrow win, it is a broken measurement: most records in the
training pool are under 128 tokens, so the whole state fits the budget
and retrieval never has to choose. Capping negatives at 60 per list
compounded it. Aborted, and the exporter now keeps only states that
exceed the budget, with every negative.

The general lesson is worth more than the run: a metric computed over a
population that does not contain the failure mode reports success
whatever the model does.

## R3 and R4: the ranker won its own metric and lost the real one

R3 raised budget recall from 0.592 to 0.731 on held-out records and
*dropped* suite recall from 0.306 to 0.238. Two causes, both fixed in R4:
the fit maximised the count of annotated segments retrieved while the
suite measures token-level containment of the gold string, and a weight
of +4.03 on a binary negation flag was a generator artefact that a
record-level selection split could not detect.

R4 fixed both, with token-share targets and a split across structural
template signatures, and reached 0.285. Better, still short of 0.306.

## R5: an ablation that did not test what it looked like

Disabling the heuristic's document-order fill drops recall to 0.160, but
it also leaves the budget unspent: 526 of 1548 rows fall under 100 words
of the 140 available. Without the fill the selector can only use segments
BM25 scored above zero, and BM25 scores too few. The fallback's
contribution is mostly that it spends the budget at all. Recorded as
aborted rather than presented as a finding about position.

## R8: the answer was contiguity, not ranking

The span is a median 38 words and runs across several segments. The
recall metric scores the longest unbroken run of gold tokens, and a
reader needs a whole sentence for the same reason, so a span picked in
pieces counts only as its largest fragment. Taking the neighbours of each
chosen segment fixes that:

| glue | 0 | 1 | 2 | 3 | 4 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| fully retrieved | 0.306 | 0.403 | 0.446 | 0.462 | 0.463 | 0.457 | 0.457 |
| lost entirely | 0.504 | 0.339 | 0.315 | 0.316 | 0.316 | 0.317 | 0.318 |

Monotone to 3, flat beyond. At 4096 tokens recall goes from 0.353 to
0.584, at 16384 from 0.366 to 0.581. It costs nothing at inference.

It also settles the ranker. Giving the learned R4 ranker the same
contiguity yields 0.444 against the heuristic ordering's 0.446: the
contiguity is the entire gain and the learned ranking adds nothing. Three
experiments to establish that the answer was not a model.

## C1: the curriculum stopped short of the evaluation

The training pool's long-context split ran to 1024 tokens while the
evaluation suite runs to 16384, so every model had to extrapolate
sixteenfold on the axis that already failed hardest. 3600 records
spanning 256 to 16384 tokens now close that gap, from the same disjoint
train-side template pool and verified by the generator's own checker.

## What this does and does not claim

R2 and R8 change retrieval, which only the neural path uses, so the
symbolic champion's numbers are unaffected by them. The recall figures
are a property of the selector, measured without a model. The accuracy
benefit arrives when a model is trained against the improved evidence,
which is what E5 is for.

## The internal benchmark overstates transfer by an order of magnitude

M1 is the first JevBench milestone taken against a neural champion, and
it is the most important number in this file.

| tier | symbolic | hybrid E1f | gain | target |
|---|---|---|---|---|
| easy | 0.812 | 0.833 | +0.021 | 0.98 |
| standard | 0.444 | 0.472 | +0.028 | 0.80 |
| hard | 0.396 | 0.405 | +0.009 | 0.65 |

The same change measured on the internal suite was worth +0.106 easy,
+0.099 standard and +0.211 hard. On JevBench it is worth +0.021, +0.028
and +0.009: roughly a tenth as much.

The internal benchmark is synthetic and built by the same generator that
produced the training pool. The template pools are disjoint and the
leakage checks pass, so this is not contamination in the usual sense. It
is something subtler and harder to fix: the generator has one idea of
what a decision problem looks like, and a model can learn that shape
instead of the task. Disjoint templates prevent memorising an instance;
they do not prevent learning a house style.

Two consequences, both binding on everything that follows. Internal gains
are a weak signal about real capability and should be discounted heavily
until a milestone says otherwise. And the targets are a long way off:
0.833 against 0.98, 0.472 against 0.80, 0.405 against 0.65.

Sample sizes are 48, 72 and 111, so two or three points on a tier is one
or two questions. The direction is consistent across all three tiers,
which is worth more than any single figure.
