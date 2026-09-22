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
