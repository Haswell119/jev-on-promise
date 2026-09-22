# Algorithms

Everything here is classical, deterministic and inspectable. There are no
neural networks, embeddings or external models anywhere in the runtime.

## 1. Text pipeline

* **Normalization**: Unicode NFKC, unified quotes/dashes/spaces, zero-width and
  control characters removed; case preserved for display, lowercased for
  matching.
* **Tokenizer**: a single compiled regex recognises URLs, e-mails, ISO and
  slash dates, times, currency amounts (`$12,840.00`, `12 USD`, `5k`),
  percentages, ordinals and identifiers (`#4471`, `INV-2026-001`); the gaps
  are split into Unicode words and punctuation. Contractions are expanded
  (`can't` → `can not`, `we're` → `we are`, possessive `'s` dropped).
* **Stemming**: Snowball English (rust-stemmers). Terms are interned by stem;
  the surface form is kept for lexicon lookups.
* **Segmentation**: abbreviation- and decimal-aware sentence splitting, hard
  breaks at newlines, long fragments chunked at clause punctuation (≤ 400
  chars) so retrieval works at a useful granularity.
* **Numbers and dates**: numeric values with thousands separators, magnitude
  suffixes, currency codes and percent; number words; ISO / slash / textual
  dates reduced to days-since-epoch (Hinnant's civil-from-days).

## 2. Scope annotation (negation, modality, requests, exceptions, directives)

A NegEx/ConText-style scanner marks tokens with flags. A cue opens a scope
that extends over the following content tokens until a window limit
(negation 6, hypothetical 8, request 8, exception 2) or a clause terminator
(`, ; . ! ? :`, `but`, `however`, `although`, `unless`, …). A second
negation cue inside an active scope toggles it (double negation). Idioms
such as *not only*, *no doubt* do not negate. Hard cues (pure function
words: *not, no, never, whether, if, please, except, very, …*) are not
evidence themselves; content-bearing cues (*failed, refuse, possible,
want*) stay evidence but still open scopes. Interrogative segments are
flagged as a whole. Directive segments (imperative meta-verbs like *ignore,
select, classify, answer* combined with meta-nouns like *classifier,
assistant, system, instructions*, or "the correct X is", "regardless of")
are flagged as data that addresses the reader rather than describing a
situation.

Falsey JSON values negate their key tokens (`"refund_requested": false` →
*refund requested* negated).

## 3. Shared state index

Per request: fields with dotted paths and key terms; segments with sorted
(term, tf) vectors, tf-idf norms, sorted char 4-gram hashes, valence,
intensity, flag union and lowercase offsets; postings term → (segment, tf);
global tf; phrase bigrams over consecutive evidence terms (gap ≤ 3);
numbers; dates; arrays with lengths; exact-path and key-term indexes;
hypernym-ancestor and topic-domain weight maps from the lexical graph.

**Background IDF**: `idf(t) = ln(max_count / count(t)) / ln(max_count /
min_count)` from a 10k-word Google Books frequency list, clipped to [0, 1];
unknown words get 1 (rare), digits 0.7.

**BM25**: k1 = 1.2, b = 0.75 over segments with the background IDF as the
term weight; the whole state is also scored as one document.

## 4. Question and criteria understanding

* Instruction objects are split into *question text* (question-like keys or
  strings ending in `?`) and *reference data* (other fields). Backticked
  names resolve to state paths (→ focus fields) or to reference data.
* Criterion objects are flattened recursively with polarity: keys containing
  negation/exclusion morphemes (`not_for`, `excludes`, `negative_examples`,
  `unless`, `except`, …) flip polarity; example-like keys (`examples`,
  `phrases`, `keywords`, `synonyms`, …) become individually scored phrases;
  label-like keys (`what`, `description`, …) contribute no words of their
  own; `{"flag": false}` yields a negated key term.
* Option keys are tokenised as names (`track_order` → *track*, *order*),
  including negation (`no_refund` → *refund* negated).
* Literals: the key and short descriptions (≤ 6 words) are candidates for
  verbatim matching with word boundaries.
* Numeric ranges: "Under $1,000", "$1,000 to $10,000", "Over 1M", "at least
  10%", "3–5 days", "Net 30" → `[lo, hi]` with inclusivity, kind (currency /
  percent / plain) and unit.
* Family detector: keyword rules over the instruction and criteria shape
  select one of 16 archetypes (enum extraction, intent, routing, topic,
  policy, factual yes/no, adequacy, sentiment, severity, request detection,
  relevance, compatibility, similarity, numeric, temporal, generic). Families
  only select a weight profile.

## 5. Feature ensemble (per criterion)

Retrieval and lexical: `bm25_best`, `bm25_global`, `cov_w` (idf-weighted
term coverage), `cov_best`, `cov_rare`, `cov_name`, `cos_global`,
`cos_best`, `jaccard_best`, `bigram_hits`, `gram_dice_best`, `gram_cov`,
`literal_hit/len/count`, `syn_cov` (WordNet synonym-only matches),
`hyper_match` (hypernym ancestry with depth discount 0.75^d, ancestors within
3 levels of the root ignored), `domain_match` (WordNet topic domains),
`example_max/mean`, `focus_cov`, `focus_literal`, `key_match`,
`key_negated`, `evidence_density`, `q_cov_best`.

Polarity and modality: `neg_agree` / `neg_conflict` (criterion term negated
vs. state occurrence negated), `hyp_conflict` (state only asks or speculates
while the criterion asserts), `hyp_state`, `req_agree`, `antonym_hits`,
`antonym_negated`, `neg_field_cov/best` (exclusion fields present in the
state), `directive_frac`.

Sentiment / ordinal: `valence_agree` (criterion valence × state valence,
VADER with negation flip and intensity scaling), `valence_abs`,
`intensity_dist`, `intensity_cov`, `ord_pos_int`, `ord_pos_val`, `ord_hit`
(state intensity/valence mapped onto a level scale whose descriptions are
monotone in that dimension).

Numeric: `range_hit`, `range_dist`.

Cross-field (when the question references two state fields):
`xfield_cov_ab/ba`, `xfield_jaccard`, `xfield_cos`, `xfield_gram`,
`xfield_neg_conflict`, `xfield_antonym`, `xfield_num_conflict`, crossed with
the option's polarity profile (`x_sim_pos`, `x_conflict_neg`,
`x_low_neutral`; `opt_neg_share`, `opt_hyp_share`).

Structural: `term_count_log`, `ood`, `null_desc`, `bias`.

Expensive channels (cosine, Jaccard, char-gram Dice) run only on the top-5
segments by BM25 per criterion; everything else is a pass over postings.

## 6. Symbolic resolvers

Fire only when the answer is mechanically determined; otherwise return
nothing and the semantic path runs.

| Resolver | Fires when | Output |
|---|---|---|
| `enum_extraction` | all options are literal-like (null or ≤ 3 words) and exactly one option has the longest verbatim match (focus-aware, directive segments excluded) | +6 winner, partial credit to shorter matches |
| `numeric_range` | ≥ 2 levels/options parse as numeric ranges and every candidate number (kind/unit/focus filtered, digits only) lands in the same range | +6 winner, −distance for others |
| `numeric_comparison` | the question has a comparator + threshold and a unique quantity (array length for "N items", unit-adjacent numbers, focus numbers) | ±4.8 |
| `date_comparison` | the question has before/after/by + a date and all (focused) state dates agree | ±4.8 |
| `reference_equality` | the question asks match/equal/appear and cites specific reference values; all present → yes, none → no | ±4.8 |
| `boolean_field` | a unique boolean field whose key words are all in the question | ±4.8 |

## 7. Fusion

* Choice / Score: conditional logit `z_k = (w_base + Δw_family) · f_k`,
  softmax over options. Trained with full-batch Adam (lr 0.05, 600 epochs)
  on cross-entropy (+ ordinal label smoothing 0.08 to neighbours for Score,
  + soft targets when annotator distributions exist), L2 = 0.001 on the base
  and 0.01 on family deltas (families with < 40 examples share the base).
* Noul: logistic regression on the concatenation of `f_yes − f_no` and
  `f_yes`, where the *no* hypothesis carries the polarity-flipped
  proposition terms; unknown evidence keeps the logit near the bias.
* Symbolic logits bypass the heads.

## 8. Probabilities, calibration and confidence

See `docs/CALIBRATION.md`. In short: stable softmax with a calibrated
temperature per primitive × family × cardinality bucket; Platt scaling for
Noul; Laplace-smoothed uniform mixing for symbolic answers; ordinal adjacency
smoothing `p'_k ∝ Σ_j p_j λ^{|k−j|}` (|k−j| ≤ 2) for Score; exact-sum fix-up;
confidence `σ(a·concentration + b·margin + c·evidence + d·ood + e)` with
`concentration = 1 − H(p)/log K` and `margin = p₁ − p₂`, fitted against
empirical correctness on the calibration split.

## 9. Contamination scanning

`sextant leakage` hashes every state and instruction+criteria text of the
training/calibration sets (raw and normalized), builds token 5-shingles and
128-permutation MinHash signatures with 16-band LSH, and reports exact,
normalized and near-duplicate (Jaccard ≥ 0.5) overlaps with the evaluation
sets. The verdict is driven by state text; instruction matches are listed for
manual review.
