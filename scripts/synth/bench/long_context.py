#!/usr/bin/env python3
"""Family 8: long-context wrappers around families 1-7.

An inner item from another family is embedded in a document of a target length
(64 ... 16384 whitespace tokens) padded with plausible same-domain material:
ticket histories, handbook sections, meeting notes, log excerpts, inventory
notes and policy boilerplate. Some padding paragraphs deliberately reuse the
vocabulary of the *wrong* options.

The decisive evidence sits at `start`, `middle`, `end`, or is `split` over two
or three separated sections. The placement is verified with the same function
`verify.py` uses, so the stored `evidence_position` is always re-derivable.
"""
from . import core
from . import intent_routing, facts_extraction, numeric_temporal, policy_rules
from . import adequacy_relevance, compatibility, ordinal

FAMILY = "long_context"

BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
POSITIONS = ["start", "middle", "end", "split"]
BUCKET_DIFFICULTY = {64: "easy", 128: "easy", 256: "standard", 512: "standard",
                     1024: "hard", 2048: "hard", 4096: "judge", 8192: "judge", 16384: "judge"}

INNER = [
    ("intent_routing", intent_routing),
    ("facts_extraction", facts_extraction),
    ("numeric_temporal", numeric_temporal),
    ("policy_rules", policy_rules),
    ("adequacy_relevance", adequacy_relevance),
    ("compatibility", compatibility),
    ("ordinal", ordinal),
]

DECOY_FRAMES = [
    "An earlier thread in this file discussed {kw} at some length without reaching a conclusion.",
    "The index card for this case lists {kw} among the topics touched on previously.",
    "A superseded note in the bundle is headed {kw}; it has been left in for completeness.",
    "Colleagues have asked about {kw} on three occasions this quarter.",
    "The archived correspondence mentions {kw} but relates to a different matter entirely.",
    "For historical reasons this file also carries material about {kw}.",
]
SPLIT_LINKS = [
    "The remainder of this matter is continued further down in this file.",
    "See also the later section of this bundle for the rest of the detail.",
    "The relevant facts are recorded in more than one place in this document.",
    "Further material on the same matter appears elsewhere in this bundle.",
]


PROSE_ADQ_INSTR = [
    "Does the agent's reply in this file resolve the customer's question?",
    "Read the customer question and the agent reply recorded in this file: has the request been dealt with in full?",
    "Is the reply recorded here a complete response to the question recorded here?",
    "Judging from this file alone, does the reply settle what the customer asked?",
]
PROSE_ADQ_CLS_INSTR = [
    "How does the agent reply recorded in this file relate to the customer question recorded in this file?",
    "Classify the recorded reply against the recorded question.",
    "Which description fits the reply held in this file?",
    "Judge the recorded reply and pick the matching description.",
]
PROSE_NLI_INSTR = [
    "Given the statement of record in this file, how should the claim under assessment be classified?",
    "Does the statement of record support, contradict or leave open the claim under assessment?",
    "Compare the claim under assessment against the statement of record.",
    "Read the statement of record. What does it imply about the claim under assessment?",
]
PROSE_NLI_NOUL_INSTR = [
    "Does the statement of record guarantee that the claim under assessment is true?",
    "Can the claim under assessment be concluded from the statement of record alone?",
    "Is the claim under assessment entailed by the statement of record?",
    "Reading only the statement of record, must the claim hold?",
]


def _proseify(rec, rng):
    """Turn a dict-state record into an equivalent prose record, or None."""
    st = rec["request"]["state"]
    if not isinstance(st, dict):
        return rec
    keys = set(st)
    qs = rec["request"]["questions"]
    if keys == {"question", "answer"}:
        text = core.sent_join(["Customer question on file: %s" % st["question"],
                               "Reply sent by the agent: %s" % st["answer"]])
        pool = PROSE_ADQ_CLS_INSTR if any(q["type"] == "choice" for q in qs.values()) \
            else PROSE_ADQ_INSTR
    elif keys == {"premise", "hypothesis"}:
        text = core.sent_join(["Statement of record: %s" % st["premise"],
                               "Claim under assessment: %s" % st["hypothesis"]])
        pool = PROSE_NLI_INSTR if any(q["type"] == "choice" for q in qs.values()) \
            else PROSE_NLI_NOUL_INSTR
    else:
        return None
    out = dict(rec)
    out["request"] = {"model": rec["request"]["model"], "state": text,
                      "questions": {k: dict(v, instructions=pool[rng.randrange(len(pool))])
                                    for k, v in qs.items()}}
    return out


def _split_sentences(text):
    out = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".!?" and len(buf.strip()) > 3:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def _decoy_words(inner):
    words = []
    for q in inner["request"]["questions"].values():
        crit = q.get("criteria")
        if isinstance(crit, dict):
            gold_vals = set(str(v) for v in inner["gold"].values())
            for k, v in crit.items():
                if k in gold_vals:
                    continue
                words.append(k.replace("_", " "))
                if isinstance(v, str):
                    toks = [t for t in v.split() if len(t) > 6]
                    if toks:
                        words.append(" ".join(toks[:2]).lower().strip(",.;"))
        elif isinstance(crit, list):
            for v in crit:
                toks = [t for t in str(v).split() if len(t) > 6]
                if toks:
                    words.append(toks[0].lower().strip(",.;"))
    return [w for w in words if w]


def _paragraph(ctx, rng, padfam, decoys, n_sent):
    sents = []
    for _ in range(n_sent):
        if decoys and rng.random() < 0.18:
            sents.append(ctx.pick_val(rng, "lc/decoy", DECOY_FRAMES).format(
                kw=decoys[rng.randrange(len(decoys))]))
        else:
            sents.append(ctx.pad_sentence(rng, padfam))
    title = ctx.pick_val(rng, "lc/titles", core.SECTION_TITLES)
    return "%s %d. %s" % (title, 1 + rng.randrange(40), core.sent_join(sents))


def _build_doc(ctx, rng, inner, target, position):
    padfam = inner["meta"].get("pad_family", "ticket")
    decoys = _decoy_words(inner)
    inner_text = inner["request"]["state"]
    sents = _split_sentences(inner_text)
    if position == "split":
        k = 3 if len(sents) >= 3 and rng.random() < 0.5 else 2
        chunks = []
        step = max(1, len(sents) // k)
        for i in range(k):
            part = sents[i * step:(i + 1) * step] if i < k - 1 else sents[i * step:]
            if part:
                chunks.append(core.sent_join(part))
        if len(chunks) < 2:
            return None
    else:
        chunks = [inner_text]
    # padding paragraphs
    pads = []
    tokens = sum(len(c.split()) for c in chunks)
    guard = 0
    while tokens < target and guard < 4000:
        base_n = max(1, min(6, target // 48))
        p = _paragraph(ctx, rng, padfam, decoys, base_n + rng.randrange(2))
        pads.append(p)
        tokens += len(p.split())
        guard += 1
    if len(pads) < len(chunks) + 1:
        pads.extend(_paragraph(ctx, rng, padfam, decoys, 2) for _ in range(len(chunks) + 1 - len(pads)))
    n = len(pads)
    if position == "start":
        cand = [0, 1, 2]
    elif position == "end":
        cand = [n, n - 1, n - 2]
    elif position == "middle":
        cand = [n // 2, n // 2 + 1, n // 2 - 1]
    else:
        cand = [None]
    if position == "split":
        spots = [max(0, int(n * f)) for f in (0.08, 0.5, 0.9)][:len(chunks)]
        parts = list(pads)
        for i, (sp, ch) in enumerate(sorted(zip(spots, chunks), reverse=True)):
            link = ctx.pick_val(rng, "lc/links", SPLIT_LINKS) if i else ""
            parts.insert(min(sp, len(parts)), core.sent_join([ch, link]))
        doc = "\n\n".join(parts)
        ev = chunks
        if core.evidence_position(doc, ev) != "split":
            return None
        return doc, ev
    for c in cand:
        if c is None or c < 0 or c > n:
            continue
        parts = list(pads)
        parts.insert(c, chunks[0])
        doc = "\n\n".join(parts)
        if core.evidence_position(doc, [chunks[0]]) == position:
            return doc, [chunks[0]]
    return None


def _inner_stream(ctx, name, mod):
    sub = core.Ctx(ctx.seed, ctx.side, "%s-lc-%s" % (ctx.stream, name))
    rng = ctx.rng("lc-prose", name)
    for recs in mod.scenarios(sub):
        for r in recs:
            r = _proseify(r, rng)
            if r is None:
                continue
            if isinstance(r["request"]["state"], str) and len(r["request"]["state"].split()) >= 6:
                ctx.template_ids.update(r["meta"].get("template_ids", []))
                yield r


def scenarios(ctx, limit=None, buckets=None, positions=None):
    """Yield one-record lists, cycling bucket x position x inner family."""
    buckets = buckets or BUCKETS
    positions = positions or POSITIONS
    streams = {name: _inner_stream(ctx, name, mod) for name, mod in INNER}
    order = [n for n, _ in INNER]
    emitted = 0
    i = 0
    while True:
        for bucket in buckets:
            for pos in positions:
                name = order[i % len(order)]
                i += 1
                made = False
                for attempt in range(12):
                    try:
                        inner = next(streams[name])
                    except StopIteration:
                        break
                    ctx.reset_tids()
                    rng = ctx.rng("lc", name, bucket, pos, i, attempt)
                    built = _build_doc(ctx, rng, inner, bucket, pos)
                    if built is None:
                        continue
                    doc, ev = built
                    rid = "%s-lc-%s-%d-%s-%d" % (ctx.stream, name, bucket, pos, i)
                    diff = BUCKET_DIFFICULTY[bucket]
                    rec = ctx.rec(rid, FAMILY, inner["recipe"], "lc_%s_%d" % (pos, bucket),
                                  "lc/%s/%d" % (inner["group"], bucket), doc,
                                  inner["request"]["questions"], inner["gold"], diff,
                                  "long_context", inner["meta"]["checks"],
                                  evidence=ev, position=pos)
                    rec["meta"]["template_ids"] = sorted(
                        set(rec["meta"]["template_ids"]) | set(inner["meta"].get("template_ids", [])))
                    rec["meta"]["base_family"] = inner["family"]
                    rec["meta"]["length_bucket"] = bucket
                    rec["meta"]["pad_family"] = inner["meta"].get("pad_family", "ticket")
                    emitted += 1
                    made = True
                    yield [rec]
                    break
                if not made:
                    continue
                if limit and emitted >= limit:
                    return
