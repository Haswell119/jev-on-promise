#!/usr/bin/env python3
"""Family 3: numeric and temporal reasoning.

Comparisons, range buckets, array counts, sums stated across separate
sentences, durations against windows with mixed units (hours / days / weeks /
months), relative dates anchored on a stated reference date, and thresholds
deliberately placed far away from the facts they apply to.
"""
from . import core

FAMILY = "numeric_temporal"

OPS = {
    "gt": ("greater than", lambda a, b: a > b),
    "ge": ("at least", lambda a, b: a >= b),
    "lt": ("less than", lambda a, b: a < b),
    "le": ("at most", lambda a, b: a <= b),
}
UNIT_HOURS = {"hour": 1, "day": 24, "week": 168, "month": 720}

AMOUNT_EDGES = [100, 1000, 10000, 100000]
AMOUNT_LEVELS = ["Under $100", "$100 to $1,000", "$1,000 to $10,000",
                 "$10,000 to $100,000", "Over $100,000"]
COUNT_EDGES = [1, 3, 6, 11]
COUNT_LEVELS = ["No line items", "A single line item", "Two line items",
                "Three to five line items", "Six to ten line items", "More than ten line items"]

FAR_FILLER = [
    "The account manager confirmed the contact details during the last review.",
    "All figures in this note are stated exclusive of tax unless marked otherwise.",
    "This section is reproduced verbatim from the previous quarter's pack.",
    "The finance team files a copy of every note with the monthly ledger.",
    "Questions about the formatting of this document go to the reporting desk.",
    "A scanned signature page is stored alongside the original submission.",
    "Rounding differences of less than one unit are ignored throughout.",
    "The reference data was refreshed before this note was assembled.",
    "Nothing in this paragraph changes any threshold stated elsewhere.",
    "The distribution list for this note has not changed since the spring.",
]

AMOUNT_FRAMES = [
    "The invoice raised against {company} comes to ${v}.",
    "{company} has been billed ${v} for the period.",
    "Total payable by {company}: ${v}.",
    "The settlement figure agreed with {company} is ${v}.",
    "An amount of ${v} is outstanding on the {company} account.",
    "{company} owes ${v} under the current arrangement.",
]
THRESHOLD_FRAMES = [
    "Any amount {op} ${t} must be countersigned by a second approver.",
    "Escalate to the finance director when the figure is {op} ${t}.",
    "The delegated authority limit applies to sums {op} ${t}.",
    "A sum {op} ${t} triggers the secondary review path.",
    "Approvals are required whenever the value is {op} ${t}.",
    "The board is notified of anything {op} ${t}.",
]
LINE_FRAMES = [
    "The first line of the claim covers {d} at ${v}.",
    "A second entry records {d} costing ${v}.",
    "There is a further charge of ${v} for {d}.",
    "The item {d} was added later at a cost of ${v}.",
    "The remaining item, {d}, is priced at ${v}.",
    "An amount of ${v} is listed for {d}.",
]
DURATION_FRAMES = [
    "The repair on the {product} took {n} {u}s from drop-off to collection.",
    "Work on the {product} ran for {n} {u}s in total.",
    "The turnaround for the {product} was {n} {u}s.",
    "It took {n} {u}s to complete the job on the {product}.",
    "The engineer spent {n} {u}s on the {product}.",
    "From start to finish the {product} job lasted {n} {u}s.",
]
WINDOW_FRAMES = [
    "The service commitment for this class of work is {m} {w}s.",
    "Policy allows {m} {w}s for this type of job.",
    "The agreed window is {m} {w}s.",
    "Work of this kind must finish within {m} {w}s.",
    "The contractual limit for this job type is {m} {w}s.",
    "Jobs like this are expected to close inside {m} {w}s.",
]


def _pick(ctx, rng, key, pool):
    return ctx.pick_val(rng, key, pool)


def _amount_scenario(ctx, rng, rid, group):
    recs = []
    company = ctx.slot(rng, "company")
    v = ctx.slot(rng, "amount")
    sent = _pick(ctx, rng, "nt/amount", AMOUNT_FRAMES).format(company=company, v="%.2f" % v)
    # range bucket
    q = {"type": "score", "instructions": "How large is the amount stated?", "criteria": AMOUNT_LEVELS}
    recs.append(ctx.rec(rid + "-bucket", FAMILY, "numeric_temporal", "range_bucket", group,
                        sent, {"size": q}, {"size": _bucket(v, AMOUNT_EDGES)}, "easy", "numeric",
                        [{"qid": "size", "rule": "bucket", "args": {"value": v, "edges": AMOUNT_EDGES}}],
                        evidence=[sent]))
    # direct comparison
    opk = rng.choice(sorted(OPS))
    t = ctx.slot(rng, "amount")
    q2 = {"type": "noul", "instructions": "Is the amount stated %s $%s?" % (OPS[opk][0], "{:,.2f}".format(t))}
    recs.append(ctx.rec(rid + "-cmp", FAMILY, "numeric_temporal", "comparison", group,
                        sent, {"cmp": q2}, {"cmp": OPS[opk][1](v, t)}, "easy", "numeric",
                        [{"qid": "cmp", "rule": "cmp_num", "args": {"lhs": v, "op": opk, "rhs": t}}],
                        evidence=[sent]))
    # threshold far from the fact
    opk2 = rng.choice(sorted(OPS))
    t2 = ctx.slot(rng, "amount")
    thr = _pick(ctx, rng, "nt/threshold", THRESHOLD_FRAMES).format(op=OPS[opk2][0], t="{:,.2f}".format(t2))
    filler = [_pick(ctx, rng, "nt/filler", FAR_FILLER) for _ in range(6)]
    far = core.sent_join([thr] + filler + [sent])
    q3 = {"type": "noul", "instructions": "Does the amount stated in this note meet the condition that triggers the extra approval step?"}
    recs.append(ctx.rec(rid + "-far", FAMILY, "numeric_temporal", "distant_threshold", group,
                        far, {"triggers": q3}, {"triggers": OPS[opk2][1](v, t2)}, "hard", "multi_hop",
                        [{"qid": "triggers", "rule": "cmp_num", "args": {"lhs": v, "op": opk2, "rhs": t2}}],
                        evidence=[thr, sent]))
    return recs


def _sum_scenario(ctx, rng, rid, group):
    recs = []
    parts = []
    vals = []
    for i in range(3):
        v = round(ctx.slot(rng, "amount") / (3 + i), 2)
        vals.append(v)
        parts.append(_pick(ctx, rng, "nt/line", LINE_FRAMES).format(
            d=ctx.slot(rng, "product"), v="%.2f" % v))
    filler = [_pick(ctx, rng, "nt/filler", FAR_FILLER) for _ in range(2)]
    text = core.sent_join([parts[0], filler[0], parts[1], filler[1], parts[2]])
    total = round(sum(vals), 2)
    opk = rng.choice(sorted(OPS))
    t = round(ctx.slot(rng, "amount"), 2)
    q = {"type": "noul", "instructions": "Taking all the charges in this note together, is the total %s $%s?"
         % (OPS[opk][0], "{:,.2f}".format(t))}
    recs.append(ctx.rec(rid + "-sum", FAMILY, "numeric_temporal", "sum_across_sentences", group,
                        text, {"total": q}, {"total": OPS[opk][1](total, t)}, "hard", "numeric",
                        [{"qid": "total", "rule": "sum_cmp",
                          "args": {"addends": vals, "op": opk, "rhs": t}}],
                        evidence=parts))
    q2 = {"type": "score", "instructions": "How large is the combined value of all charges in this note?",
          "criteria": AMOUNT_LEVELS}
    recs.append(ctx.rec(rid + "-sumbucket", FAMILY, "numeric_temporal", "sum_bucket", group,
                        text, {"size": q2}, {"size": _bucket(total, AMOUNT_EDGES)}, "hard", "numeric",
                        [{"qid": "size", "rule": "sum_bucket",
                          "args": {"addends": vals, "edges": AMOUNT_EDGES}}],
                        evidence=parts))
    return recs


def _count_scenario(ctx, rng, rid, group):
    recs = []
    n = rng.randrange(0, 13)
    items = [{"sku": "SKU-%d" % (100 + ctx.slot(rng, "small")),
              "description": ctx.slot(rng, "product"),
              "qty": 1 + ctx.slot(rng, "small") % 4} for _ in range(n)]
    state = {"claim": {"reference": ctx.ticket_id(rng), "customer": ctx.slot(rng, "company"),
                       "lines": items, "handler": ctx.slot(rng, "agent")}}
    thr = rng.randrange(1, 8)
    opk = rng.choice(sorted(OPS))
    q = {"type": "noul", "instructions": "Does `claim.lines` contain %s %d entries?" % (OPS[opk][0], thr)}
    recs.append(ctx.rec(rid + "-count", FAMILY, "numeric_temporal", "array_count", group,
                        state, {"count": q}, {"count": OPS[opk][1](n, thr)}, "standard", "numeric",
                        [{"qid": "count", "rule": "count_cmp", "args": {"count": n, "op": opk, "rhs": thr}}],
                        evidence=[]))
    q2 = {"type": "score", "instructions": "How many line items does the claim carry?", "criteria": COUNT_LEVELS}
    recs.append(ctx.rec(rid + "-countbucket", FAMILY, "numeric_temporal", "count_bucket", group,
                        state, {"how_many": q2}, {"how_many": _bucket(n, COUNT_EDGES)}, "standard", "numeric",
                        [{"qid": "how_many", "rule": "count_bucket", "args": {"count": n, "edges": COUNT_EDGES}}],
                        evidence=[]))
    return recs


def _duration_scenario(ctx, rng, rid, group):
    recs = []
    product = ctx.slot(rng, "product")
    u = rng.choice(["hour", "day", "week", "month"])
    w = rng.choice([x for x in ["hour", "day", "week", "month"] if x != u])
    n = 1 + ctx.slot(rng, "small") % 20
    m = 1 + ctx.slot(rng, "small") % 20
    d_sent = _pick(ctx, rng, "nt/dur", DURATION_FRAMES).format(n=n, u=u, product=product)
    w_sent = _pick(ctx, rng, "nt/win", WINDOW_FRAMES).format(m=m, w=w)
    filler = [_pick(ctx, rng, "nt/filler", FAR_FILLER) for _ in range(3)]
    text = core.sent_join([w_sent] + filler + [d_sent])
    within = n * UNIT_HOURS[u] <= m * UNIT_HOURS[w]
    q = {"type": "noul", "instructions": "Did the job finish inside the allowed window?",
         "criteria": {"true": "The elapsed time is within the stated window",
                      "false": "The elapsed time exceeds the stated window"}}
    recs.append(ctx.rec(rid + "-dur", FAMILY, "numeric_temporal", "duration_vs_window", group,
                        text, {"within": q}, {"within": within}, "hard", "temporal",
                        [{"qid": "within", "rule": "duration_cmp",
                          "args": {"left": {"value": n, "unit": u}, "op": "le",
                                   "right": {"value": m, "unit": w}}}],
                        evidence=[w_sent, d_sent]))
    q2 = {"type": "choice", "instructions": "How does the elapsed time compare with the allowed window?",
          "criteria": {"within_window": "The job finished inside the allowed window",
                       "over_window": "The job took longer than the allowed window"}}
    recs.append(ctx.rec(rid + "-durchoice", FAMILY, "numeric_temporal", "duration_choice", group,
                        text, {"verdict": q2},
                        {"verdict": "within_window" if within else "over_window"}, "hard", "temporal",
                        [{"qid": "verdict", "rule": "duration_choice",
                          "args": {"left": {"value": n, "unit": u}, "right": {"value": m, "unit": w},
                                   "within_key": "within_window", "over_key": "over_window"}}],
                        evidence=[w_sent, d_sent]))
    return recs


def _relative_date_scenario(ctx, rng, rid, group):
    recs = []
    ref = ctx.date(rng)
    base = ctx.date(rng)
    offset = rng.choice([7, 14, 21, 28, 30, 45, 60, 90])
    unit = rng.choice(["day", "week", "month"])
    shown = {"day": offset, "week": max(1, offset // 7), "month": max(1, offset // 30)}[unit]
    real_offset = shown * {"day": 1, "week": 7, "month": 30}[unit]
    s_ref = "For the purposes of this review the reference date is %s." % core.pretty_date(ref)
    s_base = "The inspection of the %s took place on %s." % (ctx.slot(rng, "product"), core.pretty_date(base))
    s_off = "The certificate issued at the inspection remains valid for %d %ss." % (shown, unit)
    filler = [_pick(ctx, rng, "nt/filler", FAR_FILLER) for _ in range(3)]
    text = core.sent_join([s_ref, filler[0], s_base, filler[1], s_off, filler[2]])
    expiry = core.date_add(base, real_offset)
    expired = expiry < ref
    q = {"type": "noul", "instructions": "Has the certificate expired as of the reference date?"}
    recs.append(ctx.rec(rid + "-rel", FAMILY, "numeric_temporal", "relative_date", group,
                        text, {"expired": q}, {"expired": expired}, "hard", "temporal",
                        [{"qid": "expired", "rule": "date_offset_cmp",
                          "args": {"base": base, "offset_days": real_offset, "op": "lt", "ref": ref}}],
                        evidence=[s_ref, s_base, s_off]))
    other = ctx.date(rng)
    q2 = {"type": "noul", "instructions": "Did the inspection take place before %s?" % core.pretty_date(other)}
    recs.append(ctx.rec(rid + "-datecmp", FAMILY, "numeric_temporal", "date_comparison", group,
                        text, {"before": q2}, {"before": base < other}, "standard", "temporal",
                        [{"qid": "before", "rule": "date_cmp", "args": {"left": base, "op": "lt", "right": other}}],
                        evidence=[s_base]))
    return recs


def _bucket(v, edges):
    i = 0
    for e in edges:
        if v < e:
            return i
        i += 1
    return i


SCENARIOS = [("amount", _amount_scenario), ("sum", _sum_scenario), ("count", _count_scenario),
             ("duration", _duration_scenario), ("reldate", _relative_date_scenario)]


def scenarios(ctx, limit=None):
    emitted = 0
    rep = 0
    while True:
        for name, fn in SCENARIOS:
            ctx.reset_tids()
            rng = ctx.rng("nt", name, rep)
            group = "nt/%s/%d" % (name, rep)
            rid = "%s-nt-%s-%d" % (ctx.stream, name, rep)
            recs = fn(ctx, rng, rid, group)
            tids = sorted(set(ctx.cur_tids))
            for r in recs:
                r["meta"]["template_ids"] = sorted(set(tids + r["meta"]["template_ids"]))
                r["meta"]["pad_family"] = "meeting"
            emitted += len(recs)
            yield recs
            if limit and emitted >= limit:
                return
        rep += 1
        if rep > 6000:
            return
