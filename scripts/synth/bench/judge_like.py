#!/usr/bin/env python3
"""Family 9: judge-like tasks.

(a) Capability routing: technical requests must be sent to a handler whose
    option description states only what the handler *can do* - never the words
    the request itself uses.
(b) Worked-answer adjudication: an arithmetic word problem or an ordering
    puzzle is generated together with a short worked answer that is either
    correct or wrong in one specific, programmatically known way.
"""
from . import core

FAMILY = "judge_like"

HANDLERS = {
 "query_specialist": ("Can read a relational schema, write and tune queries and explain why a "
                      "result set looks the way it does",
  ["The monthly revenue figure on the finance page counts the same order twice whenever a customer "
   "has two addresses on file.",
   "We need the top fifty accounts by spend in the last quarter, excluding internal test accounts.",
   "A report that used to finish in seconds now takes four minutes after the table grew.",
   "Two dashboards disagree about how many active customers we have and nobody can say which is right.",
   "Someone needs to work out why the join between the orders and refunds tables loses rows.",
   "We want a weekly figure for repeat purchases, defined consistently across all regions."]),
 "pipeline_engineer": ("Can build and repair scheduled data movements between systems, including "
                       "backfills and recovery after a failed run",
  ["The nightly load finished with an error at 02:10 and yesterday's numbers are missing everywhere.",
   "We changed the upstream file layout and everything downstream now fails silently.",
   "Six weeks of history need to be reprocessed after the mapping was corrected.",
   "The scheduled job ran twice last night and duplicated every row it wrote.",
   "A new source system has to be brought in on the same nightly cadence as the others.",
   "Whenever the source is late the whole chain fails instead of waiting."]),
 "interface_developer": ("Can change what a person sees and clicks in the browser: layout, styling "
                         "and behaviour on the page itself",
  ["On a narrow screen the checkout button sits underneath the footer and cannot be pressed.",
   "The date picker lets people choose a day in the past even though the form rejects it later.",
   "Text in the summary panel overflows its box in the German translation.",
   "Pressing enter in the search field reloads the page instead of running the search.",
   "The loading spinner never disappears once the results have come back.",
   "Colour contrast on the warning banner fails the accessibility check."]),
 "operations_oncall": ("Can change how much server capacity is running, restart or drain services "
                       "and put a previous release back in place",
  ["Response times tripled after the 14:00 release and are still climbing.",
   "One instance is serving errors while the other three are healthy.",
   "Memory on the workers grows until they are killed roughly every two hours.",
   "The queue has two hundred thousand messages and the consumers cannot keep up.",
   "We need last week's version back in production while the cause is investigated.",
   "Traffic doubled unexpectedly and the autoscaler is not reacting quickly enough."]),
 "security_reviewer": ("Can judge who could reach what, whether a secret is exposed and what must "
                       "be rotated or restricted",
  ["A configuration file with a live database password was committed to a public repository.",
   "Contractors who left in March can still open the shared drive.",
   "An endpoint returns another customer's record if you change the number in the URL.",
   "Logs are storing the full card number in plain text for every transaction.",
   "A third-party script was added to the checkout page without review.",
   "The same administrative credential is shared by nine people."]),
 "evaluation_designer": ("Can design how a system's output quality is measured and say what the "
                         "resulting numbers do and do not mean",
  ["We want to know whether the new ranking is actually better before turning it on for everyone.",
   "Our accuracy number went up but complaints went up too, and we cannot explain it.",
   "Someone has to decide what counts as a correct answer before we can score anything.",
   "The sample we score each week is drawn from whatever is convenient, which worries me.",
   "Two teams report different quality figures for the same component.",
   "We need a way to tell a real improvement from noise between two versions."]),
 "release_engineer": ("Can change how software is built, versioned and published to the places "
                      "that consume it",
  ["The build succeeds locally and fails on the shared runner with a missing dependency.",
   "Two packages were published with the same version number an hour apart.",
   "The artefact for last Tuesday cannot be reproduced from the tag it claims to come from.",
   "Every merge triggers the full three-hour job even when only documentation changed.",
   "Consumers are picking up a pre-release build because the channel is misconfigured.",
   "The signing step is skipped whenever the pipeline is triggered manually."]),
 "technical_writer": ("Can produce clear prose for other people to read and keep it in step with "
                      "what the system actually does",
  ["New starters keep asking the same five questions because nothing is written down.",
   "The published guide still describes the flow we replaced two releases ago.",
   "Support keeps pasting the same long explanation into tickets by hand.",
   "The parameter list in the reference page is missing three of the options.",
   "Customers follow the setup steps in order and end up with a broken configuration.",
   "Two pages give contradictory instructions for the same task."]),
 "performance_analyst": ("Can find out why something takes as long or uses as much as it does, and "
                         "reduce it",
  ["Opening a large account takes eleven seconds and users are abandoning the page.",
   "The same request costs four times as much processing as it did before the upgrade.",
   "One function appears in ninety percent of the sampled stacks.",
   "The export uses so much memory that the machine starts swapping.",
   "Cold starts add two seconds to every first request after a quiet period.",
   "The batch takes eight hours and must finish inside four."]),
 "contract_designer": ("Can shape the agreement between two systems: what is called, what comes "
                       "back and how changes are introduced without breaking callers",
  ["Callers break whenever we add a field, because they validate strictly.",
   "The same concept is called three different things in three different endpoints.",
   "We need to remove a field that two partners still depend on.",
   "Errors come back as free text, so nobody can handle them automatically.",
   "Pagination works differently on every list endpoint.",
   "There is no way for a caller to tell a retryable failure from a permanent one."]),
}

ROUTE_FRAMES = [
    "{req}",
    "Ticket {ref} raised by {person}: {req}",
    "{req} Reported on {date} by {agent}.",
    "Incoming work item {ref}. {req}",
    "From the {channel} queue: {req} No owner assigned yet.",
    "{req} This came in from the {city} site.",
    "Triage note for {ref}: {req}",
    "{agent} writes: {req} Please assign an owner.",
]

ROUTE_INSTR = [
    "Which handler has the capability this request needs?",
    "Route the request to the handler whose stated capability covers it.",
    "Pick the handler able to deal with what is described.",
    "Which of these handlers should take this piece of work?",
]

# arithmetic word problems: (text template, expression template, slot names)
ARITH = [
    ("The {city} depot ships {a} crates a day for {b} days, then receives {c} extra crates. "
     "How many crates does it handle in total?", "{a}*{b}+{c}"),
    ("A team of {a} people each log {b} hours a week. {c} hours are reassigned to another project. "
     "How many hours remain on this project in a week?", "{a}*{b}-{c}"),
    ("An invoice of ${a} is split equally between {b} departments, and each department then adds "
     "${c} of its own costs. What does each department pay?", "{a}/{b}+{c}"),
    ("A warehouse holds {a} units. {b} units are shipped out and {c} units arrive. "
     "How many units are in stock afterwards?", "{a}-{b}+{c}"),
    ("A course runs for {a} weeks with {b} sessions a week, and {c} sessions are cancelled. "
     "How many sessions take place?", "{a}*{b}-{c}"),
    ("A subscription costs ${a} a month. After {b} months the price rises by ${c} a month. "
     "What is the cost of the {b}th month plus the month after it?", "{a}+{a}+{c}"),
    ("{a} tickets arrive on Monday and {b} on Tuesday. {c} of them are duplicates and are merged. "
     "How many distinct tickets remain?", "{a}+{b}-{c}"),
    ("A van covers {a} kilometres on each of {b} runs and then {c} kilometres returning to base. "
     "How far does it travel altogether?", "{a}*{b}+{c}"),
]
ERROR_KINDS = ["correct", "arithmetic_slip", "wrong_operation", "misread_quantity"]
ERROR_DESC = {
    "correct": "The working and the final answer are both right",
    "arithmetic_slip": "The method is right but one calculation is wrong",
    "wrong_operation": "The wrong operation was applied to the numbers",
    "misread_quantity": "A quantity from the problem was read incorrectly",
}
WORKED = [
    "First {s1}. Then {s2}. The answer is {ans}.",
    "Step one: {s1}. Step two: {s2}. So the answer is {ans}.",
    "{s1}, and then {s2}. That gives {ans}.",
    "Working: {s1}; {s2}. Final answer: {ans}.",
]
JUDGE_INSTR = [
    "Is the final answer in `answer` correct for `problem`?",
    "Check the working: does `answer` reach the right result for `problem`?",
    "Decide whether the result given in `answer` is right.",
    "Verify the arithmetic in `answer` against `problem`.",
]
FAULT_INSTR = [
    "What, if anything, is wrong with the working in `answer`?",
    "Diagnose `answer` against `problem`.",
    "Which description fits the working shown in `answer`?",
]
PEOPLE_A = ["Ana", "Bo", "Cai", "Dev", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jo", "Kit", "Lou"]
CHAIN_REL = [
    ("finished ahead of", "in the running order"),
    ("is taller than", "by height"),
    ("scored higher than", "in the assessment"),
    ("was hired before", "by start date"),
    ("costs more than", "by price"),
    ("is closer to the depot than", "by distance"),
]
CHAIN_INSTR = [
    "Does the claim in `claim` follow from the statements in `facts`?",
    "Using only `facts`, is `claim` established?",
    "Can `claim` be derived from `facts` alone?",
]
CHAIN3_INSTR = [
    "How does `claim` stand against `facts`?",
    "Classify `claim` given `facts`.",
    "What do `facts` say about `claim`?",
]
CHAIN3_CRIT = {
    "follows": "The facts establish the claim",
    "contradicted": "The facts establish the opposite of the claim",
    "unknown": "The facts neither establish the claim nor its opposite",
}


def _route(ctx, rng, rid, group):
    keys = sorted(HANDLERS)
    gold = ctx.pick_val(rng, "jl/handlers", keys)
    desc, reqs = HANDLERS[gold]
    core_req = ctx.pick_val(rng, "jl/req/%s" % gold, reqs)
    req = ctx.pick_val(rng, "jl/rframes", ROUTE_FRAMES).format(
        req=core_req, ref=ctx.ticket_id(rng), person=ctx.slot(rng, "person"),
        date=core.pretty_date(ctx.date(rng)), agent=ctx.slot(rng, "agent"),
        channel=ctx.slot(rng, "channel"), city=ctx.slot(rng, "city"))
    others = [k for k in keys if k != gold]
    rng.shuffle(others)
    opts = others[:rng.choice([3, 4, 5])] + [gold]
    rng.shuffle(opts)
    crit = {k: HANDLERS[k][0] for k in opts}
    q = {"type": "choice", "instructions": ctx.pick_val(rng, "jl/instr", ROUTE_INSTR),
         "criteria": crit}
    return [ctx.rec(rid + "-route", FAMILY, "judge_like", "capability_routing", group,
                    req, {"handler": q}, {"handler": gold}, "judge", "implicit",
                    [{"qid": "handler", "rule": "choice_const",
                      "args": {"value": gold, "options": sorted(crit)}}], evidence=[core_req])]


def _safe_eval(expr):
    allowed = set("0123456789+-*/(). ")
    assert set(expr) <= allowed, expr
    return eval(expr, {"__builtins__": {}}, {})


def _arith(ctx, rng, rid, group):
    text_t, expr_t = ctx.pick_val(rng, "jl/arith", ARITH)
    a = 3 + ctx.slot(rng, "small") % 24
    b = 2 + ctx.slot(rng, "small") % 9
    c = 1 + ctx.slot(rng, "small") % 30
    if "/" in expr_t:
        a = b * (2 + ctx.slot(rng, "small") % 12)
    problem = text_t.format(a=a, b=b, c=c, city=ctx.slot(rng, "city"))
    expr = expr_t.format(a=a, b=b, c=c)
    true_val = _safe_eval(expr)
    assert abs(true_val - round(true_val)) < 1e-9, expr
    kind = rng.choice(ERROR_KINDS)
    if kind == "correct":
        claimed = true_val
    elif kind == "arithmetic_slip":
        claimed = true_val + rng.choice([-3, -2, -1, 1, 2, 4])
    elif kind == "wrong_operation":
        swapped = expr.replace("*", "#").replace("+", "*").replace("#", "+") \
            if "*" in expr and "+" in expr else expr.replace("-", "+") if "-" in expr else expr + "+1"
        claimed = _safe_eval(swapped)
        if abs(claimed - true_val) < 1e-9:
            claimed = true_val + 5
    else:
        claimed = _safe_eval(expr_t.format(a=a + 10, b=b, c=c))
        if abs(claimed - true_val) < 1e-9:
            claimed = true_val + 7
    fmt = (lambda v: ("%g" % v))
    s1 = "the first part gives %s" % fmt(_safe_eval(expr.split("+")[0].split("-")[0]))
    s2 = "combining with the remaining figure gives %s" % fmt(claimed)
    worked = ctx.pick_val(rng, "jl/worked", WORKED).format(s1=s1, s2=s2, ans=fmt(claimed))
    st = {"problem": problem, "answer": worked}
    recs = []
    q = {"type": "noul", "instructions": ctx.pick_val(rng, "jl/judge", JUDGE_INSTR),
         "criteria": {"true": "The final number is correct", "false": "The final number is wrong"}}
    recs.append(ctx.rec(rid + "-arith", FAMILY, "judge_like", "arith_" + kind, group, st,
                        {"correct": q}, {"correct": kind == "correct"}, "judge", "numeric",
                        [{"qid": "correct", "rule": "arith",
                          "args": {"expr": expr, "claimed": claimed}}], evidence=[]))
    crit = core.shuffled_map(rng, {k: ERROR_DESC[k] for k in ERROR_KINDS})
    q2 = {"type": "choice", "instructions": ctx.pick_val(rng, "jl/fault", FAULT_INSTR),
          "criteria": crit}
    recs.append(ctx.rec(rid + "-fault", FAMILY, "judge_like", "fault_" + kind, group, st,
                        {"fault": q2}, {"fault": kind}, "judge", "adequacy",
                        [{"qid": "fault", "rule": "arith_fault",
                          "args": {"expr": expr, "claimed": claimed, "kind": kind,
                                   "options": sorted(crit)}}], evidence=[]))
    return recs


def _chain(ctx, rng, rid, group):
    rel, qual = ctx.pick_val(rng, "jl/chainrel", CHAIN_REL)
    names = [ctx.pick_val(rng, "jl/people", PEOPLE_A) for _ in range(8)]
    names = list(dict.fromkeys(names))
    while len(names) < 5:
        names.append("%s." % chr(ord("M") + len(names)))
    chain1 = names[:3]
    chain2 = names[3:5]
    facts = ["%s %s %s %s." % (chain1[i], rel, chain1[i + 1], qual) for i in range(len(chain1) - 1)]
    facts.append("%s %s %s %s." % (chain2[0], rel, chain2[1], qual))
    rng.shuffle(facts)
    relations = [[chain1[i], ">", chain1[i + 1]] for i in range(len(chain1) - 1)]
    relations.append([chain2[0], ">", chain2[1]])
    ctype = rng.choice(["follows", "contradicted", "unknown"])
    if ctype == "follows":
        x, y = chain1[0], chain1[2]
    elif ctype == "contradicted":
        x, y = chain1[2], chain1[0]
    else:
        x, y = chain1[0], chain2[1]
    claim = "%s %s %s %s." % (x, rel, y, qual)
    st = {"facts": facts, "claim": claim}
    recs = []
    q = {"type": "noul", "instructions": ctx.pick_val(rng, "jl/chain", CHAIN_INSTR)}
    recs.append(ctx.rec(rid + "-chain", FAMILY, "judge_like", "chain_" + ctype, group, st,
                        {"follows": q}, {"follows": ctype == "follows"}, "hard", "multi_hop",
                        [{"qid": "follows", "rule": "chain_bool",
                          "args": {"relations": relations, "claim": [x, ">", y]}}], evidence=[]))
    crit = core.shuffled_map(rng, dict(CHAIN3_CRIT))
    q2 = {"type": "choice", "instructions": ctx.pick_val(rng, "jl/chain3", CHAIN3_INSTR),
          "criteria": crit}
    recs.append(ctx.rec(rid + "-chain3", FAMILY, "judge_like", "chain3_" + ctype, group, st,
                        {"verdict": q2}, {"verdict": ctype}, "judge", "multi_hop",
                        [{"qid": "verdict", "rule": "chain3",
                          "args": {"relations": relations, "claim": [x, ">", y],
                                   "options": sorted(crit)}}], evidence=[]))
    return recs


PARTS = [("route", _route), ("arith", _arith), ("chain", _chain)]


def scenarios(ctx, limit=None):
    emitted = 0
    rep = 0
    while True:
        for name, fn in PARTS:
            for k in range(2 if name == "route" else 1):
                ctx.reset_tids()
                rng = ctx.rng("jl", name, k, rep)
                group = "jl/%s/%d/%d" % (name, k, rep)
                rid = "%s-jl-%s-%d-%d" % (ctx.stream, name, k, rep)
                recs = fn(ctx, rng, rid, group)
                tids = sorted(set(ctx.cur_tids))
                for r in recs:
                    r["meta"]["template_ids"] = sorted(set(tids + r["meta"]["template_ids"]))
                    r["meta"]["pad_family"] = "log"
                emitted += len(recs)
                yield recs
                if limit and emitted >= limit:
                    return
        rep += 1
        if rep > 8000:
            return
