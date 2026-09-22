#!/usr/bin/env python3
"""Family 6: premise / hypothesis compatibility, built programmatically.

entailment    paraphrase (verb synonym + reordering), hypernym generalisation,
              quantity relaxation ("at least n-k")
contradiction negation, antonym, changed number, different entity (the premise
              is made exclusive so a different actor really is impossible)
neutral       unrelated addition, unstated detail
"""
from . import core

FAMILY = "compatibility"

# verb, synonym, negation, antonym
VERBS = [
    ("shipped", "dispatched", "did not ship", "held back"),
    ("approved", "signed off", "did not approve", "rejected"),
    ("delivered", "handed over", "did not deliver", "returned"),
    ("installed", "fitted", "did not install", "removed"),
    ("ordered", "purchased", "did not order", "cancelled"),
    ("inspected", "examined", "did not inspect", "skipped"),
    ("repaired", "fixed", "did not repair", "scrapped"),
    ("logged", "recorded", "did not log", "deleted"),
    ("renewed", "extended", "did not renew", "terminated"),
    ("accepted", "took on", "did not accept", "declined"),
    ("processed", "worked through", "did not process", "set aside"),
    ("collected", "picked up", "did not collect", "left behind"),
]
# plural object, hypernym plural
OBJECTS = [
    ("coffee grinders", "kitchen appliances"), ("winter jackets", "items of clothing"),
    ("laptop stands", "desk accessories"), ("garden hoses", "garden supplies"),
    ("office chairs", "pieces of furniture"), ("smart thermostats", "home devices"),
    ("camping tents", "outdoor goods"), ("label printers", "office machines"),
    ("cordless drills", "power tools"), ("air purifiers", "household appliances"),
    ("bike helmets", "safety items"), ("water filters", "household goods"),
    ("desk lamps", "light fittings"), ("yoga mats", "fitness products"),
    ("electric kettles", "kitchen appliances"), ("travel backpacks", "luggage items"),
    ("monitor arms", "desk accessories"), ("paper shredders", "office machines"),
    ("induction hobs", "kitchen appliances"), ("sewing machines", "household machines"),
]
ACTORS = [
    "the north depot", "the riverside plant", "the east warehouse", "the harbour office",
    "the night shift", "the returns team", "the field crew", "the packing line",
    "the central store", "the service desk", "the dispatch office", "the quality team",
]
UNRELATED = [
    "the depot manager was on leave that week",
    "the yard was resurfaced during the same period",
    "a new shift pattern started the following month",
    "the canteen changed supplier at the end of the quarter",
    "two agency staff joined the team that morning",
    "the weather closed the northern road for a day",
]
UNSTATED = [
    "the consignment was insured for its full value",
    "the paperwork was countersigned by a supervisor",
    "the items were packed in recycled cardboard",
    "a photograph was taken before the handover",
    "the batch came from the same production run",
    "the customer had asked for this in advance",
]
PREM_FRAMES = [
    "On {day} {actor} {verb} {qty} {obj} in total.",
    "{Actor} {verb} {qty} {obj} in total on {day}.",
    "The record shows that on {day}, {actor} {verb} {qty} {obj} in total.",
    "According to the log, {actor} {verb} {qty} {obj} in total on {day}.",
    "During {day}, {actor} {verb} a total of {qty} {obj}.",
    "It is documented that {actor} {verb} {qty} {obj} in total on {day}.",
]
EXCLUSIVE_SUFFIX = [
    "No other site moved anything that day.",
    "Nowhere else in the network did so on that date.",
    "No other team was involved on that day.",
    "That was the only site to do so on the day in question.",
]
NLI_INSTR = [
    "Given `premise`, how should `hypothesis` be classified?",
    "Does `premise` support, contradict or leave open `hypothesis`?",
    "Compare `hypothesis` against `premise` and classify the relationship.",
    "Read `premise`. What does it imply about `hypothesis`?",
]
NOUL_INSTR = [
    "Does `premise` guarantee that `hypothesis` is true?",
    "Can `hypothesis` be concluded from `premise` alone?",
    "Is `hypothesis` entailed by `premise`?",
    "Reading only `premise`, must `hypothesis` hold?",
]
CRIT = {
    "entailment": "Reading the premise guarantees that the hypothesis holds.",
    "neutral": "The premise neither confirms nor rules out the hypothesis.",
    "contradiction": "The premise makes the hypothesis impossible.",
}


def scenarios(ctx, limit=None):
    emitted = 0
    rep = 0
    while True:
        for slot in range(len(VERBS)):
            ctx.reset_tids()
            rng = ctx.rng("nli", slot, rep)
            verb, syn, neg, ant = ctx.pick_val(rng, "nli/verbs", VERBS)
            obj, hyper = ctx.pick_val(rng, "nli/objects", OBJECTS)
            actor = ctx.pick_val(rng, "nli/actors", ACTORS)
            other_actor = ctx.pick_val(rng, "nli/actors", ACTORS)
            while other_actor == actor:
                other_actor = ctx.pick_val(rng, "nli/actors", ACTORS)
            day = ctx.slot(rng, "weekday")
            qty = 6 + ctx.slot(rng, "small") % 40
            frame = ctx.pick_val(rng, "nli/prem", PREM_FRAMES)
            unrel = ctx.pick_val(rng, "nli/unrel", UNRELATED)
            premise = frame.format(day=day, actor=actor, Actor=actor[0].upper() + actor[1:],
                                   verb=verb, qty=qty, obj=obj)
            excl = ctx.pick_val(rng, "nli/excl", EXCLUSIVE_SUFFIX)
            relax = max(1, qty - rng.choice([1, 2, 3, 5]))
            changed = qty + rng.choice([-4, -3, -2, 2, 3, 7])
            pairs = [
                ("entailment", "paraphrase",
                 "%s %s %d %s on %s." % (actor[0].upper() + actor[1:], syn, qty, obj, day), premise),
                ("entailment", "hypernym",
                 "%s %s %d %s on %s." % (actor[0].upper() + actor[1:], verb, qty, hyper, day), premise),
                ("entailment", "quantity_relaxation",
                 "%s %s at least %d %s on %s." % (actor[0].upper() + actor[1:], verb, relax, obj, day), premise),
                ("contradiction", "negation",
                 "%s %s any %s on %s." % (actor[0].upper() + actor[1:], neg, obj, day), premise),
                ("contradiction", "antonym",
                 "%s %s the %s on %s." % (actor[0].upper() + actor[1:], ant, obj, day), premise),
                ("contradiction", "changed_number",
                 "%s %s %d %s in total on %s." % (actor[0].upper() + actor[1:], verb, changed, obj, day), premise),
                ("contradiction", "different_entity",
                 "%s %s %d %s on %s." % (other_actor[0].upper() + other_actor[1:], verb, qty, obj, day),
                 core.sent_join([premise, excl])),
                ("neutral", "unrelated_addition",
                 "%s%s." % (unrel[0].upper(), unrel[1:]), premise),
                ("neutral", "unstated_detail",
                 "%s %s %d %s on %s, and %s." % (actor[0].upper() + actor[1:], verb, qty, obj, day,
                                                 ctx.pick_val(rng, "nli/unst", UNSTATED)), premise),
            ]
            group = "nli/%d/%d" % (slot, rep)
            rid = "%s-nli-%d-%d" % (ctx.stream, slot, rep)
            recs = []
            hard = {"hypernym", "quantity_relaxation", "changed_number", "different_entity",
                    "unstated_detail", "antonym"}
            for label, kind, hyp, prem in pairs:
                st = {"premise": prem, "hypothesis": hyp}
                crit = core.shuffled_map(rng, dict(CRIT))
                q = {"type": "choice", "instructions": ctx.pick_val(rng, "nli/instr", NLI_INSTR),
                     "criteria": crit}
                recs.append(ctx.rec("%s-%s" % (rid, kind), FAMILY, "compatibility", kind, group,
                                    st, {"relation": q}, {"relation": label},
                                    "hard" if kind in hard else "easy",
                                    "paraphrase" if kind == "paraphrase" else
                                    ("negation" if kind == "negation" else "implicit"),
                                    [{"qid": "relation", "rule": "nli",
                                      "args": {"label": label, "options": sorted(crit)}}],
                                    evidence=[prem, hyp]))
            for label, kind, hyp, prem in [pairs[0], pairs[3], pairs[7]]:
                st = {"premise": prem, "hypothesis": hyp}
                q = {"type": "noul", "instructions": ctx.pick_val(rng, "nli/noul", NOUL_INSTR)}
                recs.append(ctx.rec("%s-%s-n" % (rid, kind), FAMILY, "compatibility", kind + "_noul",
                                    group, st, {"entailed": q}, {"entailed": label == "entailment"},
                                    "standard", "implicit",
                                    [{"qid": "entailed", "rule": "nli_bool", "args": {"label": label}}],
                                    evidence=[prem, hyp]))
            tids = sorted(set(ctx.cur_tids))
            for r in recs:
                r["meta"]["template_ids"] = sorted(set(tids + r["meta"]["template_ids"]))
                r["meta"]["pad_family"] = "log"
            emitted += len(recs)
            yield recs
            if limit and emitted >= limit:
                return
        rep += 1
        if rep > 4000:
            return
