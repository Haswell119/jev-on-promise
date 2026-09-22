#!/usr/bin/env python3
"""Family 5: answer adequacy and passage relevance.

Every knowledge item is a two-part question with a reference answer, from which
six candidate replies are derived by construction:

    complete        answers both parts
    partial         answers only the first part
    off_topic       answers nothing that was asked
    contradicting   asserts the opposite of the reference answer
    refusing        declines to answer
    other_question  a correct answer to a *different* question in the pool

Only `complete` resolves the request; the Choice variant asks which of the six
shapes the reply has, and the relevance variant asks which of four passages
answers the query.
"""
from . import core

FAMILY = "adequacy_relevance"

KINDS = ["complete", "partial", "off_topic", "contradicting", "refusing", "other_question"]
KIND_DESC = {
    "complete": "The reply answers everything that was asked",
    "partial": "The reply answers only part of what was asked",
    "off_topic": "The reply is about something else entirely",
    "contradicting": "The reply asserts the opposite of the established facts",
    "refusing": "The reply declines to answer",
    "other_question": "The reply is accurate but answers a different question",
}

# question, complete answer, partial answer, contradicting answer  (slots are filled per scenario)
ITEMS = [
    ("What is the return window for a {product}, and who pays the return postage?",
     "A {product} can be returned within {n} days of delivery, and we pay the return postage.",
     "A {product} can be returned within {n} days of delivery.",
     "A {product} cannot be returned at all, and any postage would be at your own cost."),
    ("How long does delivery to {city} take, and is a signature required?",
     "Delivery to {city} takes {n} working days and a signature is required on arrival.",
     "Delivery to {city} takes {n} working days.",
     "We do not deliver to {city}, so no signature question arises."),
    ("What does the warranty on the {product} cover, and how long does it last?",
     "The warranty on the {product} covers manufacturing defects and lasts {n} months.",
     "The warranty on the {product} covers manufacturing defects.",
     "The {product} carries no warranty of any kind."),
    ("How do I reset my password, and how long is the reset link valid?",
     "Open Settings, choose Security and select Reset password; the emailed link stays valid for {n} hours.",
     "Open Settings, choose Security and select Reset password.",
     "Passwords cannot be reset by customers under any circumstances."),
    ("What are the opening hours of the {city} branch, and is parking available?",
     "The {city} branch opens from 09:00 to 17:00 on weekdays and has {n} free parking spaces.",
     "The {city} branch opens from 09:00 to 17:00 on weekdays.",
     "The {city} branch closed permanently and has no parking."),
    ("How many seats does the standard plan include, and what happens if we exceed them?",
     "The standard plan includes {n} seats; extra seats are billed monthly at the list rate.",
     "The standard plan includes {n} seats.",
     "The standard plan has no seat limit, so nothing happens if you add more."),
    ("When is my invoice due, and which payment methods do you accept?",
     "The invoice falls due {n} days after issue and we accept bank transfer and card.",
     "The invoice falls due {n} days after issue.",
     "Invoices are payable immediately and only in cash."),
    ("Can I change my delivery address after ordering, and is there a fee?",
     "You may change the address until the order is packed, at no charge.",
     "You may change the address until the order is packed.",
     "The delivery address can never be altered once an order exists."),
    ("How do I cancel my subscription, and will I be refunded for the rest of the term?",
     "Cancel from Billing in the account settings; the unused part of the term is refunded pro rata.",
     "Cancel from Billing in the account settings.",
     "Subscriptions cannot be cancelled before the term ends and nothing is refundable."),
    ("Which documents are needed for the claim, and where do I send them?",
     "Send the completed form and the itemised receipt to the claims team at the {city} office.",
     "Send the completed form and the itemised receipt.",
     "No documents are needed and there is nowhere to send them."),
    ("How much notice must I give to end the tenancy, and who arranges the final inspection?",
     "Two months' written notice is required and the letting agent arranges the final inspection.",
     "Two months' written notice is required.",
     "No notice is needed and there is never a final inspection."),
    ("What time should I arrive for the appointment, and what should I bring?",
     "Arrive {n} minutes before the appointment and bring your referral letter and a photo ID.",
     "Arrive {n} minutes before the appointment.",
     "There is no need to arrive early and nothing needs to be brought."),
    ("How is the fuel surcharge calculated, and how often does it change?",
     "The surcharge is a percentage of the line haul rate and is reviewed every {n} weeks.",
     "The surcharge is a percentage of the line haul rate.",
     "There is no fuel surcharge on any shipment."),
    ("How do I enrol on the module, and what is the deadline?",
     "Enrol through the student portal; the deadline is {n} days before the term begins.",
     "Enrol through the student portal.",
     "Enrolment is closed and there is no deadline to meet."),
    ("What is included in the maintenance contract, and who do I call out of hours?",
     "The contract covers parts and labour on the {product}; the out-of-hours line is staffed by {agent}'s team.",
     "The contract covers parts and labour on the {product}.",
     "Nothing is included and there is no out-of-hours cover."),
    ("How long are records kept, and can I ask for them to be deleted?",
     "Records are kept for {n} years and you may request deletion in writing at any time.",
     "Records are kept for {n} years.",
     "Records are kept forever and deletion can never be requested."),
]

REFUSALS = [
    "I'm not able to help with that; please contact the store directly.",
    "Sorry, that is outside what I can assist with here.",
    "Unfortunately I cannot answer questions of that kind.",
    "I would rather not comment on that; try the published documentation.",
    "That is not something I can look into, I'm afraid.",
    "I have no way of dealing with this request.",
]
OFFTOPIC_PASSAGES = [
    "The staff canteen has moved to the second floor and now opens at seven in the morning.",
    "Our head office relocated to {city} last year and the postal address changed accordingly.",
    "The company newsletter is published on the first working day of each month.",
    "Cycle parking at the {city} site was doubled following the staff survey.",
    "The annual charity quiz raised a considerable sum for the local hospice.",
    "A new coffee supplier was appointed after a tasting session in {month}.",
    "The car park barrier is being replaced and access will be by intercom for a fortnight.",
    "Reception is trialling a digital visitor book at the {city} entrance.",
]
ADQ_INSTR = [
    "Does `answer` resolve `question`?",
    "Read `question` and `answer`: has the request been dealt with in full?",
    "Is `answer` a complete response to `question`?",
    "Judging only from the text, does the reply settle what was asked?",
]
CLS_INSTR = [
    "How does `answer` relate to `question`?",
    "Classify the reply against the question that was asked.",
    "Which description fits `answer` best?",
    "Judge the reply and pick the matching description.",
]
REL_INSTR = [
    "Which passage answers `query`?",
    "Pick the passage that contains the answer to `query`.",
    "Which of the passages is relevant to `query`?",
    "Select the passage that responds to `query`.",
]


def _fill(ctx, rng, s):
    return s.format(product=ctx.slot(rng, "product"), city=ctx.slot(rng, "city"),
                    n=ctx.slot(rng, "small"), agent=ctx.slot(rng, "agent"),
                    month=ctx.slot(rng, "month"))


def scenarios(ctx, limit=None):
    emitted = 0
    rep = 0
    n_items = len(ITEMS)
    while True:
        for i in range(n_items):
            ctx.reset_tids()
            rng = ctx.rng("adq", i, rep)
            tid, item = ctx.pick(rng, "adq/items", ITEMS)
            ctx.use(tid)
            q_t, full_t, part_t, contra_t = item
            question = _fill(ctx, rng, q_t)
            full = _fill(ctx, rng, full_t)
            partial = _fill(ctx, rng, part_t)
            contra = _fill(ctx, rng, contra_t)
            otid, other = ctx.pick(rng, "adq/items", ITEMS)
            while other is item:
                otid, other = ctx.pick(rng, "adq/items", ITEMS)
            ctx.use(otid)
            other_ans = _fill(ctx, rng, other[1])
            refusal = ctx.pick_val(rng, "adq/refusal", REFUSALS)
            offtopic = _fill(ctx, rng, ctx.pick_val(rng, "adq/off", OFFTOPIC_PASSAGES))
            answers = {"complete": full, "partial": partial, "off_topic": offtopic,
                       "contradicting": contra, "refusing": refusal, "other_question": other_ans}
            group = "adq/%d/%d" % (i, rep)
            rid = "%s-adq-%d-%d" % (ctx.stream, i, rep)
            recs = []
            diff = {"complete": "easy", "off_topic": "easy", "refusing": "standard",
                    "partial": "hard", "contradicting": "hard", "other_question": "hard"}
            negs = [k for k in KINDS if k != "complete"]
            rng.shuffle(negs)
            emit_kinds = ["complete", "complete_swapped"] + negs[:2]
            for kind in emit_kinds:
                real = "complete" if kind == "complete_swapped" else kind
                atext = answers[real]
                if kind == "complete_swapped":
                    bits = [b.strip() for b in atext.replace(";", ",").rstrip(".").split(", ")]
                    if len(bits) >= 2:
                        atext = "%s. %s." % (bits[-1][0].upper() + bits[-1][1:], ", ".join(bits[:-1]))
                    if atext == answers[real]:
                        atext = atext + " That covers both parts of your question."
                st = {"question": question, "answer": atext}
                q = {"type": "noul", "instructions": ctx.pick_val(rng, "adq/instr", ADQ_INSTR),
                     "criteria": {"true": "The reply answers the whole question",
                                  "false": "The reply leaves the question unanswered in whole or in part"}}
                recs.append(ctx.rec("%s-%s" % (rid, kind), FAMILY, "adequacy_relevance",
                                    "noul_" + kind, group, st, {"resolved": q},
                                    {"resolved": real == "complete"},
                                    diff[real] if kind != "complete_swapped" else "standard",
                                    "adequacy" if kind != "complete_swapped" else "paraphrase",
                                    [{"qid": "resolved", "rule": "adequacy_bool", "args": {"kind": real}}],
                                    evidence=[]))
            # classification of one reply shape (judge tier)
            ckind = rng.choice(KINDS)
            crit = core.shuffled_map(rng, {k: KIND_DESC[k] for k in KINDS})
            st = {"question": question, "answer": answers[ckind]}
            qc = {"type": "choice", "instructions": ctx.pick_val(rng, "adq/cls", CLS_INSTR),
                  "criteria": crit}
            recs.append(ctx.rec(rid + "-class", FAMILY, "adequacy_relevance", "classify_" + ckind,
                                group, st, {"shape": qc}, {"shape": ckind}, "judge", "adequacy",
                                [{"qid": "shape", "rule": "adequacy_label",
                                  "args": {"kind": ckind, "options": sorted(crit)}}], evidence=[]))
            # passage relevance
            pids = ["p1", "p2", "p3", "p4"]
            kinds4 = ["answer", "off_topic", "other_question", "contradicting"]
            texts = {"answer": full, "off_topic": offtopic, "other_question": other_ans,
                     "contradicting": contra}
            rng.shuffle(pids)
            assign = dict(zip(pids, kinds4))
            passages = {p: texts[assign[p]] for p in sorted(assign)}
            qr = {"type": "choice", "instructions": ctx.pick_val(rng, "adq/rel", REL_INSTR),
                  "criteria": {p: None for p in sorted(passages)}}
            gold_p = [p for p in sorted(assign) if assign[p] == "answer"][0]
            recs.append(ctx.rec(rid + "-rel", FAMILY, "adequacy_relevance", "passage_relevance",
                                group, {"query": question, "passages": passages}, {"best": qr},
                                {"best": gold_p}, "standard", "adequacy",
                                [{"qid": "best", "rule": "relevance",
                                  "args": {"passages": assign, "answer_kind": "answer"}}],
                                evidence=[]))
            tids = sorted(set(ctx.cur_tids))
            for r in recs:
                r["meta"]["template_ids"] = sorted(set(tids + r["meta"]["template_ids"]))
                r["meta"]["pad_family"] = "handbook"
            emitted += len(recs)
            yield recs
            if limit and emitted >= limit:
                return
        rep += 1
        if rep > 4000:
            return
