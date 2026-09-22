#!/usr/bin/env python3
"""Family 7: ordinal judgement over six graded scales.

Each scale has four ordered levels. Every level owns several *state* templates
(what the world looks like at that level, including graded magnitude words,
counts and consequences) and several *description* phrasings, so the same gold
level can be asked with differently worded criteria. Variants: full scale,
re-worded scale, collapsed three-point scale, keyed Choice labels, and a Noul
threshold question ("is this at or above ...").
"""
from . import core

FAMILY = "ordinal"

# scale -> (question phrasings, choice key set, levels[(desc phrasings, state templates)])
SCALES = {
"severity": (
 ["How severe is the incident described?", "Grade the impact of this incident.",
  "Assess the blast radius of what is reported.", "How bad is this incident?"],
 ["S4", "S3", "S2", "S1"],
 [
  (["Cosmetic only; nothing stops working",
    "Appearance is affected but every function still works",
    "A visual imperfection with no functional impact"],
   ["The logo sits {mag} pixels too far left on one screen; every feature behaves normally.",
    "Two menu labels use different fonts in the {city} tenant, otherwise the product is fully usable.",
    "A tooltip shows a stale colour after the theme change; nothing functional is affected.",
    "The footer date is {mag} out of alignment on printed reports; the content is correct."]),
  (["One customer or an optional feature is affected and a workaround exists",
    "Limited blast radius with a usable workaround",
    "A single account or a non-essential feature is degraded; people can still work"],
   ["Only the {company} account cannot download invoices; support is emailing the PDFs meanwhile.",
    "The optional calendar sync fails for {n} user, who is using the web calendar instead.",
    "One tenant's weekly summary email is {mag} late; the same numbers are on their dashboard.",
    "Exports to CSV fail for {company} only; the same data can be copied from the table view."]),
  (["A core workflow is unavailable for many customers; records remain intact",
    "Widespread loss of a primary function with no data loss",
    "Something essential is down across a large share of accounts, but nothing was destroyed"],
   ["Since the 09:15 deploy, sign-in fails for roughly {n} percent of accounts; nothing in the database was altered.",
    "Payments cannot be submitted from the mobile app in any region; stored orders are unaffected.",
    "Search is down for every workspace and {n} tickets have arrived in an hour, though all documents remain safe.",
    "The scheduler has not run for {mag} hours across all tenants; queued jobs are still intact."]),
  (["Data has been destroyed beyond recovery, or someone was hurt",
    "Irreversible loss or physical harm",
    "Permanent destruction of records or an injury to a person"],
   ["Last night's cleanup job erased {mag} weeks of audit logs and the backups had already expired.",
    "A charger unit caught fire during testing and a technician suffered burns to the arm.",
    "The customer table was dropped in production; the most recent snapshot predates the incident by {mag} weeks.",
    "A forklift struck a worker at the {city} depot and the injury required hospital treatment."]),
 ]),
"urgency": (
 ["How urgent is this request?", "How quickly does this need attention?",
  "Rate the time pressure in the message.", "How pressing is the matter described?"],
 ["routine", "soon", "priority", "immediate"],
 [
  (["No deadline; can wait indefinitely", "Whenever convenient, nothing depends on it",
    "There is no time pressure at all"],
   ["Whenever you get a moment, could you confirm the spelling of my address? No rush at all.",
    "At some point this year it would be nice to tidy up the old records; nothing depends on it.",
    "Purely for my own curiosity, how many {product}s does a pallet hold?",
    "No hurry, but I wondered what the archive policy says about old invoices."]),
  (["Needed within the next week or two", "A soft deadline some days away",
    "Should be dealt with this week or next"],
   ["Could you look at this before the end of next week? The report is due shortly after.",
    "We would like an answer within {n} days so the paperwork can go to the board.",
    "This should be sorted before the {month} cycle closes, so there is a little time.",
    "Please deal with it in the next week or so; nothing breaks before then."]),
  (["Needed today or tomorrow; something is blocked", "A hard deadline within a day",
    "Work is blocked until this is resolved, and it must move today"],
   ["The line stops tomorrow morning unless the parts are released today.",
    "We cannot invoice anything until this is fixed, and {n} invoices are waiting.",
    "The auditor arrives tomorrow and this document must be signed before then.",
    "Our team is blocked and {n} people are sitting idle until this is unblocked."]),
  (["An emergency: safety, money or the business is at immediate risk",
    "Right now; there is danger or active loss",
    "Immediate action required to prevent harm or continuing loss"],
   ["There is a strong smell of gas in the plant room and the alarm is sounding right now.",
    "We are losing roughly ${mag} a minute while checkout is down; this is happening as I type.",
    "A patient is waiting in the chair and the system will not release the record.",
    "Water is pouring through the ceiling into the server room at this moment."]),
 ]),
"satisfaction": (
 ["How satisfied is the customer?", "Rate the customer's satisfaction.",
  "How happy does the customer sound?", "Grade the sentiment of this feedback."],
 ["very_negative", "negative", "positive", "very_positive"],
 [
  (["Angry: the customer is furious and threatens to leave",
    "Extremely dissatisfied and hostile", "As negative as it gets"],
   ["This is the worst service I have ever had. {n} emails and nobody answers. I am done with you.",
    "Absolutely disgraceful. You have taken my money and given me nothing. I will report this.",
    "I am beyond angry. {mag} weeks of excuses and still no {product}. Cancel everything.",
    "Outrageous treatment from start to finish. I will be telling everyone I know."]),
  (["Unhappy but measured", "Dissatisfied, though still civil",
    "Mildly negative; disappointed rather than angry"],
   ["I am disappointed that the {product} arrived {mag} days late; it was not what I expected.",
    "This is the second time I have had to ask, which is frustrating.",
    "Not great, honestly. The packaging was damaged and nobody warned me.",
    "A bit let down by the experience, though I appreciate you looking at it."]),
  (["Content; the request was handled acceptably", "Mildly positive",
    "Satisfied without enthusiasm"],
   ["Thanks, that sorted it. The {product} works fine now.",
    "All good, the replacement arrived on {weekday} as promised.",
    "That answers my question, thank you for coming back to me.",
    "No complaints; the order turned up and everything was correct."]),
  (["Delighted; the customer praises the service warmly",
    "Extremely positive and grateful", "Enthusiastic praise"],
   ["Absolutely brilliant service. {agent} went out of their way and I could not be happier.",
    "Fantastic from start to finish, the best experience I have had with any supplier.",
    "I am delighted. The {product} is superb and the delivery was {mag} days early.",
    "Outstanding. Please pass my thanks to {agent}; genuinely excellent."]),
 ]),
"risk": (
 ["How risky is the change described?", "Rate the risk of proceeding.",
  "What level of risk does this carry?", "Grade the exposure in this proposal."],
 ["negligible", "low", "moderate", "high"],
 [
  (["Negligible: reversible, isolated and tested",
    "Essentially no exposure", "Trivial and easily undone"],
   ["A typo in a help page is being corrected; the change is reversible in one click.",
    "We are renaming an internal variable in a module covered by {n} tests.",
    "The change adds a log line behind a flag that is off everywhere.",
    "A single static image is being replaced with a smaller version of itself."]),
  (["Low: contained, with a rehearsed rollback",
    "Small exposure and a practised way back", "Limited scope, recovery is routine"],
   ["A new endpoint is being added alongside the old one; rollback is a config toggle.",
    "We are upgrading a library by one patch version with {n} integration tests green.",
    "The change affects one internal report used by {n} people, and the old view remains.",
    "A cache is being resized; the previous value can be restored in minutes."]),
  (["Moderate: customer-facing and hard to reverse quickly",
    "Material exposure with a slow rollback", "Visible to customers and awkward to undo"],
   ["The checkout page layout changes for all customers and reverting needs a full deploy.",
    "We are migrating {mag} percent of traffic to the new service with no shadow run.",
    "Pricing logic changes for {n} accounts and the old figures are not kept.",
    "A schema column is being renamed; the rollback script has never been rehearsed."]),
  (["High: irreversible, wide blast radius or touching money and safety",
    "Severe exposure with no way back", "Irreversible and affecting everyone or everything critical"],
   ["The production database is being migrated in place with no verified backup.",
    "Payment routing switches to a new provider for every customer at once, with no fallback.",
    "We are deleting {mag} million historical rows that cannot be reconstructed.",
    "The access control model changes for all {n} thousand users in a single step."]),
 ]),
"quality": (
 ["How good is the written work described?", "Rate the quality of the submission.",
  "Grade the standard of this piece of work.", "How well was the work done?"],
 ["poor", "fair", "good", "excellent"],
 [
  (["Poor: incomplete and full of errors", "Well below standard",
    "Unusable as it stands"],
   ["The report is {mag} pages short, has {n} broken references and no conclusion.",
    "Half the sections are placeholder text and the figures do not add up.",
    "The submission ignores the brief and contains {n} factual errors on the first page.",
    "It arrived unformatted, with the wrong client's name throughout."]),
  (["Fair: acceptable but needing rework", "Passable with noticeable gaps",
    "Meets the minimum but needs another pass"],
   ["The draft covers the brief but {n} sections need rewriting and the tone is uneven.",
    "It is workable, though {n} of the charts are mislabelled.",
    "The structure is fine; the analysis is thin in {n} places and the summary is vague.",
    "Acceptable overall, with {mag} typos and one missing appendix."]),
  (["Good: solid, complete and clear", "Meets the standard comfortably",
    "Well made with only trivial nitpicks"],
   ["The report is complete, clearly argued and {n} reviewers had only minor comments.",
    "Everything asked for is there, well presented, with one small formatting nit.",
    "Clear structure, correct figures and a summary that stands on its own.",
    "A solid piece of work; the only change requested was a caption wording."]),
  (["Excellent: exceeds the brief and needs no changes",
    "Outstanding, publishable as is", "Exceptional work beyond what was asked"],
   ["The submission exceeds the brief, anticipates {n} objections and needs no changes.",
    "Reviewers described it as the best example they had seen this year, with nothing to correct.",
    "It answers the brief in full and adds a sensitivity analysis nobody had asked for.",
    "Flawless: accurate throughout, elegantly written and ready to publish unchanged."]),
 ]),
"verbosity": (
 ["How long-winded is the message?", "Rate how verbose the text is.",
  "How concise is this message?", "Grade the message for length and padding."],
 ["terse", "concise", "wordy", "rambling"],
 [
  (["Extremely terse: a few words, no context", "Minimal, almost telegraphic",
    "Barely a sentence"],
   ["Broken. Fix.", "Order {n}. Where?", "Refund please.", "Still waiting."]),
  (["Concise: one or two clear sentences", "Short and to the point",
    "Says what is needed and stops"],
   ["Order {n} has not arrived. Could you check the status, please?",
    "The {product} stopped working yesterday. I would like a replacement.",
    "Please cancel my subscription at the end of the term. Thanks.",
    "My invoice shows the wrong address. Could you reissue it?"]),
  (["Wordy: several sentences with avoidable padding",
    "Longer than it needs to be", "Repeats itself and adds background nobody needs"],
   ["I hope this finds you well. I am writing, as I mentioned in my previous message, about order {n}, "
    "which as you may recall I placed some time ago, and which, as things stand, has not arrived. "
    "I did check with my neighbour, who was in all day, and she saw nothing either.",
    "Firstly, thank you for the service so far. Secondly, and this is the main point, the {product} "
    "is faulty. Thirdly, I should mention that I bought it partly on a recommendation. "
    "In short, I would like a replacement if that is possible.",
    "Just to give you the full picture before I get to the question, we have been customers for "
    "{n} years and have rarely had trouble. That said, the current problem concerns the invoice, "
    "which appears to be addressed to the wrong company entirely.",
    "Apologies for the long message. Some context first: we moved offices in {month}, which "
    "changed our delivery arrangements, and separately our finance system was upgraded. "
    "The upshot is that the last two deliveries went to the old address."]),
  (["Rambling: many paragraphs of digression around a small point",
    "Very long and largely off the point", "Buries a short request under a great deal of narrative"],
   ["I will start at the beginning, because the background matters. We first bought from you "
    "{n} years ago, on the recommendation of a colleague who has since retired to {city}. "
    "At that time the {product} was a different model entirely, and I remember the packaging "
    "being much larger than it is now. Over the years we have ordered perhaps {mag} times. "
    "Last spring there was an issue with a delivery that was eventually resolved, although it "
    "took several weeks and a number of telephone calls, none of which I am raising again now. "
    "Anyway, the reason I am writing is that this month's invoice appears to be duplicated.",
    "Bear with me, there is a question at the end. Our department was reorganised in {month}, "
    "which meant that responsibility for ordering moved from my colleague to me, and the handover "
    "was not as complete as it might have been. I inherited a spreadsheet, various email threads "
    "and a filing cabinet whose key nobody can find. Since then I have been piecing together who "
    "ordered what. Most of it is now clear. What I still cannot work out is whether the {product} "
    "delivered in {month} was ever paid for.",
    "This is probably a short question wrapped in a long story. When we opened the {city} site "
    "we chose your company because of the delivery times. The building itself had been empty for "
    "{n} years and needed a great deal of work, which is why the loading bay was not ready at first. "
    "We managed with the side entrance for a while. The neighbours were understanding. "
    "In any case, the loading bay is now open, and I would like future deliveries to use it.",
    "Forgive the length of this. I have been meaning to write for {mag} weeks. Our team uses the "
    "{product} daily and on the whole we are happy. There was a period last year when the supply "
    "was interrupted, but that was clearly not your doing. We also changed our internal process, "
    "which caused some confusion of our own making. Having said all that, the point of this email "
    "is that the reorder link in your portal no longer works for us."]),
 ]),
}

COLLAPSE3 = [0, 1, 1, 2]
COLLAPSE3_LEVELS = {
    "severity": ["Cosmetic or contained", "Significant but recoverable", "Catastrophic"],
    "urgency": ["No real time pressure", "Needs attention soon", "Emergency"],
    "satisfaction": ["Unhappy", "Content", "Delighted"],
    "risk": ["Little exposure", "Real exposure", "Severe exposure"],
    "quality": ["Below standard", "Acceptable", "Outstanding"],
    "verbosity": ["Very short", "Reasonable length", "Very long"],
}
CHOICE_INSTR = {
    "severity": "Which severity label fits this incident?",
    "urgency": "Which urgency label fits this request?",
    "satisfaction": "Which satisfaction label fits this feedback?",
    "risk": "Which risk label fits this change?",
    "quality": "Which quality label fits this work?",
    "verbosity": "Which length label fits this message?",
}
MAGS = ["three", "four", "six", "nine", "twelve", "twenty", "thirty", "forty"]


def _fill(ctx, rng, s):
    return s.format(city=ctx.slot(rng, "city"), company=ctx.slot(rng, "company"),
                    product=ctx.slot(rng, "product"), agent=ctx.slot(rng, "agent"),
                    month=ctx.slot(rng, "month"), weekday=ctx.slot(rng, "weekday"),
                    n=ctx.slot(rng, "small"), mag=ctx.pick_val(rng, "ord/mag", MAGS))


def scenarios(ctx, limit=None):
    emitted = 0
    rep = 0
    names = sorted(SCALES)
    while True:
        for name in names:
            questions, keys, levels = SCALES[name]
            for li in range(len(levels)):
                ctx.reset_tids()
                rng = ctx.rng("ord", name, li, rep)
                descs, states = levels[li]
                st = _fill(ctx, rng, ctx.pick_val(rng, "ord/state/%s/%d" % (name, li), states))
                group = "ord/%s/%d/%d" % (name, li, rep)
                rid = "%s-ord-%s-%d-%d" % (ctx.stream, name, li, rep)
                recs = []
                for variant, di in (("scale_a", 0), ("scale_b", 1)):
                    crit = [ctx.pick_val(rng, "ord/desc/%s/%d" % (name, j), levels[j][0])
                            for j in range(len(levels))]
                    q = {"type": "score", "instructions": ctx.pick_val(rng, "ord/q/%s" % name, questions),
                         "criteria": crit}
                    recs.append(ctx.rec("%s-%s" % (rid, variant), FAMILY, "ordinal", variant, group,
                                        st, {"level": q}, {"level": li},
                                        "easy" if variant == "scale_a" else "standard",
                                        "ordinal" if variant == "scale_a" else "paraphrase",
                                        [{"qid": "level", "rule": "ordinal_level",
                                          "args": {"level": li, "n_levels": len(levels)}}],
                                        evidence=[st]))
                q3 = {"type": "score", "instructions": ctx.pick_val(rng, "ord/q/%s" % name, questions),
                      "criteria": COLLAPSE3_LEVELS[name]}
                recs.append(ctx.rec(rid + "-collapsed", FAMILY, "ordinal", "collapsed_scale", group,
                                    st, {"level": q3}, {"level": COLLAPSE3[li]}, "hard", "ordinal",
                                    [{"qid": "level", "rule": "ordinal_level",
                                      "args": {"level": li, "collapse": COLLAPSE3}}],
                                    evidence=[st]))
                crit_c = {}
                for j, k in enumerate(keys):
                    crit_c[k] = levels[j][0][0]
                crit_c = core.shuffled_map(rng, crit_c)
                q4 = {"type": "choice", "instructions": CHOICE_INSTR[name], "criteria": crit_c}
                recs.append(ctx.rec(rid + "-label", FAMILY, "ordinal", "keyed_label", group,
                                    st, {"label": q4}, {"label": keys[li]}, "standard", "ordinal",
                                    [{"qid": "label", "rule": "ordinal_choice",
                                      "args": {"level": li, "keys": list(keys),
                                               "options": sorted(crit_c)}}],
                                    evidence=[st]))
                thr = rng.randrange(1, len(levels))
                q5 = {"type": "noul", "instructions":
                      "Is this at or above the level described as: %s?" % levels[thr][0][0]}
                recs.append(ctx.rec(rid + "-threshold", FAMILY, "ordinal", "threshold", group,
                                    st, {"at_or_above": q5}, {"at_or_above": li >= thr},
                                    "hard", "ordinal",
                                    [{"qid": "at_or_above", "rule": "ordinal_threshold",
                                      "args": {"level": li, "op": "ge", "rhs": thr}}],
                                    evidence=[st]))
                tids = sorted(set(ctx.cur_tids))
                for r in recs:
                    r["meta"]["template_ids"] = sorted(set(tids + r["meta"]["template_ids"]))
                    r["meta"]["pad_family"] = "meeting"
                emitted += len(recs)
                yield recs
                if limit and emitted >= limit:
                    return
        rep += 1
        if rep > 3000:
            return
