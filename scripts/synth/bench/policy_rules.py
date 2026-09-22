#!/usr/bin/env python3
"""Family 4: policy application with conditions, exceptions and precedence.

A policy is assembled from 2-4 generated conditions, an optional override
clause ("notwithstanding ...") and an optional exception attached to one
condition ("unless ...", "except when ..."). The case facts are rendered either
as prose spread over several sentences or as JSON fields.

Evaluation order (reproduced independently by verify.py):
    1. if the override clause holds          -> allowed
    2. otherwise the first blocking clause   -> that clause's key
    3. otherwise                             -> allowed

Two multi-hop shapes are always available: a condition that depends on a
definition given in a different paragraph, and a condition whose threshold is
stated in a different section of the document.
"""
from . import core

FAMILY = "policy_rules"

ALLOW = "approved"
CATEGORIES = ["perishable food", "consumer electronics", "clothing", "furniture", "books",
              "cosmetics", "toys", "garden equipment", "jewellery", "sports gear"]
SEGMENTS = ["standard", "priority", "platinum", "trial", "legacy"]

DOMAINS = [
    ("refund", "Refund policy", "refund request", "the customer"),
    ("expense", "Expense reimbursement policy", "expense claim", "the employee"),
    ("leave", "Leave approval policy", "leave request", "the employee"),
    ("claim", "Claims handling policy", "insurance claim", "the policyholder"),
    ("warranty", "Warranty policy", "warranty repair", "the owner"),
    ("discount", "Discount authorisation policy", "discount request", "the account"),
    ("permit", "Permit issuing policy", "permit application", "the applicant"),
    ("grant", "Grant funding policy", "funding application", "the organisation"),
]


def _c_window(ctx, rng, dom):
    n = rng.choice([7, 14, 21, 30, 45, 60, 90])
    d = rng.choice([2, 5, 9, 13, 18, 25, 31, 40, 55, 70, 95, 120])
    return dict(
        key="outside_window",
        clause="A %s is only considered when it is submitted within %d days of the event it relates to." % (DOM_NOUN[dom], n),
        fact="The submission was made %d days after the event." % d,
        json=("days_since_event", d),
        blocks=d > n,
        desc="Too many days passed before the request was submitted",
        rank=1,
    )


def _c_amount(ctx, rng, dom):
    lim = rng.choice([250, 500, 1000, 2500, 5000, 10000])
    amt = round(ctx.slot(rng, "amount"), 2)
    return dict(
        key="over_limit",
        clause="The value at stake must not exceed $%s." % "{:,}".format(lim),
        fact="The value at stake is $%s." % "{:,.2f}".format(amt),
        json=("amount", amt),
        blocks=amt > lim,
        desc="The amount exceeds the permitted limit",
        rank=2,
    )


def _c_category(ctx, rng, dom):
    excl = rng.sample(CATEGORIES, 2)
    cat = rng.choice(CATEGORIES)
    return dict(
        key="excluded_category",
        clause="Items in the categories %s and %s are excluded from this policy." % (excl[0], excl[1]),
        fact="The item concerned is classed as %s." % cat,
        json=("category", cat),
        blocks=cat in excl,
        desc="The item belongs to an excluded category",
        rank=3,
    )


def _c_documentation(ctx, rng, dom):
    present = rng.random() < 0.55
    doc = rng.choice(["an itemised receipt", "a signed declaration", "the original packing note",
                      "a photograph of the damage", "the completed form B"])
    return dict(
        key="missing_documentation",
        clause="%s must be attached to the submission." % (doc[0].upper() + doc[1:]),
        fact="%s was %sattached." % (doc[0].upper() + doc[1:], "" if present else "not "),
        json=("documentation_attached", present),
        blocks=not present,
        desc="A required document is missing",
        rank=1,
    )


def _c_condition(ctx, rng, dom):
    opened = rng.random() < 0.5
    return dict(
        key="item_used",
        clause="The item must be returned unused and in its original packaging.",
        fact="The item was %s." % ("opened and used" if opened else "still sealed in its packaging"),
        json=("item_used", opened),
        blocks=opened,
        desc="The item is no longer unused",
        rank=2,
    )


def _c_frequency(ctx, rng, dom):
    k = rng.choice([1, 2, 3, 4])
    j = rng.choice([1, 2, 3, 4, 5, 6])
    return dict(
        key="too_frequent",
        clause="No more than %d such submissions are permitted in a rolling twelve months." % k,
        fact="This is submission number %d in the current twelve month period." % j,
        json=("submissions_this_year", j),
        blocks=j > k,
        desc="The permitted number of submissions has been exceeded",
        rank=2,
    )


def _c_residency(ctx, rng, dom):
    area = [ctx.slot(rng, "city") for _ in range(2)]
    here = ctx.slot(rng, "city")
    return dict(
        key="outside_area",
        clause="Only submissions relating to the %s and %s districts are handled here." % (area[0], area[1]),
        fact="The address on the submission is in %s." % here,
        json=("district", here),
        blocks=here not in area,
        desc="The address falls outside the districts covered",
        rank=3,
    )


def _c_approval(ctx, rng, dom):
    got = rng.random() < 0.5
    return dict(
        key="no_prior_approval",
        clause="Prior written approval from a line manager is required before the work begins.",
        fact="Written approval %s obtained before the work began." % ("was" if got else "was not"),
        json=("prior_approval", got),
        blocks=not got,
        desc="Prior written approval was never obtained",
        rank=1,
    )


CONDS = [_c_window, _c_amount, _c_category, _c_documentation, _c_condition, _c_frequency,
         _c_residency, _c_approval]
DOM_NOUN = {d[0]: d[2] for d in DOMAINS}

EXC_CONNECTIVES = [
    "unless %s", "except when %s", "save where %s", "this does not apply when %s",
    "the preceding sentence is disapplied if %s", "but not where %s",
]
OVERRIDE_FRAMES = [
    "Notwithstanding anything above, a submission from a %s account is always approved.",
    "Notwithstanding the conditions in this section, %s accounts are approved without further checks.",
    "The conditions above do not apply to %s accounts, which are approved in every case.",
    "Where the account is marked %s, approval follows automatically regardless of the conditions above.",
]
SEVERITY_LEVELS = ["Compliant - no deviation from the policy",
                   "Minor deviation - paperwork or process gap",
                   "Material deviation - a substantive condition is not met",
                   "Disqualifying - the submission falls outside the policy altogether"]

ELIG_INSTR = [
    "Under the policy quoted, may the submission be approved?",
    "Applying the rules as written, is the submission eligible?",
    "Does the case satisfy every applicable condition of the policy?",
    "Read the policy and decide whether the request can go ahead.",
]
OUTCOME_INSTR = [
    "What is the outcome of applying the policy to this case?",
    "Which provision decides this case?",
    "State the outcome under the policy, naming the blocking provision if there is one.",
    "Apply the policy and report the result.",
]
DEFINITION_FRAMES = [
    "In this document a covered member means an account with at least %d months of continuous standing.",
    "For the purposes of this document, 'covered member' refers to any account in continuous standing for %d months or more.",
    "Definition: a covered member is an account that has been in continuous standing for no fewer than %d months.",
    "The term covered member, wherever used, means an account with %d or more months of continuous standing.",
]
SECTION_THRESHOLD_FRAMES = [
    "Section 4 (Limits) sets the authorisation limit for this class of submission at $%s.",
    "The limit referred to elsewhere in this document is fixed by Section 4 at $%s.",
    "Under Section 4 the applicable authorisation limit is $%s.",
    "Section 4 of this document states the limit as $%s.",
]


def _evaluate(clauses, override_holds):
    if override_holds:
        return ALLOW
    for c in clauses:
        if c["blocks"]:
            return c["key"]
    return ALLOW


def _assemble(ctx, rng, dom, multi_hop):
    clauses = []
    fns = rng.sample(CONDS, rng.choice([2, 3, 4]))
    for fn in fns:
        clauses.append(fn(ctx, rng, dom))
    extra_paras = []
    if multi_hop == "definition":
        months_req = rng.choice([3, 6, 12, 18, 24])
        months_has = rng.choice([1, 2, 4, 8, 11, 13, 20, 30])
        extra_paras.append(ctx.pick_val(rng, "pol/def", DEFINITION_FRAMES) % months_req)
        clauses.insert(rng.randrange(len(clauses) + 1), dict(
            key="not_covered_member",
            clause="Only a covered member may submit under this policy.",
            fact="The account has been in continuous standing for %d months." % months_has,
            json=("months_standing", months_has),
            blocks=months_has < months_req,
            desc="The submitter does not meet the definition of a covered member",
            rank=3,
        ))
    elif multi_hop == "section":
        lim = rng.choice([400, 900, 2000, 4500, 12000])
        amt = round(ctx.slot(rng, "amount"), 2)
        extra_paras.append(ctx.pick_val(rng, "pol/sec", SECTION_THRESHOLD_FRAMES) % "{:,}".format(lim))
        clauses.insert(rng.randrange(len(clauses) + 1), dict(
            key="above_section_limit",
            clause="A submission above the limit set out in Section 4 may not be handled under this section.",
            fact="The submission is for $%s." % "{:,.2f}".format(amt),
            json=("amount", amt),
            blocks=amt > lim,
            desc="The value is above the limit fixed in Section 4",
            rank=3,
        ))
    # exception attached to one clause
    exc_idx = rng.randrange(len(clauses))
    defective = rng.random() < 0.4
    exc_text = ctx.pick_val(rng, "pol/exc", EXC_CONNECTIVES) % "the item is confirmed defective"
    clauses[exc_idx]["clause"] = clauses[exc_idx]["clause"].rstrip(".") + ", " + exc_text + "."
    exc_fact = "The item %s confirmed as defective." % ("has been" if defective else "has not been")
    if defective:
        clauses[exc_idx] = dict(clauses[exc_idx], blocks=False, exception_applied=True)
    # override clause
    seg = rng.choice(SEGMENTS)
    acct_seg = rng.choice(SEGMENTS)
    override_text = ctx.pick_val(rng, "pol/ovr", OVERRIDE_FRAMES) % seg
    override_holds = (acct_seg == seg)
    override_fact = "The account is marked %s." % acct_seg
    return clauses, extra_paras, exc_fact, override_text, override_holds, override_fact


def scenarios(ctx, limit=None):
    emitted = 0
    rep = 0
    shapes = ["plain", "definition", "section"]
    while True:
        for dcode, dtitle, dnoun, dactor in DOMAINS:
            for shape in shapes:
                ctx.reset_tids()
                rng = ctx.rng("pol", dcode, shape, rep)
                clauses, extra, exc_fact, ovr_text, ovr_holds, ovr_fact = _assemble(ctx, rng, dcode, shape)
                outcome = _evaluate(clauses, ovr_holds)
                eligible = outcome == ALLOW
                group = "pol/%s/%s/%d" % (dcode, shape, rep)
                rid = "%s-pol-%s-%s-%d" % (ctx.stream, dcode, shape, rep)

                policy_parts = ["%s." % dtitle] + [c["clause"] for c in clauses] + extra + [ovr_text]
                case_parts = [c["fact"] for c in clauses] + [exc_fact, ovr_fact]
                rng.shuffle(case_parts)
                policy_text = core.sent_join(policy_parts)
                case_text = core.sent_join(["Case notes for %s." % dactor] + case_parts)
                prose = core.sent_join([policy_text, case_text])
                evidence = [c["clause"] for c in clauses] + [c["fact"] for c in clauses]

                chk_clauses = [{"key": c["key"], "blocks": bool(c["blocks"]),
                                "rank": c["rank"]} for c in clauses]
                pol_args = {"clauses": chk_clauses, "override_holds": bool(ovr_holds),
                            "allow_key": ALLOW}
                recs = []
                q1 = {"type": "noul", "instructions": ctx.pick_val(rng, "pol/elig", ELIG_INSTR),
                      "criteria": {"true": "Every applicable condition is satisfied",
                                   "false": "At least one condition or exclusion blocks it"}}
                diff = "hard" if shape != "plain" else "standard"
                reason = "multi_hop" if shape != "plain" else "policy"
                recs.append(ctx.rec(rid + "-elig", FAMILY, "policy_rules", "eligibility_" + shape,
                                    group, prose, {"eligible": q1}, {"eligible": eligible},
                                    diff, reason,
                                    [{"qid": "eligible", "rule": "policy_bool", "args": pol_args}],
                                    evidence=evidence))
                crit = {ALLOW: "Every condition is satisfied and the submission goes ahead"}
                for c in clauses:
                    crit[c["key"]] = c["desc"]
                crit = core.shuffled_map(rng, crit)
                q2 = {"type": "choice", "instructions": ctx.pick_val(rng, "pol/out", OUTCOME_INSTR),
                      "criteria": crit}
                recs.append(ctx.rec(rid + "-outcome", FAMILY, "policy_rules", "outcome_" + shape,
                                    group, prose, {"outcome": q2}, {"outcome": outcome},
                                    diff, reason,
                                    [{"qid": "outcome", "rule": "policy_outcome",
                                      "args": dict(pol_args, options=sorted(crit))}],
                                    evidence=evidence))
                ranks = {ALLOW: 0}
                for c in clauses:
                    ranks[c["key"]] = c["rank"]
                q3 = {"type": "score", "instructions": "How far does this case depart from the policy?",
                      "criteria": SEVERITY_LEVELS}
                recs.append(ctx.rec(rid + "-severity", FAMILY, "policy_rules", "severity_" + shape,
                                    group, prose, {"severity": q3}, {"severity": ranks[outcome]},
                                    "standard", "policy",
                                    [{"qid": "severity", "rule": "policy_severity",
                                      "args": dict(pol_args, ranks=ranks)}],
                                    evidence=evidence))
                # JSON state: the same case with the facts spread over fields
                case_obj = {"reference": ctx.ticket_id(rng), "handler": ctx.slot(rng, "agent")}
                for c in clauses:
                    k, v = c["json"]
                    case_obj[k] = v
                case_obj["defect_confirmed"] = "has been" in exc_fact
                case_obj["account_segment"] = ovr_fact.rstrip(".").split()[-1]
                state_json = {"policy": {"title": dtitle,
                                         "clauses": [c["clause"] for c in clauses],
                                         "notes": extra, "override": ovr_text},
                              "case": case_obj}
                q4 = {"type": "noul", "instructions":
                      "Apply `policy` to `case`: may the submission be approved?"}
                recs.append(ctx.rec(rid + "-json", FAMILY, "policy_rules", "json_" + shape, group,
                                    state_json, {"eligible": q4}, {"eligible": eligible},
                                    "hard", "policy",
                                    [{"qid": "eligible", "rule": "policy_bool", "args": pol_args}],
                                    evidence=[]))
                tids = sorted(set(ctx.cur_tids))
                for r in recs:
                    r["meta"]["template_ids"] = sorted(set(tids + r["meta"]["template_ids"]))
                    r["meta"]["pad_family"] = "policy"
                emitted += len(recs)
                yield recs
                if limit and emitted >= limit:
                    return
        rep += 1
        if rep > 4000:
            return
