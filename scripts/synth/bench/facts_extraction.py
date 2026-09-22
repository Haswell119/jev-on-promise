#!/usr/bin/env python3
"""Family 2: fact checking and enum extraction over JSON and text records.

Seven record types (orders, tickets, invoices, shifts, inventory, patient
admin, server logs). For every scenario the module emits

  * a Noul fact that is *stated* in the record,
  * a Noul fact that is *absent* from the record,
  * a Noul fact that is explicitly *negated*,
  * a Noul fact that is stated about a *different entity* than the one asked
    about,
  * a Choice enum extraction whose distractor options are literal substrings /
    reorderings of the gold value,
  * a multi-field lookup (two questions answered from two different fields).
"""
from . import core

FAMILY = "facts_extraction"

ORDER_STATUS = ["processing", "packed", "shipped", "in transit", "delivered", "returned"]
TICKET_PRIORITY = ["low", "normal", "high", "urgent"]
TICKET_STATUS = ["open", "pending", "on hold", "solved", "closed"]
INVOICE_STATUS = ["draft", "sent", "part paid", "paid", "overdue", "written off"]
SHIFT_ROLE = ["nurse", "cashier", "driver", "technician", "porter", "dispatcher"]
LOG_LEVEL = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
SERVICES = ["auth", "billing-api", "scheduler", "ingest", "search", "notifier", "gateway"]
CLINICS = ["cardiology", "dermatology", "orthopaedics", "audiology", "respiratory", "endocrine"]
REFERRALS = ["general practitioner", "self referral", "emergency department", "internal transfer"]
INSURERS = ["Steepleton Mutual", "Harrowgate Health", "Vellmont Assurance", "Cobalt Care",
            "Fernbank Provident", "Larkspur Benefit"]
SITES = ["north depot", "riverside plant", "central clinic", "east warehouse", "harbour office"]


def _order(ctx, rng):
    oid = "ORD-%d" % (10000 + ctx.slot(rng, "mid") * 11 + ctx.slot(rng, "small"))
    ent = {
        "order_id": oid,
        "customer": ctx.slot(rng, "company"),
        "status": rng.choice(ORDER_STATUS),
        "carrier": ctx.slot(rng, "carrier"),
        "placed_on": ctx.date(rng),
        "total": ctx.slot(rng, "amount"),
        "paid": rng.random() < 0.6,
        "items": [{"sku": "SKU-%d" % (100 + ctx.slot(rng, "small")),
                   "description": ctx.slot(rng, "product"),
                   "qty": 1 + ctx.slot(rng, "small") % 5} for _ in range(1 + rng.randrange(4))],
    }
    lines = [
        ("order_id", "Order %s was placed on %s by %s." % (oid, core.pretty_date(ent["placed_on"]), ent["customer"])),
        ("status", "The current status of order %s is %s." % (oid, ent["status"])),
        ("carrier", "The consignment is being carried by %s." % ent["carrier"]),
        ("total", "The order total is $%.2f." % ent["total"]),
        ("paid", "Payment for the order has %sbeen received." % ("" if ent["paid"] else "not ")),
    ]
    return ent, lines, "order %s" % oid, "ticket"


def _ticket(ctx, rng):
    tid = ctx.ticket_id(rng)
    ent = {
        "ticket_id": tid,
        "requester": ctx.slot(rng, "person"),
        "channel": ctx.slot(rng, "channel"),
        "priority": rng.choice(TICKET_PRIORITY),
        "status": rng.choice(TICKET_STATUS),
        "assignee": ctx.slot(rng, "agent"),
        "opened_on": ctx.date(rng),
        "replies": 1 + ctx.slot(rng, "small") % 7,
    }
    lines = [
        ("ticket_id", "Ticket %s was opened on %s by %s via %s." % (tid, core.pretty_date(ent["opened_on"]), ent["requester"], ent["channel"])),
        ("priority", "The ticket carries %s priority." % ent["priority"]),
        ("status", "Its status is recorded as %s." % ent["status"]),
        ("assignee", "The conversation is assigned to %s." % ent["assignee"]),
        ("replies", "There have been %d replies so far." % ent["replies"]),
    ]
    return ent, lines, "ticket %s" % tid, "ticket"


def _invoice(ctx, rng):
    num = "INV-%d" % (20000 + ctx.slot(rng, "mid") * 13 + ctx.slot(rng, "small"))
    issued = ctx.date(rng)
    ent = {
        "invoice_number": num,
        "customer": ctx.slot(rng, "company"),
        "issued_on": issued,
        "due_on": core.date_add(issued, rng.choice([14, 30, 45, 60])),
        "total": ctx.slot(rng, "amount"),
        "currency": rng.choice(["USD", "EUR", "GBP"]),
        "status": rng.choice(INVOICE_STATUS),
        "purchase_order": "PO-%d" % (5000 + ctx.slot(rng, "mid")),
    }
    lines = [
        ("invoice_number", "Invoice %s was issued to %s on %s." % (num, ent["customer"], core.pretty_date(issued))),
        ("total", "The amount payable is %s %.2f." % (ent["currency"], ent["total"])),
        ("due_on", "Payment falls due on %s." % core.pretty_date(ent["due_on"])),
        ("status", "The invoice is marked %s in the ledger." % ent["status"]),
        ("purchase_order", "It quotes purchase order %s." % ent["purchase_order"]),
    ]
    return ent, lines, "invoice %s" % num, "handbook"


def _shift(ctx, rng):
    ent = {
        "employee": ctx.slot(rng, "person"),
        "site": rng.choice(SITES),
        "date": ctx.date(rng),
        "role": rng.choice(SHIFT_ROLE),
        "start": "%02d:00" % (5 + ctx.slot(rng, "small") % 8),
        "hours": 4 + ctx.slot(rng, "small") % 8,
        "break_minutes": rng.choice([0, 15, 20, 30, 45]),
        "overtime": rng.random() < 0.4,
    }
    lines = [
        ("employee", "%s is rostered at the %s on %s." % (ent["employee"], ent["site"], core.pretty_date(ent["date"]))),
        ("role", "The shift is a %s shift." % ent["role"]),
        ("hours", "It runs for %d hours from %s." % (ent["hours"], ent["start"])),
        ("break_minutes", "A break of %d minutes is scheduled." % ent["break_minutes"]),
        ("overtime", "Overtime has %sbeen approved for this shift." % ("" if ent["overtime"] else "not ")),
    ]
    return ent, lines, "the shift of %s" % ent["employee"], "meeting"


def _inventory(ctx, rng):
    sku = "SKU-%d" % (700 + ctx.slot(rng, "mid"))
    ent = {
        "sku": sku,
        "description": ctx.slot(rng, "product"),
        "location": "%s depot" % ctx.slot(rng, "city"),
        "on_hand": ctx.slot(rng, "mid"),
        "reserved": ctx.slot(rng, "small"),
        "reorder_point": ctx.slot(rng, "small") + 10,
        "supplier": ctx.slot(rng, "company"),
        "last_counted": ctx.date(rng),
    }
    lines = [
        ("sku", "Item %s is the %s held at the %s." % (sku, ent["description"], ent["location"])),
        ("on_hand", "There are %d units on hand." % ent["on_hand"]),
        ("reserved", "Of those, %d units are reserved for open orders." % ent["reserved"]),
        ("supplier", "The item is supplied by %s." % ent["supplier"]),
        ("last_counted", "The last stock count took place on %s." % core.pretty_date(ent["last_counted"])),
    ]
    return ent, lines, "item %s" % sku, "inventory"


def _patient(ctx, rng):
    mrn = "MRN-%d" % (400000 + ctx.slot(rng, "mid") * 17)
    ent = {
        "mrn": mrn,
        "patient": ctx.slot(rng, "person"),
        "clinic": rng.choice(CLINICS),
        "appointment_on": ctx.date(rng),
        "referral_source": rng.choice(REFERRALS),
        "insurer": rng.choice(INSURERS),
        "consent_on_file": rng.random() < 0.6,
        "interpreter_required": rng.random() < 0.3,
    }
    lines = [
        ("mrn", "Record %s belongs to %s." % (mrn, ent["patient"])),
        ("clinic", "The appointment is with the %s clinic on %s." % (ent["clinic"], core.pretty_date(ent["appointment_on"]))),
        ("referral_source", "The referral came from the %s." % ent["referral_source"]),
        ("insurer", "Cover is provided by %s." % ent["insurer"]),
        ("consent_on_file", "Written consent is %son file." % ("" if ent["consent_on_file"] else "not ")),
    ]
    return ent, lines, "record %s" % mrn, "handbook"


def _serverlog(ctx, rng):
    rid = "req-%x" % (0x100000 + ctx.slot(rng, "mid") * 977)
    ent = {
        "request_id": rid,
        "host": "%s-%02d" % (ctx.slot(rng, "city").lower(), ctx.slot(rng, "small") % 20),
        "service": rng.choice(SERVICES),
        "level": rng.choice(LOG_LEVEL),
        "status_code": rng.choice([200, 201, 304, 400, 404, 409, 429, 500, 503]),
        "duration_ms": ctx.slot(rng, "mid") * 3,
        "retries": ctx.slot(rng, "small") % 4,
        "region": ctx.slot(rng, "city"),
    }
    lines = [
        ("request_id", "Request %s was handled by host %s." % (rid, ent["host"])),
        ("service", "The entry was written by the %s service." % ent["service"]),
        ("level", "It was logged at level %s." % ent["level"]),
        ("status_code", "The response status was %d." % ent["status_code"]),
        ("duration_ms", "The call took %d ms and was retried %d times." % (ent["duration_ms"], ent["retries"])),
    ]
    return ent, lines, "request %s" % rid, "log"


TYPES = [
    ("order", _order, "status", ORDER_STATUS, ["paid"], "customer"),
    ("ticket", _ticket, "priority", TICKET_PRIORITY, [], "assignee"),
    ("invoice", _invoice, "status", INVOICE_STATUS, [], "customer"),
    ("shift", _shift, "role", SHIFT_ROLE, ["overtime"], "employee"),
    ("inventory", _inventory, "location", None, [], "supplier"),
    ("patient", _patient, "clinic", CLINICS, ["consent_on_file", "interpreter_required"], "insurer"),
    ("serverlog", _serverlog, "level", LOG_LEVEL, [], "service"),
]

ABSENT_FIELDS = {
    "order": [("gift_wrap", "Was gift wrapping requested for this order?"),
              ("signature_required", "Does the record state that a signature is required on delivery?"),
              ("discount_code", "Was a discount code applied to this order?")],
    "ticket": [("csat_score", "Does the record contain a customer satisfaction score?"),
               ("due_by", "Is a resolution deadline recorded for this ticket?"),
               ("phone_callback", "Is a phone callback recorded for this ticket?")],
    "invoice": [("late_fee", "Has a late fee been applied to this invoice?"),
                ("credit_note", "Is a credit note recorded against this invoice?"),
                ("dispute_flag", "Is this invoice marked as disputed?")],
    "shift": [("night_allowance", "Is a night allowance recorded for this shift?"),
              ("swap_requested", "Has a shift swap been requested?"),
              ("training_cover", "Is this shift marked as training cover?")],
    "inventory": [("quarantined", "Are any of these units quarantined?"),
                  ("batch_recall", "Is a batch recall recorded for this item?"),
                  ("serial_tracked", "Is this item serial tracked?")],
    "patient": [("transport_booked", "Is patient transport booked for this appointment?"),
                ("allergy_alert", "Is an allergy alert recorded on this record?"),
                ("prior_dna", "Does the record note a previously missed appointment?")],
    "serverlog": [("alert_paged", "Did this entry page an on-call engineer?"),
                  ("sampled", "Is this entry marked as sampled?"),
                  ("client_cancelled", "Does the entry state the client cancelled the request?")],
}

NEGATED_TMPL = {
    "order": ("No gift wrapping was requested for this order.", "Was gift wrapping requested for this order?"),
    "ticket": ("No callback was ever scheduled for this conversation.", "Was a callback scheduled for this ticket?"),
    "invoice": ("No late fee has been applied to this invoice.", "Has a late fee been applied to this invoice?"),
    "shift": ("No night allowance applies to this shift.", "Does a night allowance apply to this shift?"),
    "inventory": ("None of these units are quarantined.", "Are any of these units quarantined?"),
    "patient": ("Patient transport has not been booked for this appointment.", "Is patient transport booked for this appointment?"),
    "serverlog": ("The entry did not page the on-call engineer.", "Did this entry page an on-call engineer?"),
}

OTHER_ENTITY_TMPL = {
    "order": ("A separate order %s was gift wrapped at the customer's request.", "Was gift wrapping requested for %s?"),
    "ticket": ("A callback was scheduled on the neighbouring ticket %s instead.", "Was a callback scheduled for %s?"),
    "invoice": ("A late fee was applied to the earlier invoice %s.", "Has a late fee been applied to %s?"),
    "shift": ("A night allowance was approved for the shift of %s.", "Does a night allowance apply to %s?"),
    "inventory": ("Units of the unrelated item %s are quarantined.", "Are any units of %s quarantined?"),
    "patient": ("Transport is booked for the appointment on record %s.", "Is patient transport booked for %s?"),
    "serverlog": ("The on-call engineer was paged by request %s.", "Did %s page the on-call engineer?"),
}

MULTI_Q = {
    "order": [("carrier", "Which carrier is named in the record?"), ("customer", "Which customer placed the order?")],
    "ticket": [("assignee", "Who is the conversation assigned to?"), ("requester", "Who opened the ticket?")],
    "invoice": [("customer", "Who was the invoice issued to?"), ("purchase_order", "Which purchase order is quoted?")],
    "shift": [("site", "At which site is the shift worked?"), ("employee", "Who is rostered for the shift?")],
    "inventory": [("supplier", "Who supplies this item?"), ("description", "What is the item?")],
    "patient": [("insurer", "Which insurer covers the patient?"), ("clinic", "Which clinic is the appointment with?")],
    "serverlog": [("host", "Which host handled the request?"), ("service", "Which service wrote the entry?")],
}


def _overlapping(rng, value, others):
    """Distractor options that literally overlap the gold string."""
    cands = {value: None}
    w = str(value).split()
    if len(w) >= 2:
        cands[" ".join(w[:-1])] = None
        cands[" ".join(reversed(w))] = None
        cands[w[-1]] = None
    else:
        s = str(value)
        cands[s.upper() if s.islower() else s.lower()] = None
        cands[s + "-2"] = None
    for o in others:
        if str(o) != str(value):
            cands[str(o)] = None
        if len(cands) >= 6:
            break
    return cands


def scenarios(ctx, limit=None):
    emitted = 0
    rep = 0
    while True:
        for tname, builder, enum_field, enum_pool, bool_fields, other_field in TYPES:
            ctx.reset_tids()
            rng = ctx.rng("fx", tname, rep)
            ent, lines, subject, padfam = builder(ctx, rng)
            group = "fx/%s/%d" % (tname, rep)
            rid = "%s-fx-%s-%d" % (ctx.stream, tname, rep)
            text = core.sent_join([s for _f, s in lines])
            recs = []

            # 1. stated fact (JSON state)
            fname, fq = rng.choice([(f, s) for f, s in lines if f in ent])[0], None
            sf = rng.choice([f for f, _ in lines])
            sval = ent[sf]
            if isinstance(sval, bool):
                sq = "Does the record state that this is true: `%s`?" % sf
                gold_stated = bool(sval)
                mode = "stated" if sval else "negated"
            else:
                sq = "Does the record state a value for `%s`?" % sf
                gold_stated = True
                mode = "stated"
            q = {"type": "noul", "instructions": sq,
                 "criteria": {"true": "The record states it", "false": "The record does not state it"}}
            recs.append(ctx.rec(rid + "-stated", FAMILY, "facts_extraction", "stated", group,
                                {tname: ent}, {"fact": q}, {"fact": gold_stated}, "easy", "lexical",
                                [{"qid": "fact", "rule": "fact_mode", "args": {"mode": mode}}],
                                evidence=[]))

            # 2. absent fact
            afield, aq = rng.choice(ABSENT_FIELDS[tname])
            q2 = {"type": "noul", "instructions": aq}
            recs.append(ctx.rec(rid + "-absent", FAMILY, "facts_extraction", "absent", group,
                                {tname: ent}, {"fact": q2}, {"fact": False}, "standard", "lexical",
                                [{"qid": "fact", "rule": "fact_mode",
                                  "args": {"mode": "absent", "field": afield}}], evidence=[]))

            # 3. explicitly negated fact (text state)
            nsent, nq_text = NEGATED_TMPL[tname]
            ntext = core.sent_join([text, nsent])
            q3 = {"type": "noul", "instructions": nq_text,
                  "criteria": {"true": "The record states that it happened",
                               "false": "The record states that it did not happen, or says nothing"}}
            recs.append(ctx.rec(rid + "-negated", FAMILY, "facts_extraction", "negated", group,
                                ntext, {"fact": q3}, {"fact": False}, "hard", "negation",
                                [{"qid": "fact", "rule": "fact_mode", "args": {"mode": "negated"}}],
                                evidence=[nsent]))

            # 4. fact stated about a different entity
            osent_t, oq_t = OTHER_ENTITY_TMPL[tname]
            other_subject = {
                "order": "ORD-%d" % (90000 + ctx.slot(rng, "mid")),
                "ticket": ctx.ticket_id(rng),
                "invoice": "INV-%d" % (70000 + ctx.slot(rng, "mid")),
                "shift": ctx.slot(rng, "person"),
                "inventory": "SKU-%d" % (900 + ctx.slot(rng, "mid")),
                "patient": "MRN-%d" % (800000 + ctx.slot(rng, "mid")),
                "serverlog": "req-%x" % (0x900000 + ctx.slot(rng, "mid")),
            }[tname]
            osent = osent_t % other_subject
            otext = core.sent_join([text, osent])
            q4 = {"type": "noul", "instructions": oq_t % subject}
            recs.append(ctx.rec(rid + "-other-entity", FAMILY, "facts_extraction", "other_entity",
                                group, otext, {"fact": q4}, {"fact": False}, "hard", "multi_hop",
                                [{"qid": "fact", "rule": "fact_mode",
                                  "args": {"mode": "other_entity", "asked_about": subject,
                                           "stated_about": other_subject}}],
                                evidence=[osent]))

            # 5. enum extraction with overlapping literal candidates
            pool = enum_pool if enum_pool else [ent[enum_field], "%s depot" % ctx.slot(rng, "city"),
                                                "%s depot" % ctx.slot(rng, "city")]
            crit = _overlapping(rng, ent[enum_field], [p for p in pool if p != ent[enum_field]])
            crit = core.shuffled_map(rng, crit)
            state5 = {"source_text": text} if rng.random() < 0.5 else text
            instr5 = {"field": {"name": enum_field, "type": "string"},
                      "question": "Which option is the value of `field` in `source_text`?"} \
                if isinstance(state5, dict) else "Which option is the %s recorded for %s?" % (enum_field.replace("_", " "), subject)
            q5 = {"type": "choice", "instructions": instr5, "criteria": crit}
            ev5 = [s for f, s in lines if f == enum_field] or [text.split(".")[0] + "."]
            recs.append(ctx.rec(rid + "-enum", FAMILY, "facts_extraction", "enum_overlap", group,
                                state5, {"value": q5}, {"value": str(ent[enum_field])},
                                "standard", "lexical",
                                [{"qid": "value", "rule": "lookup",
                                  "args": {"record": {enum_field: str(ent[enum_field])},
                                           "path": enum_field, "options": sorted(crit)}}],
                                evidence=ev5))

            # 6. multi-field lookup (two questions from two fields)
            (f1, qt1), (f2, qt2) = MULTI_Q[tname]
            c1 = core.shuffled_map(rng, _overlapping(rng, ent[f1], [ent[f2], subject]))
            c2 = core.shuffled_map(rng, _overlapping(rng, ent[f2], [ent[f1], subject]))
            qs = {"a": {"type": "choice", "instructions": qt1, "criteria": c1},
                  "b": {"type": "choice", "instructions": qt2, "criteria": c2}}
            state6 = {tname: ent} if rng.random() < 0.5 else text
            recs.append(ctx.rec(rid + "-multi", FAMILY, "facts_extraction", "multi_field", group,
                                state6, qs, {"a": str(ent[f1]), "b": str(ent[f2])},
                                "hard", "multi_hop",
                                [{"qid": "a", "rule": "lookup",
                                  "args": {"record": {f1: str(ent[f1])}, "path": f1, "options": sorted(c1)}},
                                 {"qid": "b", "rule": "lookup",
                                  "args": {"record": {f2: str(ent[f2])}, "path": f2, "options": sorted(c2)}}],
                                evidence=[]))

            tids = sorted(set(ctx.cur_tids))
            for r in recs:
                r["meta"]["template_ids"] = sorted(set(tids + r["meta"]["template_ids"]))
                r["meta"]["pad_family"] = padfam
                r["meta"]["record_type"] = tname
            emitted += len(recs)
            yield recs
            if limit and emitted >= limit:
                return
        rep += 1
        if rep > 3000:
            return
