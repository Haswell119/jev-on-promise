#!/usr/bin/env python3
"""Shared infrastructure for the Sextant internal benchmark + training generator.

Everything here is deterministic: all randomness comes from `random.Random`
instances seeded with a sha256 of (global seed, side, stream, key parts).

The central idea is the *side*: every template list and every slot vocabulary is
halved deterministically into a `bench` pool and a `train` pool (`split_pool`).
The frozen internal benchmark is generated only from the `bench` pools and the
training pool only from the `train` pools, so no template, no slot value and
therefore no state string can leak between them.
"""
import hashlib
import json
import random
import re

MODEL = "sextant-1"
LICENSE = "Apache-2.0 (generated)"
SOURCE_PREFIX = "synthetic/bench"

DIFFICULTIES = ("easy", "standard", "hard", "judge")
REASONINGS = (
    "lexical", "paraphrase", "negation", "implicit", "multi_hop", "numeric",
    "temporal", "policy", "long_context", "adequacy", "ordinal", "adversarial",
)
POSITIONS = ("start", "middle", "end", "split", "na")
SIDES = ("bench", "train")

# states shorter than this are not considered to have a meaningful evidence
# position (there is nowhere to hide the evidence).
POSITION_MIN_TOKENS = 48


# --------------------------------------------------------------------------- hashing / pools

def hkey(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hint(s: str, mod: int) -> int:
    return int(hkey(s)[:12], 16) % mod


_POOL_CACHE = {}
POOL_SIZES = {}   # pool key -> number of items (registry used by verify.py)


def pool_sides(key: str, n: int):
    """Rank the indices of a pool by hash and alternate sides -> exact halves."""
    order = sorted(range(n), key=lambda i: hkey("%s#%d" % (key, i)))
    sides = [None] * n
    for rank, i in enumerate(order):
        sides[i] = "bench" if rank % 2 == 0 else "train"
    return sides


def split_pool(key: str, items, side: str):
    """Return the `side` half of `items` as a list of (template_id, item)."""
    POOL_SIZES[key] = len(items)
    ck = (key, side, len(items))
    hit = _POOL_CACHE.get(ck)
    if hit is not None:
        return hit
    sides = pool_sides(key, len(items))
    out = [("%s#%d" % (key, i), items[i]) for i in range(len(items)) if sides[i] == side]
    if not out:
        raise ValueError("empty %s pool for %r (n=%d)" % (side, key, len(items)))
    _POOL_CACHE[ck] = out
    return out


def all_template_ids(key: str, n: int, side: str):
    sides = pool_sides(key, n)
    return ["%s#%d" % (key, i) for i in range(n) if sides[i] == side]


# --------------------------------------------------------------------------- text helpers

def state_text(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def count_tokens(state) -> int:
    return len(state_text(state).split())


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def evidence_position(state, evidence) -> str:
    """Where the decisive evidence sits inside the state (re-derivable)."""
    txt = state_text(state)
    if not evidence or not txt:
        return "na"
    if len(txt.split()) < POSITION_MIN_TOKENS:
        return "na"
    n = float(len(txt))
    rel = []
    for e in evidence:
        i = txt.find(e)
        if i < 0:
            return "na"
        rel.append((i + len(e) / 2.0) / n)
    if len(rel) >= 2 and (max(rel) - min(rel)) > 0.34:
        return "split"
    c = sum(rel) / len(rel)
    if c < 1.0 / 3.0:
        return "start"
    if c < 2.0 / 3.0:
        return "middle"
    return "end"


def shuffled_map(rng, d):
    items = list(d.items())
    rng.shuffle(items)
    return dict(items)


def sent_join(parts):
    return " ".join(p.strip() for p in parts if p and p.strip())


# --------------------------------------------------------------------------- shared slot vocabularies
# Every list below is halved by `split_pool`, so the bench side and the train
# side never share a slot value.

COMPANIES = [
    "Beaver Dam Logistics", "Northwind Traders", "Blue Harbor Dental", "Ridgeway Farms",
    "Kestrel Analytics", "Maple & Oak Furniture", "Sunset Bakery", "Orion Shipping",
    "Pinecrest Clinic", "Halbrook Textiles", "Vantage Print Works", "Saltmarsh Brewing",
    "Ironwood Fabrication", "Clearwater Utilities", "Lantern Bay Media", "Foxglove Nursery",
    "Granite Peak Tooling", "Willowmere Estates", "Cobalt Ridge Software", "Harrow Lane Books",
    "Thistledown Catering", "Amberline Freight", "Quarry Street Motors", "Fernbank Pharmacy",
    "Steepleton Insurance", "Larkspur Robotics", "Dunmore Paper", "Windlass Marine",
    "Cinderhill Foundry", "Brackenfield Dairy", "Selwyn Optics", "Turnstile Retail Group",
]
PEOPLE = [
    "A. Lee", "M. Garcia", "S. Okafor", "T. Novak", "R. Dubois", "H. Yamada", "P. Andersen",
    "K. Thorne", "J. Marek", "L. Oyelaran", "D. Fitzgerald", "N. Brandt", "C. Villanueva",
    "E. Kowalski", "F. Adeyemi", "G. Lindqvist", "B. Sorensen", "V. Petrov", "W. Achebe",
    "Y. Haddad", "Z. Moreau", "I. Karlsson", "O. Nakamura", "Q. Bellweather",
]
PRODUCTS = [
    "standing desk", "wireless headset", "coffee grinder", "winter jacket", "laptop stand",
    "garden hose", "running shoe", "office chair", "smart thermostat", "camping tent",
    "espresso machine", "label printer", "cordless drill", "air purifier", "bike helmet",
    "water filter", "desk lamp", "yoga mat", "electric kettle", "travel backpack",
    "monitor arm", "paper shredder", "induction hob", "sewing machine",
]
CITIES = [
    "Ashford", "Brenton", "Calderwood", "Dunmore", "Eastvale", "Fairholt", "Garnet Falls",
    "Hollisburg", "Inverleigh", "Jessup", "Kirkhaven", "Lindenmere", "Mossgate", "Northfield",
    "Oakmere", "Portbury", "Quinnwood", "Redlake", "Stonebridge", "Thornhill", "Upperton",
    "Vellmont", "Westmarch", "Yarrowdale",
]
CARRIERS = ["Meridian Freight", "Arrowpost", "Trellis Courier", "Bluewater Express",
            "Northgate Parcel", "Sandpiper Delivery", "Kingfisher Logistics", "Redwing Transit"]
CHANNELS = ["email", "web chat", "phone", "in-app message", "support portal", "SMS", "partner portal", "voicemail"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AGENTS = ["Dana", "Miro", "Priya", "Tomas", "Ines", "Kofi", "Lena", "Ravi",
          "Signe", "Emeka", "Bruno", "Nadia", "Oskar", "Chiara", "Jonas", "Amara"]

# disjoint numeric ladders: the bench half and the train half of each list never
# share a value, so no numeric slot can collide either.
SMALL_INTS = list(range(2, 66))
MID_INTS = list(range(70, 400, 3))
AMOUNTS = [round(v, 2) for v in
           [18.4, 29.95, 42.0, 57.5, 63.25, 84.9, 99.99, 118.0, 137.4, 164.5, 189.9, 212.0,
            248.75, 279.0, 312.6, 355.2, 401.0, 448.9, 512.3, 588.0, 640.45, 712.8, 799.0,
            864.2, 940.6, 1024.0, 1188.5, 1290.0, 1444.75, 1610.0, 1808.4, 2020.0, 2244.9,
            2480.0, 2760.5, 3050.0, 3388.25, 3740.0, 4128.6, 4560.0, 5032.5, 5560.0, 6144.0,
            6780.4, 7488.0, 8260.5, 9120.0, 10060.0, 11104.5, 12250.0, 13520.0, 14910.0,
            16440.0, 18130.0, 20000.0, 22060.5, 24330.0, 26840.0, 29600.0, 32660.0, 36020.0,
            39740.0, 43840.0, 48360.0, 53340.0, 58840.0, 64900.0, 71580.0, 78960.0, 87080.0,
            96040.0, 105920.0, 116840.0, 128880.0, 142160.0, 156800.0, 172900.0, 190700.0]]
TICKET_PREFIX = ["TCK", "REQ", "CASE", "INC", "SR", "HD", "OPS", "SUP"]


def ymd(y, m, d):
    return "%04d-%02d-%02d" % (y, m, d)


_MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def to_ord(y, m, d):
    """Days since 2000-01-01 on a proleptic 365/366 calendar (no leap-day edge cases:
    generators only ever use day <= 28)."""
    days = 0
    for yy in range(2000, y):
        days += 366 if (yy % 4 == 0 and (yy % 100 != 0 or yy % 400 == 0)) else 365
    for mm in range(1, m):
        days += _MDAYS[mm - 1] + (1 if (mm == 2 and y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 0)
    return days + d


def from_ord(n):
    y = 2000
    while True:
        ylen = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        if n > ylen:
            n -= ylen
            y += 1
        else:
            break
    m = 1
    while True:
        mlen = _MDAYS[m - 1] + (1 if (m == 2 and y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 0)
        if n > mlen:
            n -= mlen
            m += 1
        else:
            break
    return y, m, n


def date_add(iso, days):
    y, m, d = (int(x) for x in iso.split("-"))
    return ymd(*from_ord(to_ord(y, m, d) + days))


def pretty_date(iso):
    y, m, d = (int(x) for x in iso.split("-"))
    return "%s %d, %d" % (MONTHS[m - 1], d, y)


# --------------------------------------------------------------------------- padding corpora (long context)
# Plausible same-domain filler. Slots are filled from side-specific pools, so a
# padded bench document can never coincide with a padded training document.

PAD_TICKET = [
    "Ticket {tid} was opened by {person} on {date} through the {channel} channel and reassigned twice before the shift handover.",
    "Follow-up note on {tid}: {agent} called {person} back, left a voicemail and set a reminder for the following business day.",
    "{agent} merged {tid} into the weekly digest because the reporter never replied to the clarification request.",
    "The queue report for {date} lists {n} open conversations, of which {m} were waiting on a customer response.",
    "A macro was applied to {tid} that pastes the standard acknowledgement and sets the priority back to normal.",
    "{person} confirmed the contact details on file for {company} and asked that future updates go to the shared mailbox.",
    "Quality review of {tid} flagged the tone as acceptable but noted that the agent skipped the identity check.",
    "Escalation policy reminder: conversations untouched for {n} hours are surfaced on the team dashboard in {city}.",
]
PAD_HANDBOOK = [
    "Section {n}.{m} of the handbook describes the escalation ladder and the sign-off required for discretionary gestures.",
    "Appendix {m} lists the approved wording for acknowledgements and the retention period for recorded calls.",
    "The {city} office keeps a printed copy of the handbook at reception; the electronic version supersedes it whenever they differ.",
    "Colleagues at {company} should complete the annual refresher before the end of the {month} training window.",
    "Paragraph {n} reiterates that discretionary decisions must be recorded with the reviewer's initials and the date.",
    "The glossary defines a business day as any weekday that is not a public holiday in the {city} region.",
    "Chapter {n} covers document handling; nothing in it changes the thresholds published elsewhere in this handbook.",
    "Training records for {agent} show the compliance module was completed in {month}.",
]
PAD_MEETING = [
    "Meeting notes, {date}: attendance was {n} people; the previous actions were reviewed and three were carried forward.",
    "{agent} presented the {month} volume figures and noted that the backlog in {city} is roughly flat week on week.",
    "Action: {agent} to circulate the revised rota for {company} before the next planning session.",
    "The group agreed to postpone the tooling discussion until the procurement review for the {product} concludes.",
    "Under any other business, {agent} raised the parking arrangements at the {city} site; no decision was taken.",
    "A short update on the {product} pilot: {n} participants enrolled and {m} completed the feedback form.",
    "The chair reminded everyone that minutes are informal and never override written policy.",
]
PAD_LOG = [
    "{date}T0{n}:{m}:12Z INFO scheduler: nightly job 'rollup-{n}' finished in {m}s with no warnings.",
    "{date}T1{n}:0{m}:47Z DEBUG cache: evicted {n}00 keys from the {city} region shard during routine compaction.",
    "{date}T0{m}:{n}:03Z INFO auth: {n} sessions refreshed for tenant {company} without incident.",
    "{date}T2{m}:1{n}:55Z WARN queue: consumer lag reached {n}00 messages before recovering on its own.",
    "{date}T0{n}:4{m}:09Z INFO deploy: build {n}{m} promoted to staging by {agent}; smoke tests green.",
    "{date}T1{m}:2{n}:31Z INFO backup: snapshot of the {product} inventory table completed, {n}.{m} GiB written.",
]
PAD_INVENTORY = [
    "Stock take on {date} counted {n} units of the {product} at the {city} depot with no variance against the system.",
    "The {product} is supplied by {company} on a {n}-day replenishment cycle; the reorder point is reviewed each {month}.",
    "Damaged goods bay currently holds {m} items awaiting disposition; none belong to the case under review.",
    "A cycle count scheduled for {date} covers aisles {n} through {m} and does not include returns processing.",
    "Transfer note: {n} cartons moved from the {city} depot to overflow storage, signed off by {agent}.",
]
PAD_POLICY_FILLER = [
    "Nothing in this clause affects the statutory rights of the customer.",
    "Where this document conflicts with a signed contract, the contract prevails for that customer only.",
    "Requests submitted through unsupported channels are re-entered by the service desk before the clock starts.",
    "The committee reviews these provisions every {n} months and publishes changes in the {month} bulletin.",
    "Definitions used in this section have the meaning given in the glossary unless stated otherwise.",
    "Records relating to decisions under this section are retained for {n} years.",
]

PAD_FAMILIES = {
    "ticket": PAD_TICKET,
    "handbook": PAD_HANDBOOK,
    "meeting": PAD_MEETING,
    "log": PAD_LOG,
    "inventory": PAD_INVENTORY,
    "policy": PAD_POLICY_FILLER,
}

SECTION_TITLES = [
    "Case history", "Correspondence log", "Internal notes", "Reference material",
    "Operational excerpt", "Background", "Appendix", "Handover summary",
    "Audit trail", "Supplementary notes", "Prior context", "Working file",
]


# --------------------------------------------------------------------------- context

class Ctx(object):
    """Generation context: a side (bench|train), a stream name and a seed."""

    def __init__(self, seed, side, stream):
        assert side in SIDES
        self.seed = seed
        self.side = side
        self.stream = stream
        self.template_ids = set()
        self.cur_tids = []

    def use(self, tid):
        self.template_ids.add(tid)
        self.cur_tids.append(tid)
        return tid

    # -- randomness ---------------------------------------------------------
    def rng(self, *parts):
        key = "|".join([str(self.seed), self.side, self.stream] + [str(p) for p in parts])
        return random.Random(hkey(key))

    # -- pools --------------------------------------------------------------
    def pool(self, key, items):
        return split_pool(key, items, self.side)

    def pick(self, rng, key, items):
        """Pick one (template_id, item) from the side-specific half of `items`."""
        return rng.choice(self.pool(key, items))

    def pick_val(self, rng, key, items):
        tid, val = self.pick(rng, key, items)
        self.use(tid)
        return val

    def sample(self, rng, key, items, k):
        p = self.pool(key, items)
        k = min(k, len(p))
        return rng.sample(p, k)

    # -- slots --------------------------------------------------------------
    def slot(self, rng, name):
        table = {
            "company": ("slot/company", COMPANIES),
            "person": ("slot/person", PEOPLE),
            "product": ("slot/product", PRODUCTS),
            "city": ("slot/city", CITIES),
            "carrier": ("slot/carrier", CARRIERS),
            "channel": ("slot/channel", CHANNELS),
            "agent": ("slot/agent", AGENTS),
            "month": ("slot/month", MONTHS),
            "weekday": ("slot/weekday", WEEKDAYS),
            "small": ("slot/small", SMALL_INTS),
            "mid": ("slot/mid", MID_INTS),
            "amount": ("slot/amount", AMOUNTS),
            "tprefix": ("slot/tprefix", TICKET_PREFIX),
        }
        key, items = table[name]
        tid, val = rng.choice(self.pool(key, items))
        self.use(tid)
        return val

    def ticket_id(self, rng):
        return "%s-%d" % (self.slot(rng, "tprefix"), 1000 + self.slot(rng, "mid") * 7 + self.slot(rng, "small"))

    def date(self, rng, year=2026):
        # bench and train draw months/days from disjoint halves of the pools
        m = MONTHS.index(self.slot(rng, "month")) + 1
        d = 1 + (self.slot(rng, "small") % 28)
        return ymd(year, m, d)

    # -- padding ------------------------------------------------------------
    def pad_sentence(self, rng, family):
        tmpl = self.slot_tmpl(rng, family)
        return tmpl.format(
            tid=self.ticket_id(rng), person=self.slot(rng, "person"), date=self.date(rng),
            channel=self.slot(rng, "channel"), agent=self.slot(rng, "agent"),
            company=self.slot(rng, "company"), city=self.slot(rng, "city"),
            product=self.slot(rng, "product"), month=self.slot(rng, "month"),
            n=self.slot(rng, "small"), m=1 + (self.slot(rng, "small") % 9),
        )

    def slot_tmpl(self, rng, family):
        items = PAD_FAMILIES[family]
        tid, t = rng.choice(self.pool("pad/" + family, items))
        self.use(tid)
        return t

    # -- record -------------------------------------------------------------
    def rec(self, rid, family, recipe, transformation, group, state, questions, gold,
            difficulty, reasoning, checks, evidence=None, position=None, gold_probs=None,
            extra=None):
        assert difficulty in DIFFICULTIES, difficulty
        assert reasoning in REASONINGS, reasoning
        pos = position if position is not None else evidence_position(state, evidence or [])
        assert pos in POSITIONS, pos
        r = {
            "id": rid,
            "source": "%s/%s" % (SOURCE_PREFIX, family),
            "license": LICENSE,
            "split": self.stream,
            "tier": difficulty,
            "family": family,
            "synthetic": True,
            "transformation": transformation,
            "group": group,
            "recipe": recipe,
            "difficulty": difficulty,
            "reasoning": reasoning,
            "state_tokens": count_tokens(state),
            "evidence_position": pos,
            "request": {"model": MODEL, "state": state, "questions": questions},
            "gold": gold,
            "meta": {
                "side": self.side,
                "checks": checks,
                "template_ids": sorted(set(self.cur_tids)),
                "evidence": list(evidence or []),
            },
        }
        if gold_probs:
            r["gold_probs"] = gold_probs
        if extra:
            r.update(extra)
        return r

    def reset_tids(self):
        self.cur_tids = []
