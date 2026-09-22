#!/usr/bin/env python3
"""Verifier for the frozen internal benchmark and the synthetic training pool.

    python3 scripts/synth/bench/verify.py

Checks
  (a) every gold label is re-derived from the record's own generation metadata
      (`meta.checks`) by an implementation written independently of the
      generators, and any mismatch fails the run;
  (b) no dev / shadow / calib / long-context state string occurs in the
      training pool, compared both exactly and after normalisation
      (case-folded, punctuation stripped, whitespace collapsed);
  (c) the template pools are disjoint: no template id is used by both sides,
      and every template id in a record belongs to the side of its file;
  (d) structural invariants (gold inside the option set, score index in range,
      noul boolean, token counts, evidence present and positioned as recorded,
      unique ids, injected directives never agreeing with the gold);
  and prints the counts table.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench import core
else:
    from . import core

BENCH_FILES = ["data/bench_internal/dev.jsonl", "data/bench_internal/shadow.jsonl",
               "data/bench_internal/calib.jsonl", "data/bench_internal/long_context.jsonl"]
TRAIN_FILES = ["data/synthetic/bench_train.jsonl"]

CMP = {
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}
UNIT_HOURS = {"hour": 1, "day": 24, "week": 168, "month": 720}
ADEQUACY_RESOLVES = {"complete": True, "partial": False, "off_topic": False,
                     "contradicting": False, "refusing": False, "other_question": False}


def _bucket(v, edges):
    i = 0
    for e in edges:
        if v < e:
            return i
        i += 1
    return i


def _eval_expr(expr):
    if not set(expr) <= set("0123456789+-*/(). "):
        raise ValueError("unsafe expression %r" % expr)
    return eval(expr, {"__builtins__": {}}, {})


def _iso(d):
    y, m, dd = (int(x) for x in d.split("-"))
    return datetime.date(y, m, dd)


def _closure(relations):
    """Transitive closure of the stated strict relations."""
    nodes = set()
    edges = defaultdict(set)
    for a, op, b in relations:
        assert op == ">", op
        nodes.add(a)
        nodes.add(b)
        edges[a].add(b)
    reach = {}
    for n in nodes:
        seen, stack = set(), [n]
        while stack:
            cur = stack.pop()
            for nxt in edges.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        reach[n] = seen
    return reach


# --------------------------------------------------------------------------- rules

def r_choice_const(a, rec):
    return a["value"]


def r_const_bool(a, rec):
    return bool(a["value"])


def r_fallback(a, rec):
    return a["fallback_key"] if a["offtopic"] else a["intent"]


def r_negated_contrast(a, rec):
    return a["target"]


def r_fact_mode(a, rec):
    return a["mode"] == "stated"


def r_lookup(a, rec):
    return a["record"][a["path"]]


def r_cmp_num(a, rec):
    return bool(CMP[a["op"]](a["lhs"], a["rhs"]))


def r_bucket(a, rec):
    return _bucket(a["value"], a["edges"])


def r_sum_cmp(a, rec):
    return bool(CMP[a["op"]](round(sum(a["addends"]), 6), a["rhs"]))


def r_sum_bucket(a, rec):
    return _bucket(round(sum(a["addends"]), 6), a["edges"])


def r_count_cmp(a, rec):
    return bool(CMP[a["op"]](a["count"], a["rhs"]))


def r_count_bucket(a, rec):
    return _bucket(a["count"], a["edges"])


def _hours(d):
    return d["value"] * UNIT_HOURS[d["unit"]]


def r_duration_cmp(a, rec):
    return bool(CMP[a["op"]](_hours(a["left"]), _hours(a["right"])))


def r_duration_choice(a, rec):
    return a["within_key"] if _hours(a["left"]) <= _hours(a["right"]) else a["over_key"]


def r_date_cmp(a, rec):
    return bool(CMP[a["op"]](_iso(a["left"]), _iso(a["right"])))


def r_date_offset_cmp(a, rec):
    d = _iso(a["base"]) + datetime.timedelta(days=a["offset_days"])
    return bool(CMP[a["op"]](d, _iso(a["ref"])))


def _policy(a):
    if a["override_holds"]:
        return a["allow_key"]
    for c in a["clauses"]:
        if c["blocks"]:
            return c["key"]
    return a["allow_key"]


def r_policy_bool(a, rec):
    return _policy(a) == a["allow_key"]


def r_policy_outcome(a, rec):
    return _policy(a)


def r_policy_severity(a, rec):
    return a["ranks"][_policy(a)]


def r_adequacy_bool(a, rec):
    return ADEQUACY_RESOLVES[a["kind"]]


def r_adequacy_label(a, rec):
    assert a["kind"] in ADEQUACY_RESOLVES, a["kind"]
    return a["kind"]


def r_relevance(a, rec):
    hits = [p for p, k in sorted(a["passages"].items()) if k == a["answer_kind"]]
    assert len(hits) == 1, hits
    return hits[0]


def r_nli(a, rec):
    assert a["label"] in ("entailment", "contradiction", "neutral")
    return a["label"]


def r_nli_bool(a, rec):
    return a["label"] == "entailment"


def r_ordinal_level(a, rec):
    if a.get("collapse"):
        return a["collapse"][a["level"]]
    return a["level"]


def r_ordinal_choice(a, rec):
    return a["keys"][a["level"]]


def r_ordinal_threshold(a, rec):
    return bool(CMP[a["op"]](a["level"], a["rhs"]))


def r_arith(a, rec):
    return abs(_eval_expr(a["expr"]) - a["claimed"]) < 1e-6


def r_arith_fault(a, rec):
    correct = abs(_eval_expr(a["expr"]) - a["claimed"]) < 1e-6
    assert correct == (a["kind"] == "correct"), \
        "fault kind %s disagrees with the arithmetic" % a["kind"]
    return a["kind"]


def r_chain_bool(a, rec):
    reach = _closure(a["relations"])
    x, op, y = a["claim"]
    return y in reach.get(x, set())


def r_chain3(a, rec):
    reach = _closure(a["relations"])
    x, op, y = a["claim"]
    if y in reach.get(x, set()):
        return "follows"
    if x in reach.get(y, set()):
        return "contradicted"
    return "unknown"


RULES = {
    "choice_const": r_choice_const, "const_bool": r_const_bool, "fallback": r_fallback,
    "negated_contrast": r_negated_contrast, "fact_mode": r_fact_mode, "lookup": r_lookup,
    "cmp_num": r_cmp_num, "bucket": r_bucket, "sum_cmp": r_sum_cmp, "sum_bucket": r_sum_bucket,
    "count_cmp": r_count_cmp, "count_bucket": r_count_bucket, "duration_cmp": r_duration_cmp,
    "duration_choice": r_duration_choice, "date_cmp": r_date_cmp,
    "date_offset_cmp": r_date_offset_cmp, "policy_bool": r_policy_bool,
    "policy_outcome": r_policy_outcome, "policy_severity": r_policy_severity,
    "adequacy_bool": r_adequacy_bool, "adequacy_label": r_adequacy_label,
    "relevance": r_relevance, "nli": r_nli, "nli_bool": r_nli_bool,
    "ordinal_level": r_ordinal_level, "ordinal_choice": r_ordinal_choice,
    "ordinal_threshold": r_ordinal_threshold, "arith": r_arith, "arith_fault": r_arith_fault,
    "chain_bool": r_chain_bool, "chain3": r_chain3,
}


# --------------------------------------------------------------------------- checks

def check_record(rec, side, errors):
    def err(msg):
        errors.append("%s: %s" % (rec.get("id", "?"), msg))

    if rec.get("license") != core.LICENSE:
        err("wrong license %r" % rec.get("license"))
    if rec.get("source") != "synthetic/bench/%s" % rec.get("family"):
        err("wrong source %r" % rec.get("source"))
    if rec.get("synthetic") is not True:
        err("not marked synthetic")
    if rec.get("difficulty") not in core.DIFFICULTIES:
        err("bad difficulty %r" % rec.get("difficulty"))
    if rec.get("tier") != rec.get("difficulty"):
        err("tier %r != difficulty %r" % (rec.get("tier"), rec.get("difficulty")))
    if rec.get("reasoning") not in core.REASONINGS:
        err("bad reasoning %r" % rec.get("reasoning"))
    if rec.get("evidence_position") not in core.POSITIONS:
        err("bad evidence_position %r" % rec.get("evidence_position"))
    if not rec.get("group"):
        err("missing group")
    meta = rec.get("meta") or {}
    if meta.get("side") != side:
        err("meta.side %r but found in the %s files" % (meta.get("side"), side))

    state = rec["request"]["state"]
    text = core.state_text(state)
    if rec.get("state_tokens") != len(text.split()):
        err("state_tokens %r != %d" % (rec.get("state_tokens"), len(text.split())))
    ev = meta.get("evidence") or []
    for e in ev:
        if e not in text:
            err("evidence not found in state: %r" % e[:60])
            break
    else:
        pos = core.evidence_position(state, ev)
        if pos != rec["evidence_position"]:
            err("evidence_position %r but recomputes to %r" % (rec["evidence_position"], pos))

    qs = rec["request"]["questions"]
    gold = rec["gold"]
    if sorted(qs) != sorted(gold):
        err("questions %s vs gold %s" % (sorted(qs), sorted(gold)))
    for qid, q in qs.items():
        g = gold.get(qid)
        t = q.get("type")
        if t == "choice":
            if g not in q["criteria"]:
                err("gold %r not among the options of %s" % (g, qid))
            if len(q["criteria"]) < 2:
                err("choice %s has fewer than two options" % qid)
        elif t == "score":
            if not isinstance(g, int) or not (0 <= g < len(q["criteria"])):
                err("score gold %r out of range for %s" % (g, qid))
        elif t == "noul":
            if not isinstance(g, bool):
                err("noul gold %r is not a boolean for %s" % (g, qid))
        else:
            err("unknown question type %r" % t)

    checks = meta.get("checks") or []
    if sorted(c["qid"] for c in checks) != sorted(gold):
        err("checks cover %s but gold has %s" % (sorted(c["qid"] for c in checks), sorted(gold)))
    for c in checks:
        fn = RULES.get(c["rule"])
        if fn is None:
            err("no verifier for rule %r" % c["rule"])
            continue
        try:
            derived = fn(c["args"], rec)
        except Exception as exc:  # noqa: BLE001
            err("rule %s raised %s: %s" % (c["rule"], type(exc).__name__, exc))
            continue
        want = gold[c["qid"]]
        if isinstance(want, bool) or isinstance(derived, bool):
            ok = bool(derived) == bool(want)
        else:
            ok = derived == want
        if not ok:
            err("rule %s re-derives %r but gold is %r" % (c["rule"], derived, want))
        args = c["args"]
        if "injected_target" in args and args["injected_target"] == want:
            err("injected directive agrees with the gold label")
        if "options" in args and want not in args["options"]:
            err("gold %r not in the recorded option set" % (want,))
        if c["rule"] == "lookup" and str(want) not in text:
            err("looked-up value %r does not occur in the state" % (want,))


def populate_pools(scenarios_per_family=650):
    """Touch every template pool on both sides so POOL_SIZES is complete."""
    from bench import (intent_routing, facts_extraction, numeric_temporal, policy_rules,
                       adequacy_relevance, compatibility, ordinal, long_context,
                       judge_like, adversarial)
    mods = [intent_routing, facts_extraction, numeric_temporal, policy_rules,
            adequacy_relevance, compatibility, ordinal, judge_like, adversarial]
    for side in ("bench", "train"):
        for mod in mods:
            ctx = core.Ctx(0, side, "probe")
            for i, _ in enumerate(mod.scenarios(ctx)):
                if i >= scenarios_per_family:
                    break
        ctx = core.Ctx(0, side, "probe_lc")
        for i, _ in enumerate(long_context.scenarios(ctx)):
            if i >= 80:
                break


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--max-errors", type=int, default=25)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    errors = []
    populate_pools()
    ids = set()
    counts = {"split_family": Counter(), "split_tier": Counter(), "family_tier": Counter(),
              "split": Counter(), "bucket_position": Counter(), "reasoning": Counter(),
              "position": Counter()}
    tids = {"bench": set(), "train": set()}
    bench_exact, bench_norm = set(), set()
    n_bench = n_train = 0

    def note(rec):
        counts["split"][rec["split"]] += 1
        counts["split_family"]["%s|%s" % (rec["split"], rec["family"])] += 1
        counts["split_tier"]["%s|%s" % (rec["split"], rec["tier"])] += 1
        counts["family_tier"]["%s|%s" % (rec["family"], rec["tier"])] += 1
        counts["reasoning"][rec["reasoning"]] += 1
        counts["position"][rec["evidence_position"]] += 1
        if "length_bucket" in rec["meta"]:
            counts["bucket_position"]["%s|%d|%s" % (rec["split"], rec["meta"]["length_bucket"],
                                                    rec["evidence_position"])] += 1

    for rel in BENCH_FILES:
        path = os.path.join(root, rel)
        for rec in load(path):
            n_bench += 1
            if rec["id"] in ids:
                errors.append("duplicate id %s" % rec["id"])
            ids.add(rec["id"])
            check_record(rec, "bench", errors)
            note(rec)
            tids["bench"].update(rec["meta"]["template_ids"])
            t = core.state_text(rec["request"]["state"])
            bench_exact.add(hashlib.sha1(t.encode("utf-8")).digest())
            bench_norm.add(hashlib.sha1(core.normalise(t).encode("utf-8")).digest())

    leaks_exact = leaks_norm = 0
    for rel in TRAIN_FILES:
        path = os.path.join(root, rel)
        for rec in load(path):
            n_train += 1
            if rec["id"] in ids:
                errors.append("duplicate id %s" % rec["id"])
            ids.add(rec["id"])
            check_record(rec, "train", errors)
            note(rec)
            tids["train"].update(rec["meta"]["template_ids"])
            t = core.state_text(rec["request"]["state"])
            if hashlib.sha1(t.encode("utf-8")).digest() in bench_exact:
                leaks_exact += 1
                if leaks_exact <= 3:
                    errors.append("LEAK exact: %s" % rec["id"])
            if hashlib.sha1(core.normalise(t).encode("utf-8")).digest() in bench_norm:
                leaks_norm += 1
                if leaks_norm <= 3:
                    errors.append("LEAK normalised: %s" % rec["id"])

    overlap = sorted(tids["bench"] & tids["train"])
    if overlap:
        errors.append("template pools overlap on %d ids, e.g. %s" % (len(overlap), overlap[:5]))
    # every template id must sit on the side of the file it was found in
    side_errors = 0
    unknown_keys = 0
    for side in ("bench", "train"):
        for tid in tids[side]:
            key, _, idx = tid.rpartition("#")
            n = core.POOL_SIZES.get(key)
            if n is None:
                unknown_keys += 1
                continue
            if core.pool_sides(key, n)[int(idx)] != side:
                side_errors += 1
    if side_errors:
        errors.append("%d template ids sit on the wrong side of their pool" % side_errors)

    # ---- report
    print("=" * 72)
    print("records: bench %d, train %d, total %d" % (n_bench, n_train, n_bench + n_train))
    print("-" * 72)
    print("%-14s %-22s %8s" % ("split", "family", "records"))
    for key in sorted(counts["split_family"]):
        split, fam = key.split("|")
        print("%-14s %-22s %8d" % (split, fam, counts["split_family"][key]))
    print("-" * 72)
    print("%-22s %8s %8s %8s %8s" % ("split", "easy", "standard", "hard", "judge"))
    for split in sorted(counts["split"]):
        print("%-22s %8d %8d %8d %8d" % tuple([split] + [counts["split_tier"].get(
            "%s|%s" % (split, t), 0) for t in core.DIFFICULTIES]))
    print("-" * 72)
    print("%-22s %8s %8s %8s %8s" % ("family", "easy", "standard", "hard", "judge"))
    fams = sorted(set(k.split("|")[0] for k in counts["family_tier"]))
    for fam in fams:
        print("%-22s %8d %8d %8d %8d" % tuple([fam] + [counts["family_tier"].get(
            "%s|%s" % (fam, t), 0) for t in core.DIFFICULTIES]))
    print("-" * 72)
    print("long-context length buckets x evidence position")
    buckets = sorted(set(int(k.split("|")[1]) for k in counts["bucket_position"]))
    print("%-10s %-8s %8s %8s %8s %8s" % ("split", "tokens", "start", "middle", "end", "split"))
    for split in sorted(set(k.split("|")[0] for k in counts["bucket_position"])):
        for b in buckets:
            row = [counts["bucket_position"].get("%s|%d|%s" % (split, b, p), 0)
                   for p in ("start", "middle", "end", "split")]
            if sum(row):
                print("%-10s %-8d %8d %8d %8d %8d" % tuple([split, b] + row))
    print("-" * 72)
    print("reasoning:", dict(sorted(counts["reasoning"].items())))
    print("evidence position:", dict(sorted(counts["position"].items())))
    print("template ids: bench %d, train %d, overlap %d, unknown pool keys %d"
          % (len(tids["bench"]), len(tids["train"]), len(overlap), unknown_keys))
    print("state leakage bench -> train: exact %d, normalised %d" % (leaks_exact, leaks_norm))
    print("=" * 72)
    if errors:
        print("FAILED with %d problems:" % len(errors))
        for e in errors[:a.max_errors]:
            print("  -", e)
        return 1
    print("OK: all gold labels re-derived, pools disjoint, no leakage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
