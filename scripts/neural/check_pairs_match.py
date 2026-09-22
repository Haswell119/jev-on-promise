#!/usr/bin/env python3
"""Refuse to evaluate a scorer against pairs retrieved differently.

The encoder only sees the evidence block the retriever selected, so an
evaluation run whose pairs were exported with different retrieval
parameters measures a model that never existed. Rather than report that
number, fail loudly and say how to re-export.

    check_pairs_match.py <scorer.json> <pairs.jsonl> [<pairs.jsonl> ...]
"""
import json
import sys
from pathlib import Path

KEYS = ("budget_words", "adaptive_cap", "strategy", "local_idf", "q_expand")


def wanted(scorer_path):
    s = json.loads(Path(scorer_path).read_text())
    return {
        "budget_words": s["evidence_words"],
        "adaptive_cap": s.get("evidence_adaptive_cap", 0),
        "strategy": s.get("evidence_strategy", "quota"),
        "local_idf": s.get("evidence_local_idf", False),
        "q_expand": s.get("evidence_q_expand", False),
    }


def flags(want):
    out = ["--budget-words", str(want["budget_words"]), "--strategy", want["strategy"]]
    if want["adaptive_cap"]:
        out += ["--adaptive-cap", str(want["adaptive_cap"])]
    if want["local_idf"]:
        out.append("--local-idf")
    if want["q_expand"]:
        out.append("--q-expand")
    return " ".join(out)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    want = wanted(sys.argv[1])
    bad, unverified = [], []
    for pairs in sys.argv[2:]:
        meta = Path(pairs).with_suffix(".meta.json")
        if not meta.exists():
            unverified.append(pairs)
            continue
        m = json.loads(meta.read_text())
        got = {k: m.get(k) for k in KEYS}
        if got != want:
            diff = {k: (got[k], want[k]) for k in KEYS if got[k] != want[k]}
            bad.append(f"  {pairs}: " + ", ".join(f"{k} is {g!r}, model needs {w!r}" for k, (g, w) in diff.items()))
    for p in unverified:
        print(f"warning: {p} has no retrieval sidecar, so it cannot be verified against the model", file=sys.stderr)
    if bad:
        raise SystemExit(
            "these evaluation pairs were retrieved differently from the model's training data:\n"
            + "\n".join(bad)
            + f"\n\nRe-export them with:\n  sextant export-pairs <records> --out <pairs> {flags(want)}\n"
        )
    print(f"retrieval matches on {len(sys.argv) - 2 - len(unverified)} pairs file(s): {want}")


if __name__ == "__main__":
    main()
