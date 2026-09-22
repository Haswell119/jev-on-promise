#!/usr/bin/env python3
"""Convert JevBench public task files into Sextant eval records (for
`sextant eval` group-level analysis only; never for training).

Adds analysis fields: tier, family, kind, state_type, n_options bucket,
has_noul_criteria, state_tokens bucket.
"""
import json, sys, os
def bucket_k(k):
    return "2" if k <= 2 else "3-4" if k <= 4 else "5-8" if k <= 8 else "9-16" if k <= 16 else "17+"
def bucket_tokens(n):
    return "<64" if n < 64 else "64-256" if n < 256 else "256-1k" if n < 1024 else "1k+"
out = []
for path in sys.argv[1:-1]:
    tier = os.path.basename(path).replace(".jsonl", "")
    tier = {"original": "standard"}.get(tier, tier)
    for line in open(path):
        if not line.strip():
            continue
        t = json.loads(line)
        q = t["question"]
        kind = q["type"]
        exp = t["expected"]
        if kind == "noul":
            gold = (exp == "yes")
        elif kind == "score":
            gold = int(exp)
        else:
            gold = exp
        n_opt = len(q.get("criteria", [])) if isinstance(q.get("criteria"), (dict, list)) else 2
        state_text = t["state"] if isinstance(t["state"], str) else json.dumps(t["state"])
        rec = {
            "id": t["id"], "source": "jevbench", "license": "MIT", "split": "eval", "tier": tier,
            "family": t.get("family", ""), "synthetic": False, "transformation": kind, "group": t.get("group"),
            "kind": kind, "state_type": "str" if isinstance(t["state"], str) else "dict",
            "n_options": bucket_k(n_opt), "has_noul_criteria": str(kind == "noul" and bool(q.get("criteria"))),
            "state_tokens": bucket_tokens(len(state_text.split())),
            "request": {"model": "sextant-1", "state": t["state"], "questions": {"decision": q}},
            "gold": {"decision": gold},
        }
        out.append(rec)
with open(sys.argv[-1], "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(len(out), "records ->", sys.argv[-1])
