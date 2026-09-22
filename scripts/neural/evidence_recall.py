#!/usr/bin/env python3
"""Retrieval diagnostic: does the evidence block handed to the encoder still
contain the decisive sentences? Compares `meta.evidence` from the benchmark
records with the `evidence` field of the exported pairs, bucketed by state
length and evidence position. A decision model cannot be right if retrieval
already lost the needle, so this separates retrieval failures from reasoning
failures."""
import argparse, json, re
from collections import defaultdict


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def toks(s):
    return [t for t in norm(s).split() if t]


def recall(gold, got):
    g = toks(gold)
    if not g:
        return 1.0
    hay = " " + " ".join(toks(got)) + " "
    if " " + " ".join(g) + " " in hay:
        return 1.0
    # longest contiguous run of gold tokens present, normalised
    best = cur = 0
    for i in range(len(g)):
        for j in range(i + cur + 1, len(g) + 1):
            if " " + " ".join(g[i:j]) + " " in hay:
                cur = j - i
                best = max(best, cur)
            else:
                break
    return best / len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    meta = {}
    for line in open(a.records):
        r = json.loads(line)
        ev = (r.get("meta") or {}).get("evidence") or []
        meta[r["id"]] = {"evidence": ev, "bucket": (r.get("meta") or {}).get("length_bucket"), "pos": r.get("evidence_position"), "tokens": r.get("state_tokens", 0), "family": (r.get("meta") or {}).get("base_family", r.get("family"))}
    by_bucket, by_pos, by_fam = defaultdict(list), defaultdict(list), defaultdict(list)
    full, partial, missed, n = 0, 0, 0, 0
    for line in open(a.pairs):
        p = json.loads(line)
        rid = p["record_id"]
        m = meta.get(rid)
        if not m or not m["evidence"]:
            continue
        rs = [recall(e, p["evidence"]) for e in m["evidence"]]
        r = sum(rs) / len(rs)
        n += 1
        full += r >= 0.999
        partial += 0.5 <= r < 0.999
        missed += r < 0.5
        by_bucket[m["bucket"]].append(r)
        by_pos[m["pos"]].append(r)
        by_fam[m["family"]].append(r)
    def table(d, label):
        print(f"\n{label:<16} {'n':>6} {'mean recall':>12} {'full':>7}")
        for k in sorted(d, key=lambda x: (isinstance(x, str), x)):
            v = d[k]
            print(f"{str(k):<16} {len(v):>6} {sum(v)/len(v):>12.3f} {sum(1 for x in v if x >= 0.999)/len(v):>7.3f}")
    print(f"records with gold evidence: {n}")
    print(f"fully retrieved {full/n:.3f} | partial {partial/n:.3f} | lost {missed/n:.3f}")
    table(by_bucket, "length bucket")
    table(by_pos, "position")
    table(by_fam, "base family")
    if a.out:
        json.dump({"n": n, "full": full / n, "partial": partial / n, "lost": missed / n,
                   "by_bucket": {str(k): sum(v)/len(v) for k, v in by_bucket.items()},
                   "by_position": {str(k): sum(v)/len(v) for k, v in by_pos.items()}}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
