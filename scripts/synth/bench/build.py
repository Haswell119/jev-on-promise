#!/usr/bin/env python3
"""Deterministic driver for the Sextant frozen internal benchmark and the
disjoint synthetic training pool.

    python3 scripts/synth/bench/build.py --seed 20260923

Outputs
    data/bench_internal/dev.jsonl           ~3000  iteration set
    data/bench_internal/shadow.jsonl        ~3000  promotion boundary only
    data/bench_internal/calib.jsonl         ~2000  calibration
    data/bench_internal/long_context.jsonl  ~1500  9 length buckets x 4 positions
    data/synthetic/bench_train.jsonl       ~72000  training pool (train-side pools)
    data/bench_internal/MANIFEST.json

The benchmark streams draw only from the `bench` half of every template and
slot pool and the training pool only from the `train` half, so no template,
slot value or state string can be shared between them.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, OrderedDict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench import core  # noqa: E402
    from bench import (intent_routing, facts_extraction, numeric_temporal, policy_rules,
                       adequacy_relevance, compatibility, ordinal, long_context,
                       judge_like, adversarial)  # noqa: E402
else:
    from . import core
    from . import (intent_routing, facts_extraction, numeric_temporal, policy_rules,
                   adequacy_relevance, compatibility, ordinal, long_context,
                   judge_like, adversarial)

FAMILIES = OrderedDict([
    ("intent_routing", intent_routing),
    ("facts_extraction", facts_extraction),
    ("numeric_temporal", numeric_temporal),
    ("policy_rules", policy_rules),
    ("adequacy_relevance", adequacy_relevance),
    ("compatibility", compatibility),
    ("ordinal", ordinal),
    ("judge_like", judge_like),
    ("adversarial", adversarial),
])

BENCH_DIR = "data/bench_internal"
TRAIN_DIR = "data/synthetic"

DEV_N, SHADOW_N, CALIB_N, LC_N = 3000, 3000, 2000, 1512  # 1512 = 9 buckets x 4 positions x 42
TRAIN_PER_FAMILY = 7600
TRAIN_LC_N = 4000
TRAIN_LC_BUCKETS = [64, 128, 256, 512, 1024]
# The evaluation suite runs to 16k tokens while the pool above stops at 1k,
# so a model trained only on it has to extrapolate 16x on the axis that
# already fails hardest. `--long-train-only` writes a supplementary split
# covering the full evaluation range, from the same disjoint train-side
# template pool, into its own file so existing training exports stay
# reproducible.
TRAIN_LC_EXT_N = 3600
TRAIN_LC_EXT_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]


def req_key(rec):
    return hashlib.sha1(json.dumps(rec["request"], sort_keys=True, ensure_ascii=False)
                        .encode("utf-8")).digest()


def quotas(total, n):
    base = total // n
    out = [base] * n
    for i in range(total - base * n):
        out[i] += 1
    return out


def probe_tiers(make_gen, scenarios=120):
    tiers = set()
    gen = make_gen()
    for i, recs in enumerate(gen):
        for r in recs:
            tiers.add(r["tier"])
        if i >= scenarios:
            break
    return sorted(tiers)


def fill(make_gen, quota, seen, max_scenarios=400000, balanced=True):
    """Yield up to `quota` unique records, balanced across the tiers the family
    actually produces. Falls back to unbalanced acceptance if a tier runs dry."""
    tiers = probe_tiers(make_gen) if balanced else []
    target = dict(zip(tiers, quotas(quota, len(tiers)))) if tiers else {}
    got = Counter()
    taken = 0
    for phase in (0, 1):
        if taken >= quota:
            break
        gen = make_gen()
        for i, recs in enumerate(gen):
            if i > max_scenarios:
                break
            for r in recs:
                if taken >= quota:
                    break
                k = req_key(r)
                if k in seen:
                    continue
                t = r["tier"]
                if phase == 0 and tiers and got[t] >= target.get(t, 0):
                    continue
                seen.add(k)
                got[t] += 1
                taken += 1
                yield r
            if taken >= quota:
                break


def bench_stream(family_mod, stream, seed):
    def make():
        ctx = core.Ctx(seed, "bench", stream)
        return family_mod.scenarios(ctx)
    return make


def train_stream(family_mod, seed, tag):
    def make():
        ctx = core.Ctx(seed, "train", tag)
        return family_mod.scenarios(ctx)
    return make


def lc_stream(seed, side, stream, buckets):
    def make():
        ctx = core.Ctx(seed, side, stream)
        return long_context.scenarios(ctx, buckets=buckets)
    return make


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--root", default=".")
    ap.add_argument("--train-per-family", type=int, default=TRAIN_PER_FAMILY)
    ap.add_argument("--train-lc", type=int, default=TRAIN_LC_N)
    ap.add_argument("--long-train-only", action="store_true",
                    help="write only the extended long-context TRAINING split "
                         "(data/synthetic/bench_train_lc.jsonl); leaves every other file alone")
    ap.add_argument("--train-lc-ext", type=int, default=TRAIN_LC_EXT_N)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    bench_dir = os.path.join(root, BENCH_DIR)
    train_dir = os.path.join(root, TRAIN_DIR)
    os.makedirs(bench_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    counts = {"split": Counter(), "family": Counter(), "split_family": Counter(),
              "split_tier": Counter(), "family_tier": Counter(), "length_bucket": Counter(),
              "position": Counter(), "reasoning": Counter(), "base_family": Counter()}
    tids = {"bench": set(), "train": set()}
    ids = set()
    files = OrderedDict()

    def account(r):
        assert r["id"] not in ids, "duplicate id %s" % r["id"]
        ids.add(r["id"])
        sp = r["split"]
        counts["split"][sp] += 1
        counts["family"][r["family"]] += 1
        counts["split_family"]["%s|%s" % (sp, r["family"])] += 1
        counts["split_tier"]["%s|%s" % (sp, r["tier"])] += 1
        counts["family_tier"]["%s|%s" % (r["family"], r["tier"])] += 1
        counts["position"][r["evidence_position"]] += 1
        counts["reasoning"][r["reasoning"]] += 1
        if "length_bucket" in r["meta"]:
            counts["length_bucket"]["%s|%d" % (sp, r["meta"]["length_bucket"])] += 1
            counts["base_family"][r["meta"].get("base_family", "?")] += 1
        tids[r["meta"]["side"]].update(r["meta"]["template_ids"])

    def write(path, records_iter):
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for r in records_iter:
                account(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        files[os.path.relpath(path, root)] = {"records": n, "bytes": os.path.getsize(path),
                                              "sha256": sha256_of(path)}
        return n

    seen_bench = set()
    fam_names = list(FAMILIES)

    if a.long_train_only:
        # A distinct stream tag keeps the ids and the sampled slots disjoint
        # from the records already in bench_train.jsonl.
        print("building the extended long-context training split ...", flush=True)
        seen_ext = set()
        path = os.path.join(train_dir, "bench_train_lc.jsonl")
        n = write(path, fill(lc_stream(a.seed, "train", "train_lc_ext", TRAIN_LC_EXT_BUCKETS),
                             a.train_lc_ext, seen_ext, balanced=False))
        side_leak = sorted(tids["bench"] & tids["train"])
        assert not side_leak, "train split reused bench templates: %s" % side_leak[:5]
        manifest = OrderedDict()
        manifest["generator"] = "scripts/synth/bench/build.py --long-train-only"
        manifest["seed"] = a.seed
        manifest["note"] = ("Supplementary long-context TRAINING records covering the full "
                            "evaluation length range. Train-side template pool only; no bench "
                            "record, template or slot is reused. Verify with verify.py.")
        manifest["length_buckets"] = TRAIN_LC_EXT_BUCKETS
        manifest["positions"] = long_context.POSITIONS
        manifest["files"] = files
        manifest["counts"] = {k: OrderedDict(sorted(v.items())) for k, v in counts.items() if v}
        with open(os.path.join(train_dir, "MANIFEST_train_lc.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print("wrote %d records -> %s" % (n, path))
        return

    def bench_records(stream, total):
        qs = quotas(total, len(fam_names))
        for name, q in zip(fam_names, qs):
            for r in fill(bench_stream(FAMILIES[name], stream, a.seed), q, seen_bench):
                yield r

    print("building frozen internal benchmark ...", flush=True)
    write(os.path.join(bench_dir, "dev.jsonl"), bench_records("dev", DEV_N))
    write(os.path.join(bench_dir, "shadow.jsonl"), bench_records("shadow", SHADOW_N))
    write(os.path.join(bench_dir, "calib.jsonl"), bench_records("calib", CALIB_N))
    write(os.path.join(bench_dir, "long_context.jsonl"),
          fill(lc_stream(a.seed, "bench", "long_context", long_context.BUCKETS),
               LC_N, seen_bench, balanced=False))

    print("building training pool ...", flush=True)
    seen_train = set()

    def train_records():
        for name in fam_names:
            for r in fill(train_stream(FAMILIES[name], a.seed, "train"),
                          a.train_per_family, seen_train):
                yield r
        for r in fill(lc_stream(a.seed, "train", "train", TRAIN_LC_BUCKETS),
                      a.train_lc, seen_train, balanced=False):
            yield r

    write(os.path.join(train_dir, "bench_train.jsonl"), train_records())

    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None

    manifest = OrderedDict()
    manifest["generator"] = "scripts/synth/bench/build.py"
    manifest["seed"] = a.seed
    manifest["git_commit"] = commit
    manifest["model"] = core.MODEL
    manifest["license"] = core.LICENSE
    manifest["note"] = ("Frozen internal benchmark and training pool generated from disjoint "
                        "template and slot pools. `tier` equals `difficulty`. Labels are "
                        "verifiable by construction from meta.checks; see verify.py. No "
                        "external benchmark was read or used at any point.")
    manifest["families"] = fam_names + ["long_context"]
    manifest["length_buckets"] = long_context.BUCKETS
    manifest["positions"] = long_context.POSITIONS
    manifest["files"] = files
    manifest["counts"] = {k: OrderedDict(sorted(v.items())) for k, v in counts.items()}
    manifest["template_pools"] = {
        "bench_template_ids": len(tids["bench"]),
        "train_template_ids": len(tids["train"]),
        "overlap": len(tids["bench"] & tids["train"]),
        "pool_keys": len(core.POOL_SIZES),
    }
    mpath = os.path.join(bench_dir, "MANIFEST.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    print(json.dumps({"files": {k: v["records"] for k, v in files.items()},
                      "template_overlap": manifest["template_pools"]["overlap"]}, indent=2))


if __name__ == "__main__":
    main()
