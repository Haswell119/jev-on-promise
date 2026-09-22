#!/usr/bin/env python3
"""Split exported retrieval lists into a fitting set and a selection set.

Two modes, both stable under re-export because the bucket is a hash.

`--by record` splits on the record id. One record's questions share a
state, so a row-level split would put near-identical lists on both sides
and make the selection set optimistic.

`--by template` splits on the generator template behind the record, and
is the stricter test. A record-level split cannot detect a feature that
is an artefact of particular templates: in experiment R3 a weight of
+4.03 on a binary negation flag passed record-level selection and then
lost 7 points of recall on a suite drawn from a disjoint template pool.
A template-level split reproduces that boundary during selection, so the
failure shows up before the model is adopted.
"""
import argparse
import hashlib
import json
from pathlib import Path


def bucket(key, parts):
    h = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % parts


# Slot templates are the value fillers (names, cities, products) and appear
# in almost every record, so they cannot partition anything. The structural
# templates are the ones that decide what a record looks like.
SLOT_PREFIX = "slot/"


def template_key(row):
    """The record's structural signature: its non-slot template families,
    deduplicated, instance numbers stripped, sorted and joined.

    Requiring every individual template to fall on one side would drop
    nearly every record, since each carries five to twelve of them. Keying
    on the whole combination holds out structures rather than templates:
    644 distinct signatures over the training pool, enough to partition
    stably while keeping records that are built the same way together.

    This is weaker than the bench/train pool boundary, since two
    signatures can share a family, but far stronger than a record-level
    split, which puts the same structure on both sides by construction.
    """
    fams = sorted(
        {
            t.split("#", 1)[0]
            for t in (row.get("template_ids") or [])
            if not t.startswith(SLOT_PREFIX)
        }
    )
    return "|".join(fams) if fams else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--dev-out", required=True)
    ap.add_argument("--dev-fraction", type=float, default=0.15)
    ap.add_argument("--by", choices=("record", "template"), default="template",
                    help="split on the record id, or on the primary structural template (stricter)")
    a = ap.parse_args()
    parts = 1000
    cut = int(a.dev_fraction * parts)
    Path(a.train_out).parent.mkdir(parents=True, exist_ok=True)
    n_tr = n_dv = fallback = 0
    dev_keys, train_keys = set(), set()
    with open(a.input) as fh, open(a.train_out, "w") as tr, open(a.dev_out, "w") as dv:
        for line in fh:
            row = json.loads(line)
            key = template_key(row) if a.by == "template" else None
            if key is None:
                if a.by == "template":
                    fallback += 1
                key = row["record_id"]
            if bucket(key, parts) < cut:
                dv.write(line)
                n_dv += 1
                dev_keys.add(key)
            else:
                tr.write(line)
                n_tr += 1
                train_keys.add(key)
    out = {
        "train_lists": n_tr,
        "dev_lists": n_dv,
        "dev_fraction": a.dev_fraction,
        "split_by": a.by,
        "train_keys": len(train_keys),
        "dev_keys": len(dev_keys),
        "keys_on_both_sides": len(dev_keys & train_keys),
    }
    if fallback:
        out["rows_keyed_by_record_instead"] = fallback
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
