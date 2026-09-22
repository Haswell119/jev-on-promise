#!/usr/bin/env python3
"""Split exported retrieval lists into a fitting set and a selection set.

The split is by RECORD id, not by row: one record's questions share a
state, so splitting by row would put near-identical lists on both sides
and make the selection set optimistic. The hash is stable, so re-running
the export reproduces the same split.
"""
import argparse
import hashlib
import json
from pathlib import Path


def bucket(record_id, parts):
    h = hashlib.blake2b(record_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--dev-out", required=True)
    ap.add_argument("--dev-fraction", type=float, default=0.1)
    a = ap.parse_args()
    parts = 1000
    cut = int(a.dev_fraction * parts)
    Path(a.train_out).parent.mkdir(parents=True, exist_ok=True)
    n_tr = n_dv = 0
    with open(a.input) as fh, open(a.train_out, "w") as tr, open(a.dev_out, "w") as dv:
        for line in fh:
            r = json.loads(line)
            if bucket(r["record_id"], parts) < cut:
                dv.write(line)
                n_dv += 1
            else:
                tr.write(line)
                n_tr += 1
    print(json.dumps({"train_lists": n_tr, "dev_lists": n_dv, "dev_fraction": a.dev_fraction}, indent=2))


if __name__ == "__main__":
    main()
