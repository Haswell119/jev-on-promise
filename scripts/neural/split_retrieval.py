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


def bucket(record_id, parts):
    h = hashlib.blake2b(record_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--dev-out", required=True)
    ap.add_argument("--dev-fraction", type=float, default=0.1)
    ap.add_argument("--by", choices=("record", "template"), default="template",
                    help="split on the record id, or on the generator template (stricter)")
    a = ap.parse_args()
    parts = 1000
    cut = int(a.dev_fraction * parts)
    Path(a.train_out).parent.mkdir(parents=True, exist_ok=True)
    n_tr = n_dv = 0
    no_templates = 0
    with open(a.input) as fh, open(a.train_out, "w") as tr, open(a.dev_out, "w") as dv:
        for line in fh:
            r = json.loads(line)
            if a.by == "template":
                tids = r.get("template_ids") or []
                if not tids:
                    no_templates += 1
                    key = r["record_id"]
                else:
                    # A record can use several templates. Send it to the
                    # selection side only if EVERY template belongs there,
                    # so no template is seen on both sides.
                    to_dev = all(bucket(t, parts) < cut for t in tids)
                    any_dev = any(bucket(t, parts) < cut for t in tids)
                    if any_dev and not to_dev:
                        continue  # straddles the boundary; drop it
                    key = None
                    if to_dev:
                        dv.write(line)
                        n_dv += 1
                    else:
                        tr.write(line)
                        n_tr += 1
                    continue
            else:
                key = r["record_id"]
            if bucket(key, parts) < cut:
                dv.write(line)
                n_dv += 1
            else:
                tr.write(line)
                n_tr += 1
    out = {"train_lists": n_tr, "dev_lists": n_dv, "dev_fraction": a.dev_fraction, "split_by": a.by}
    if no_templates:
        out["rows_without_template_ids"] = no_templates
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
