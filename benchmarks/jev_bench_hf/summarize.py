#!/usr/bin/env python3
"""Add macro and per-primitive rows to a Jevify `jevify-run report` of a Sextant run.

`jevify-run report --json` already writes one block per config (accuracy, ECE, Brier,
NLL, selective accuracy, AURC, RPS/MAE for score, AUROC for noul, TVD/KL against human
label distributions, reliability bins). This script reads that file plus preds.jsonl
and writes the leaderboard-style aggregates the harness's README uses: macro accuracy /
ECE / Brier over configs, per-primitive macro accuracy, mean TVD to human distributions
over the calibration-gold configs, error counts and latency percentiles. No metric is
recomputed from raw probabilities; everything comes from the harness's own numbers.

  summarize.py --metrics RUN/metrics.json --preds RUN/preds.jsonl \
      [--dataset-manifest data/manifest.json] [--manifest RUN/manifest.json] \
      [--environment RUN/environment.json] [--label "..."] --md RUN/summary.md --json RUN/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def percentile(values: list, q: float):
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q
    f, c = int(k), int(k) + (0 if k == int(k) else 1)
    if f == c:
        return vals[f]
    return vals[f] * (c - k) + vals[c] * (k - f)


def mean(values: list):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def fmt(x, digits=3, pct=False):
    if x is None:
        return "–"
    if isinstance(x, int):
        return str(x)
    return f"{100 * x:.1f} %" if pct else f"{x:.{digits}f}"


def fmt_ms(x):
    return "–" if x is None else f"{x:.1f} ms"


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", required=True, help="jevify-run report --json output")
    ap.add_argument("--preds", required=True, help="jevify-run api --out predictions")
    ap.add_argument("--dataset-manifest", default=None, help="the dataset's manifest.json (primitive, K, counts)")
    ap.add_argument("--manifest", default=None, help="run.sh manifest.json")
    ap.add_argument("--environment", default=None, help="run.sh environment.json")
    ap.add_argument("--label", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    metrics = load_json(args.metrics) or {}
    sources_meta = ((load_json(args.dataset_manifest) or {}).get("sources")) or {}

    preds = []
    with open(args.preds, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                preds.append(json.loads(line))
    per_source_preds = defaultdict(list)
    for p in preds:
        per_source_preds[p["id"].split("/")[0]].append(p)

    configs = {}
    for src in sorted(set(metrics) | set(per_source_preds)):
        rep = metrics.get(src)
        ps = per_source_preds.get(src, [])
        prim = next((p["primitive"] for p in ps), None) or (sources_meta.get(src) or {}).get("primitive")
        lat = [p["latency_ms"] for p in ps if p.get("latency_ms")]
        meta = sources_meta.get(src) or {}
        configs[src] = {
            "primitive": prim,
            "k": meta.get("k"),
            "has_soft_labels": meta.get("has_soft_labels"),
            "n_test_total": (meta.get("counts") or {}).get("test"),
            "n_predicted": len(ps),
            "n_errors": sum(1 for p in ps if p.get("error")),
            "n_scored": rep["n"] if rep else 0,
            "latency_ms": {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95), "n": len(lat)},
            "report": rep,
        }

    def macro(field, subset=None):
        return mean([c["report"][field] for s, c in configs.items() if c["report"] and (subset is None or s in subset)])

    prims = {"choice", "score", "noul"}
    by_prim = {p: sorted(s for s, c in configs.items() if c["primitive"] == p) for p in prims}
    soft = sorted(s for s, c in configs.items() if c["report"] and c["report"].get("tvd_to_human") is not None)
    all_lat = [p["latency_ms"] for p in preds if p.get("latency_ms")]
    aggregates = {
        "n_configs": sum(1 for c in configs.values() if c["report"]),
        "n_predicted": len(preds),
        "n_errors": sum(1 for p in preds if p.get("error")),
        "n_scored": sum(c["n_scored"] for c in configs.values()),
        "macro_accuracy": macro("accuracy"),
        "macro_ece": macro("ece"),
        "macro_brier": macro("brier"),
        "macro_nll": macro("nll"),
        "macro_selective_acc_at_90": macro("selective_acc_at_90"),
        "macro_aurc": macro("aurc"),
        "per_primitive": {
            p: {
                "configs": by_prim[p],
                "accuracy": macro("accuracy", set(by_prim[p])),
                "ece": macro("ece", set(by_prim[p])),
                "brier": macro("brier", set(by_prim[p])),
            }
            for p in ("choice", "score", "noul")
        },
        "tvd_to_human_mean": macro("tvd_to_human", set(soft)),
        "tvd_to_human_configs": soft,
        "latency_ms": {"p50": percentile(all_lat, 0.5), "p95": percentile(all_lat, 0.95), "n": len(all_lat)},
    }

    run, env = load_json(args.manifest) or {}, load_json(args.environment) or {}
    doc = {
        "benchmark": "jev-bench (Praveenrajus/jev-bench) test splits",
        "label": args.label,
        "run": run,
        "environment": env,
        "aggregates": aggregates,
        "configs": configs,
        "definitions": [
            "Every per-config number is the harness's own (jevify.bench.metrics at the pinned commit): accuracy = argmax (choice/score) or P(yes) >= 0.5 (noul); ECE = top-label, 15 equal-width bins; Brier = sum over options of (p - y)^2 for choice/score and (p_yes - y)^2 for noul; NLL = -log p(gold); sel@90 = accuracy on the 90 % most confident records (model `confidence` for choice/score, |2p-1| for noul); AURC = area under the risk-coverage curve; RPS/MAE = ordinal scores for score configs; AUROC for noul; TVD→human = total variation distance to the human label distribution on the four calibration-gold configs.",
            "macro rows are unweighted means over the configs that were scored; per-primitive rows average the configs of that primitive. With --limit, `n scored` is the number of records actually sent, not the full test split (`n test`).",
            "records with a runner error (transport failure or non-200 status) are excluded from scoring by the harness and counted under `errors`.",
            "latency = the runner's per-request wall time (httpx, concurrency as configured) in milliseconds.",
        ],
    }

    lines = [f"# jev-bench test splits — Sextant `{run.get('model', 'sextant-1')}`", ""]
    if args.label:
        lines += [f"> **{args.label}**", ""]
    limit = run.get("limit_per_source")
    if limit:
        lines += [f"> Only the first {limit} records of each config were sent (`--limit {limit}`); the full test set is 22,773 records.", ""]
    facts = [
        ("Run", f"`{run.get('run_id', '–')}` finished {run.get('finished_utc', '–')}"),
        ("Sextant", f"commit `{env.get('sextant_commit', '–')}`" + (" (dirty worktree)" if env.get("sextant_worktree_dirty") else "") + f", `{env.get('sextant_version', '–')}`"),
        ("Dataset", f"Praveenrajus/jev-bench @ `{run.get('dataset_revision', '–')}` (test splits only)"),
        ("Harness", f"uspraveen/Jevify @ `{run.get('harness_commit', '–')}` (Apache-2.0), `jevify-run api` + `jevify-run report`"),
        ("Endpoint", f"`{run.get('endpoint', '–')}`, concurrency {run.get('concurrency', '–')}"),
        ("Hardware", f"{env.get('cpu_model', '–')}, {env.get('cpu_count', '–')} cores, {env.get('ram_gib', '–')} GiB RAM, {env.get('rustc', '–')}"),
    ]
    lines += [table(["", ""], [[k, v] for k, v in facts]), ""]

    a = aggregates
    lines += ["## Leaderboard-style aggregates", ""]
    lines += [table(
        ["configs", "records scored", "errors", "macro acc", "macro ECE", "macro Brier", "sel@90", "choice acc", "score acc", "noul acc", "TVD→human", "latency p50", "p95"],
        [[a["n_configs"], a["n_scored"], a["n_errors"], fmt(a["macro_accuracy"]), fmt(a["macro_ece"]), fmt(a["macro_brier"]),
          fmt(a["macro_selective_acc_at_90"]), fmt(a["per_primitive"]["choice"]["accuracy"]), fmt(a["per_primitive"]["score"]["accuracy"]),
          fmt(a["per_primitive"]["noul"]["accuracy"]), fmt(a["tvd_to_human_mean"]), fmt_ms(a["latency_ms"]["p50"]), fmt_ms(a["latency_ms"]["p95"])]],
    ), ""]

    lines += ["## Per config", ""]
    rows = []
    for src, c in configs.items():
        r = c["report"] or {}
        rows.append([
            f"`{src}`", c["primitive"] or "–", c["k"] if c["k"] is not None else "–",
            f"{c['n_scored']} / {c['n_test_total'] if c['n_test_total'] is not None else '–'}", c["n_errors"],
            fmt(r.get("accuracy")), fmt(r.get("ece")), fmt(r.get("brier")), fmt(r.get("nll"), 2), fmt(r.get("selective_acc_at_90")),
            fmt(r.get("mae"), 2), fmt(r.get("auroc")), fmt(r.get("tvd_to_human")), fmt_ms(c["latency_ms"]["p50"]),
        ])
    lines += [table(["config", "prim", "K", "scored / test", "errors", "acc", "ECE", "Brier", "NLL", "sel@90", "MAE", "AUROC", "TVD→human", "p50"], rows), ""]

    lines += ["## Definitions", ""] + [f"- {d}" for d in doc["definitions"]] + [""]
    md = "\n".join(lines)

    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(md)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    if not args.md:
        sys.stdout.write(md)
    print(f"[summarize] {a['n_configs']} configs, {a['n_scored']} scored, {a['n_errors']} errors: "
          f"macro acc {fmt(a['macro_accuracy'])}  macro ECE {fmt(a['macro_ece'])}  macro Brier {fmt(a['macro_brier'])}  "
          f"choice {fmt(a['per_primitive']['choice']['accuracy'])}  score {fmt(a['per_primitive']['score']['accuracy'])}  "
          f"noul {fmt(a['per_primitive']['noul']['accuracy'])}  p50 {fmt_ms(a['latency_ms']['p50'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
