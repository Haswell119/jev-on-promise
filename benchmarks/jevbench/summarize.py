#!/usr/bin/env python3
"""Summarize a JevBench run of Sextant from the harness's per-item results.

Reports, per public tier and pooled: accuracy over the tier's public denominator
(easy 48, standard = original 72, hard 111; the judge tier is not public), schema
validity as the harness defines it (headline: sum within 2 %, renormalized; strict:
sum within 0.001), Brier (multi-class sum over the exact label set), ECE (top-label,
10 equal-width bins), chance-corrected "intelligence", per-family accuracy (the hard
tier in the markdown, every tier in the JSON), per-question-type accuracy,
paraphrase-pair consistency over `group`, and latency p50/p95 from the per-item
caller wall time.

Every metric re-implements the definition in the harness at the pinned commit
(jevbench/scoring.py, jevbench/metrics.py, jevbench/summarize.py), so it can be
checked against the harness's own summary.<tier>.json. The harness's scoring is
never modified: this script only reads results.jsonl and the public task files.

  summarize.py --tasks-dir harness/datasets/public --results RUN/results.jsonl \
      --md RUN/report.md --json RUN/metrics.json [--tiers easy,original,hard] \
      [--label "smoke run, untrained/bootstrap model"] \
      [--manifest RUN/manifest.json] [--environment RUN/environment.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

# tier key used on the command line -> (display name, public file, frozen item count)
TIERS = {
    "easy": ("easy", "easy.jsonl", 72),
    "original": ("standard", "original.jsonl", 96),
    "hard": ("hard", "hard.jsonl", 220),
}
JUDGE_TIER_N = 146  # imported decisions (router 78 + judge 68); never published

# Option-count histograms of every frozen item per tier, from the harness's
# jevbench/composite_v13.py at the pinned commit. The v1.3 chance baseline is the
# mean of 1/options over ALL frozen items (public + held out); the public halves
# alone give a slightly different number, so both are reported.
TIER_OPTION_COUNTS = {
    "easy": {2: 18, 4: 13, 5: 41},
    "standard": {2: 32, 4: 40, 5: 12, 6: 12},
    "judge": {2: 68, 9: 78},
    "hard": {2: 77, 3: 26, 4: 73, 5: 38, 6: 6},
}


def chance_from_option_counts(counts: dict) -> float:
    n = sum(counts.values())
    return sum(count / options for options, count in counts.items()) / n


# ----------------------------------------------------------------------------- harness replicas


def argmax_label(probs: dict) -> str:
    """jevbench.scoring.argmax_label: ties go to the lexicographically smallest label."""
    best, best_p = None, -1.0
    for k in sorted(probs.keys()):
        if probs[k] > best_p:
            best, best_p = k, probs[k]
    return best


def ece_top_label(pairs: list, n_bins: int = 10) -> dict:
    """jevbench.metrics.ece_top_label: (confidence, correct) pairs, equal-width bins."""
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "conf_sum": 0.0, "correct": 0} for i in range(n_bins)]
    for conf, correct in pairs:
        conf = min(max(float(conf), 0.0), 1.0)
        b = bins[min(int(conf * n_bins), n_bins - 1)]
        b["n"] += 1
        b["conf_sum"] += conf
        b["correct"] += 1 if correct else 0
    n_total = sum(b["n"] for b in bins)
    ece = 0.0
    for b in bins:
        if b["n"]:
            ece += (b["n"] / n_total) * abs(b["correct"] / b["n"] - b["conf_sum"] / b["n"])
    return {
        "ece": ece,
        "n": n_total,
        "bins": [
            {
                "lo": b["lo"],
                "hi": b["hi"],
                "n": b["n"],
                "mean_confidence": (b["conf_sum"] / b["n"]) if b["n"] else None,
                "accuracy": (b["correct"] / b["n"]) if b["n"] else None,
            }
            for b in bins
        ],
    }


def percentile(values: list, q: float):
    """jevbench.metrics.percentile: linear interpolation between order statistics."""
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def brier(probs: dict, labels: list, gold: str) -> float:
    """jevbench.summarize.metric: sum_k (p_k - y_k)^2 over the exact label set."""
    return sum((probs[k] - int(k == gold)) ** 2 for k in labels)


# ----------------------------------------------------------------------------- loading


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- metrics


def scorable(task: dict) -> bool:
    return task.get("expected") is not None and not (task.get("provenance") or {}).get("exclude_reason")


def block(tasks: list, by_id: dict) -> dict:
    """The harness's `metric()` block for a task subset, plus chance and a few extras."""
    rs = [by_id[t["id"]] for t in tasks if t["id"] in by_id]
    scor = [t for t in tasks if scorable(t)]
    correct = sum(1 for t in scor if by_id.get(t["id"], {}).get("correct"))
    valid = sum(1 for r in rs if r.get("valid"))
    strict = sum(1 for r in rs if r.get("strict_valid"))
    briers, pairs, mae = [], [], []
    for t in scor:
        r = by_id.get(t["id"])
        if not r or not r.get("probs"):
            continue
        p, gold = r["probs"], str(t["expected"])
        briers.append(brier(p, t["labels"], gold))
        pairs.append((max(p.values()), argmax_label(p) == gold))
        if t["question"]["type"] == "score":
            mae.append(abs(sum(float(k) * v for k, v in p.items()) - t["expected"]))
    lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
    chance = (sum(1.0 / len(t["labels"]) for t in tasks) / len(tasks)) if tasks else None
    acc = (correct / len(scor)) if scor else None
    return {
        "n_planned": len(tasks),
        "n_attempted": len(rs),
        "n_scorable": len(scor),
        "n_valid": valid,
        "n_strict_valid": strict,
        "n_renormalized": sum(1 for r in rs if r.get("renormalized")),
        "n_failed": sum(1 for r in rs if not r.get("ok")),
        "n_correct": correct,
        "accuracy": acc,
        "coverage": (len(rs) / len(tasks)) if tasks else None,
        "schema_validity": (valid / len(rs)) if rs else None,
        "schema_validity_strict": (strict / len(rs)) if rs else None,
        "operational_success": (sum(1 for r in rs if r.get("ok")) / len(rs)) if rs else None,
        "calibration_n": len(briers),
        "brier_mean": (sum(briers) / len(briers)) if briers else None,
        "ece": ece_top_label(pairs) if pairs else None,
        "ordinal_mae": (sum(mae) / len(mae)) if mae else None,
        "chance_public_items": chance,
        "latency": {
            "n": len(lat),
            "p50_s": percentile(lat, 0.5),
            "p95_s": percentile(lat, 0.95),
            "mean_s": (sum(lat) / len(lat)) if lat else None,
            "max_s": max(lat) if lat else None,
            "first_s": lat[0] if lat else None,
        },
    }


def with_intelligence(b: dict, tier_name: str | None) -> dict:
    """JevBench v1.3: 100 * (accuracy - chance) / (1 - chance), clipped to [0, 100]."""
    frozen = chance_from_option_counts(TIER_OPTION_COUNTS[tier_name]) if tier_name in TIER_OPTION_COUNTS else None
    b["chance_frozen_tier"] = frozen
    chance = frozen if frozen is not None else b["chance_public_items"]
    b["chance_basis"] = "all frozen items (harness histogram)" if frozen is not None else "public items"
    acc = b["accuracy"]
    b["intelligence"] = (max(0.0, min(100.0, 100 * (acc - chance) / (1 - chance)))
                         if acc is not None and chance is not None else None)
    return b


def paraphrase(tasks: list, by_id: dict) -> dict:
    """jevbench.summarize.agreement over `group` (pairs = groups of exactly two)."""
    groups = defaultdict(list)
    for t in tasks:
        if t.get("group"):
            groups[t["group"]].append(t)
    pairs = [g for g in groups.values() if len(g) == 2]
    both_valid = []
    for a, b in pairs:
        x, y = by_id.get(a["id"]), by_id.get(b["id"])
        if x and y and x.get("valid") and y.get("valid"):
            both_valid.append((x, y))
    agree = sum(1 for x, y in both_valid if x.get("predicted") == y.get("predicted"))
    both_correct = sum(1 for x, y in both_valid if x.get("correct") and y.get("correct"))
    return {
        "pairs": len(pairs),
        "both_valid": len(both_valid),
        "agree": agree,
        "agreement": (agree / len(both_valid)) if both_valid else None,
        "both_correct": both_correct,
        "both_correct_rate_all_pairs": (both_correct / len(pairs)) if pairs else None,
        "both_correct_rate_valid_pairs": (both_correct / len(both_valid)) if both_valid else None,
    }


def by_key(tasks: list, by_id: dict, key) -> dict:
    out = {}
    for name in sorted({key(t) for t in tasks}):
        out[name] = block([t for t in tasks if key(t) == name], by_id)
    return out


# ----------------------------------------------------------------------------- rendering


def fmt(x, digits=3, pct=False):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if pct:
        return f"{100 * x:.1f} %"
    return f"{x:.{digits}f}"


def fmt_s(x):
    return "–" if x is None else f"{1000 * x:.1f} ms"


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_md(doc: dict) -> str:
    run, env = doc.get("run") or {}, doc.get("environment") or {}
    lines = [f"# JevBench public tiers — Sextant `{run.get('model', 'sextant-1')}`", ""]
    if doc.get("label"):
        lines += [f"> **{doc['label']}**", ""]
    lines += [
        "> Public halves of the easy, standard and hard tiers, scored by the harness's own",
        "> `typesafe` adapter and scoring at the pinned commit. Unattempted and schema-invalid",
        "> answers count as wrong. Held-out items and the judge tier are not public, so these",
        "> numbers are not comparable one-to-one with the published leaderboard.",
        "",
    ]
    facts = [
        ("Run", f"`{run.get('run_id', '–')}` finished {run.get('finished_utc', '–')}"),
        ("Sextant", f"commit `{env.get('sextant_commit', '–')}`"
                    + (" (dirty worktree)" if env.get("sextant_worktree_dirty") else "")
                    + f", `{env.get('sextant_version', '–')}`"),
        ("Harness", f"fstandhartinger/jevbench @ `{run.get('harness_commit', '–')}` (MIT), adapter `typesafe`"),
        ("Endpoint", f"`{run.get('endpoint', '–')}` (serial requests, loopback)"),
        ("Hardware", f"{env.get('cpu_model', '–')}, {env.get('cpu_count', '–')} cores, {env.get('ram_gib', '–')} GiB RAM, {env.get('rustc', '–')}"),
        ("Task files", ", ".join(f"`{k}` {v[:12]}…" for k, v in sorted((run.get("dataset_sha256") or {}).items())) or "–"),
    ]
    lines += [table(["", ""], [[k, v] for k, v in facts]), ""]

    lines += ["## Tiers", ""]
    rows = []
    for key, t in doc["tiers"].items():
        b = t["metrics"]
        rows.append([
            f"**{t['name']}**" + (" (original)" if key == "original" else ""),
            f"{b['n_planned']} / {t['frozen_n']}",
            b["n_attempted"], fmt(b["schema_validity"], pct=True), fmt(b["schema_validity_strict"], pct=True),
            b["n_correct"], fmt(b["accuracy"], pct=True), fmt(b.get("chance_frozen_tier") or b["chance_public_items"], pct=True),
            fmt(b["intelligence"], 1), fmt(b["brier_mean"]), fmt(b["ece"]["ece"]) if b["ece"] else "–",
            fmt_s(b["latency"]["p50_s"]), fmt_s(b["latency"]["p95_s"]),
        ])
    rows.append(["judge", f"0 / {JUDGE_TIER_N}", "not public", "–", "–", "–", "–", fmt(chance_from_option_counts(TIER_OPTION_COUNTS["judge"]), pct=True), "–", "–", "–", "–", "–"])
    p = doc["pooled"]
    rows.append(["all public", p["n_planned"], p["n_attempted"], fmt(p["schema_validity"], pct=True), fmt(p["schema_validity_strict"], pct=True),
                 p["n_correct"], fmt(p["accuracy"], pct=True), fmt(p["chance_public_items"], pct=True), "–", fmt(p["brier_mean"]),
                 fmt(p["ece"]["ece"]) if p["ece"] else "–", fmt_s(p["latency"]["p50_s"]), fmt_s(p["latency"]["p95_s"])])
    lines += [table(["tier", "public / frozen", "attempted", "schema valid", "strict", "correct", "accuracy", "chance", "intelligence", "Brier", "ECE", "p50", "p95"], rows), ""]

    if "hard" in doc["tiers"]:
        lines += ["## Hard tier by family", ""]
        rows = []
        for fam, b in doc["tiers"]["hard"]["per_family"].items():
            rows.append([f"`{fam}`", b["n_planned"], b["n_correct"], fmt(b["accuracy"], pct=True), fmt(b["chance_public_items"], pct=True),
                         fmt(b["brier_mean"]), fmt(b["ece"]["ece"]) if b["ece"] else "–"])
        lines += [table(["family", "n", "correct", "accuracy", "chance", "Brier", "ECE"], rows), ""]

    lines += ["## By question type", ""]
    rows = []
    for key, t in doc["tiers"].items():
        for qt, b in t["per_question_type"].items():
            rows.append([t["name"], f"`{qt}`", b["n_planned"], b["n_correct"], fmt(b["accuracy"], pct=True), fmt(b["brier_mean"]),
                         fmt(b["ece"]["ece"]) if b["ece"] else "–", fmt(b["ordinal_mae"], 2)])
    lines += [table(["tier", "type", "n", "correct", "accuracy", "Brier", "ECE", "ordinal MAE"], rows), ""]

    lines += ["## Paraphrase pairs (`group`)", ""]
    rows = []
    for key, t in doc["tiers"].items():
        pp = t["paraphrase"]
        rows.append([t["name"], pp["pairs"], pp["both_valid"], pp["agree"], fmt(pp["agreement"], pct=True), pp["both_correct"],
                     fmt(pp["both_correct_rate_all_pairs"], pct=True)])
    lines += [table(["tier", "pairs", "both valid", "same answer", "agreement", "both correct", "both-correct rate (all pairs)"], rows), ""]

    lines += ["## Latency (caller wall time per decision, serial, loopback)", ""]
    rows = []
    for key, t in doc["tiers"].items():
        lt = t["metrics"]["latency"]
        rows.append([t["name"], lt["n"], fmt_s(lt["p50_s"]), fmt_s(lt["p95_s"]), fmt_s(lt["mean_s"]), fmt_s(lt["max_s"]), fmt_s(lt["first_s"]), t["metrics"]["n_failed"]])
    lt = p["latency"]
    rows.append(["all public", lt["n"], fmt_s(lt["p50_s"]), fmt_s(lt["p95_s"]), fmt_s(lt["mean_s"]), fmt_s(lt["max_s"]), fmt_s(lt["first_s"]), p["n_failed"]])
    lines += [table(["tier", "n", "p50", "p95", "mean", "max", "first request", "failed"], rows), ""]

    lines += ["## Definitions", ""]
    lines += [f"- {d}" for d in doc["definitions"]]
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks-dir", required=True, help="harness datasets/public directory")
    ap.add_argument("--results", required=True, nargs="+", help="harness results.jsonl file(s)")
    ap.add_argument("--tiers", default="easy,original,hard")
    ap.add_argument("--label", default=None, help="printed in bold at the top of the markdown")
    ap.add_argument("--manifest", default=None, help="run.sh manifest.json (run id, endpoint, hashes)")
    ap.add_argument("--environment", default=None, help="run.sh environment.json")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    records = []
    for path in args.results:
        records.extend(load_jsonl(path))
    ids = [r["task_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate task ids in results")
    by_id = {r["task_id"]: r for r in records}

    tiers, all_tasks = {}, []
    for key in [k.strip() for k in args.tiers.split(",") if k.strip()]:
        if key not in TIERS:
            raise SystemExit(f"unknown tier {key!r}; expected one of {', '.join(TIERS)}")
        name, fname, frozen_n = TIERS[key]
        tasks = load_jsonl(os.path.join(args.tasks_dir, fname))
        all_tasks.extend(tasks)
        tiers[key] = {
            "name": name,
            "file": fname,
            "frozen_n": frozen_n,
            "public_n": len(tasks),
            "metrics": with_intelligence(block(tasks, by_id), name),
            "per_family": by_key(tasks, by_id, lambda t: t["family"]),
            "per_question_type": by_key(tasks, by_id, lambda t: t["question"]["type"]),
            "paraphrase": paraphrase(tasks, by_id),
        }
    unknown = set(ids) - {t["id"] for t in all_tasks}
    if unknown:
        raise SystemExit(f"{len(unknown)} result records do not belong to the selected tiers")

    doc = {
        "benchmark": "JevBench public tiers",
        "label": args.label,
        "run": load_json(args.manifest),
        "environment": load_json(args.environment),
        "tiers": tiers,
        "pooled": block(all_tasks, by_id),
        "judge_tier": {"n": JUDGE_TIER_N, "public": False, "note": "imported decisions; ground truth and text not published"},
        "definitions": [
            "accuracy = correct / all public items of the tier (harness `n_correct / n_scorable`); unattempted, failed and schema-invalid answers count as wrong.",
            "schema valid = share of attempted items whose distribution covers exactly the label set, lies in [0, 1] and sums to 1 within 2 % (renormalized); strict = within 0.001 (harness `schema_validity` / `schema_validity_strict`).",
            "Brier = mean over valid items of sum_k (p_k - y_k)^2 over the exact label set (binary questions use both labels, i.e. 2·(p_yes − y)^2).",
            "ECE = top-label expected calibration error, 10 equal-width confidence bins, over valid scorable items (harness `ece_top_label`).",
            "chance = mean of 1/|labels| over the tier's frozen items (harness option-count histograms, JevBench v1.3); intelligence = 100·(accuracy − chance)/(1 − chance), clipped to [0, 100].",
            "paraphrase pairs = `group`s of exactly two items; agreement = same predicted label among pairs with two valid answers; both-correct rate uses all pairs as denominator (harness `agreement`).",
            "latency = caller wall time per decision including HTTP, one request at a time (harness `latency_s`); p50/p95 use the harness's interpolating percentile.",
        ],
    }
    md = render_md(doc)
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
    for key, t in tiers.items():
        b = t["metrics"]
        print(f"[summarize] {t['name']:<9} accuracy {b['n_correct']}/{b['n_scorable']} = {fmt(b['accuracy'], pct=True)}  "
              f"valid {fmt(b['schema_validity'], pct=True)}  Brier {fmt(b['brier_mean'])}  "
              f"ECE {fmt(b['ece']['ece']) if b['ece'] else '–'}  p50 {fmt_s(b['latency']['p50_s'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
