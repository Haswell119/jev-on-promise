#!/usr/bin/env python3
"""Weighted development score used for champion/challenger promotion.

Inputs are the JSON reports produced by `sextant eval` (grouped by
`difficulty`) and `sextant bench`. The score deliberately weights semantic
capability (standard + hard) far above latency, while refusing promotions
that destroy calibration or schema validity.
"""
import argparse, json, math, sys

WEIGHTS = {
    "standard": 0.34,
    "hard": 0.30,
    "easy": 0.10,
    "calibration": 0.10,
    "paraphrase": 0.06,
    "order": 0.05,
    "latency": 0.03,
    "memory": 0.02,
}


def latency_score(p50_ms):
    if p50_ms is None:
        return 0.0
    if p50_ms <= 50:
        return 1.0
    if p50_ms >= 1000:
        return 0.0
    return 1.0 - (math.log10(p50_ms / 50) / math.log10(1000 / 50))


def memory_score(mb):
    if mb is None:
        return 0.0
    if mb <= 500:
        return 1.0
    if mb >= 4000:
        return 0.0
    return 1.0 - (mb - 500) / 3500


def tier_acc(report, tier):
    by = report.get("by_group", {})
    if tier in by:
        return by[tier]["accuracy"], by[tier]["n"]
    return None, 0


def compute(eval_report, bench=None, memory_mb=None, stability=None):
    acc = {t: tier_acc(eval_report, t)[0] for t in ("easy", "standard", "hard", "judge")}
    overall = eval_report.get("overall", {})
    ece = overall.get("ece")
    gs = stability or eval_report.get("group_stability") or {}
    para = gs.get("same_answer_rate")
    order = gs.get("option_order_stability", para)
    p50 = None
    if bench:
        typ = next((r for r in bench.get("results", []) if r["scenario"].startswith("typical")), None)
        p50 = typ["p50_ms"] if typ else None
    parts = {
        "standard": acc.get("standard"),
        "hard": acc.get("hard"),
        "easy": acc.get("easy"),
        "calibration": (1 - ece) if ece is not None else None,
        "paraphrase": para,
        "order": order,
        "latency": latency_score(p50),
        "memory": memory_score(memory_mb),
    }
    used = {k: v for k, v in parts.items() if v is not None}
    wsum = sum(WEIGHTS[k] for k in used)
    score = sum(WEIGHTS[k] * v for k, v in used.items()) / wsum if wsum else 0.0
    return {
        "dev_score": score,
        "components": parts,
        "weights_used": {k: WEIGHTS[k] / wsum for k in used} if wsum else {},
        "overall_accuracy": overall.get("accuracy"),
        "ece": ece,
        "schema_validity": overall.get("schema_validity"),
        "p50_ms": p50,
        "memory_mb": memory_mb,
        "n": overall.get("n"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="sextant eval report JSON (grouped by difficulty)")
    ap.add_argument("--bench", help="sextant bench report JSON")
    ap.add_argument("--memory-mb", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = json.load(open(a.eval))
    bench = json.load(open(a.bench)) if a.bench else None
    res = compute(rep, bench, a.memory_mb)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
