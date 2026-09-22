#!/usr/bin/env python3
"""Fit the neural fusion block of model/calibration.json on a held-out
calibration split: the neural/symbolic mixing weights, the temperature per
primitive, Platt scaling for Noul and the confidence map.

Inputs: the exported pairs (carrying the symbolic logits) and the neural
logits produced by `sextant neural-probe` on the same file.
"""
import argparse, json, math
from collections import defaultdict


def load(pairs, probe):
    sym, meta = {}, {}
    for line in open(pairs):
        r = json.loads(line)
        sym[r["id"]] = [c.get("symbolic_logit", 0.0) for c in r["candidates"]]
        meta[r["id"]] = {"kind": r["kind"], "gold": r["gold"], "family": r.get("family", ""), "path": r.get("path", "semantic")}
    rows = []
    for line in open(probe):
        r = json.loads(line)
        i = r["id"]
        if i in sym and len(sym[i]) == len(r["logits"]):
            rows.append({"id": i, "neural": r["logits"], "symbolic": sym[i], **meta[i]})
    return rows


def softmax(z, t=1.0):
    m = max(z)
    e = [math.exp((v - m) / max(t, 1e-3)) for v in z]
    s = sum(e)
    return [x / s for x in e]


def nll_of(rows, wn, ws, temps):
    tot = 0.0
    for r in rows:
        z = [wn * n + ws * s for n, s in zip(r["neural"], r["symbolic"])]
        p = softmax(z, temps.get(r["kind"], 1.0))
        tot -= math.log(max(p[r["gold"]], 1e-12))
    return tot / max(len(rows), 1)


def acc_of(rows, wn, ws, temps):
    ok = 0
    for r in rows:
        z = [wn * n + ws * s for n, s in zip(r["neural"], r["symbolic"])]
        ok += max(range(len(z)), key=lambda i: z[i]) == r["gold"]
    return ok / max(len(rows), 1)


def fit_temperature(rows, wn, ws, kind):
    sub = [r for r in rows if r["kind"] == kind]
    if not sub:
        return 1.0
    best = (float("inf"), 1.0)
    for i in range(-24, 25):
        t = math.exp(i * 0.12)
        v = nll_of(sub, wn, ws, {kind: t})
        if v < best[0]:
            best = (v, t)
    return best[1]


def fit_platt(pairs):
    a, b = 1.0, 0.0
    n = max(len(pairs), 1)
    for _ in range(3000):
        ga = gb = 0.0
        for z, y in pairs:
            p = 1 / (1 + math.exp(-(a * z + b)))
            ga += (p - y) * z
            gb += p - y
        a -= 0.05 * (ga / n + 1e-3 * (a - 1.0))
        b -= 0.05 * (gb / n)
    return a, b


def fit_confidence(rows, wn, ws, temps):
    """Logistic map of (concentration, margin, evidence, ood) -> correctness."""
    X, Y = [], []
    for r in rows:
        z = [wn * n + ws * s for n, s in zip(r["neural"], r["symbolic"])]
        p = softmax(z, temps.get(r["kind"], 1.0))
        k = len(p)
        h = -sum(x * math.log(max(x, 1e-12)) for x in p) / math.log(k) if k > 1 else 0.0
        srt = sorted(p, reverse=True)
        margin = srt[0] - (srt[1] if len(srt) > 1 else 0.0)
        X.append([1 - h, margin, 0.0, 0.0, 1.0])
        Y.append(1.0 if max(range(len(z)), key=lambda i: z[i]) == r["gold"] else 0.0)
    w = [0.0] * 5
    n = max(len(X), 1)
    for _ in range(4000):
        g = [0.0] * 5
        for x, y in zip(X, Y):
            zz = sum(wi * xi for wi, xi in zip(w, x))
            p = 1 / (1 + math.exp(-max(-30, min(30, zz))))
            d = p - y
            for i in range(5):
                g[i] += d * x[i]
        for i in range(5):
            w[i] -= 0.1 * (g[i] / n + 1e-4 * w[i])
    return {"a_concentration": w[0], "b_margin": w[1], "c_evidence": w[2], "d_ood": w[3], "bias": w[4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--base-calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolver-priority", default="true")
    a = ap.parse_args()
    rows = load(a.pairs, a.probe)
    # Resolver-answered questions keep the symbolic path; fit on the rest.
    fit_rows = [r for r in rows if r["path"] == "semantic"]
    print(f"calibration rows: {len(rows)} ({len(fit_rows)} semantic)")
    best = None
    for wn in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        for ws in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            if wn == 0 and ws == 0:
                continue
            temps = {k: fit_temperature(fit_rows, wn, ws, k) for k in ("choice", "score", "noul")}
            v = nll_of(fit_rows, wn, ws, temps)
            acc = acc_of(fit_rows, wn, ws, temps)
            if best is None or v < best[0]:
                best = (v, wn, ws, temps, acc)
    nll, wn, ws, temps, acc = best
    print(f"fused: weight_neural={wn} weight_symbolic={ws} temps={ {k: round(v,3) for k,v in temps.items()} } calib NLL={nll:.4f} acc={acc:.4f}")
    noul = [r for r in fit_rows if r["kind"] == "noul"]
    platt = None
    if noul:
        pts = []
        for r in noul:
            z = [wn * n + ws * s for n, s in zip(r["neural"], r["symbolic"])]
            pts.append((z[0] - z[1], 1.0 if r["gold"] == 0 else 0.0))
        pa, pb = fit_platt(pts)
        platt = {"a": pa, "b": pb}
        print(f"noul platt a={pa:.3f} b={pb:.3f} (n={len(pts)})")
    conf = {k: fit_confidence([r for r in fit_rows if r["kind"] == k], wn, ws, temps) for k in ("choice", "score")}
    cal = json.load(open(a.base_calibration))
    cal["neural"] = {
        "weight_neural": wn,
        "weight_symbolic": ws,
        "temperature": temps,
        "noul_platt": platt,
        "resolver_priority": a.resolver_priority.lower() == "true",
        "confidence": conf,
    }
    cal["version"] = cal.get("version", "") + "+neural"
    json.dump(cal, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
