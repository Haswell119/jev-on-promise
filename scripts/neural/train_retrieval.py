#!/usr/bin/env python3
"""Fit the learned retrieval ranker.

The ranker is a linear model over the features `sextant export-retrieval`
dumps per segment. It is trained listwise: each question's segments form
one list and the loss is the negative log of the probability mass the
softmax puts on the annotated evidence segments. That matches how the
ranker is used (fill a word budget with the best segments) far better
than per-segment binary classification, which would spend its capacity on
the hundreds of easy negatives in a long state.

Deliberately linear, and deliberately over features retrieval already
computes: the ranker runs on every segment of every state on the
inference path, so it must cost one dot product. Capacity belongs in the
cross-encoder, which only ever sees the retrieved block.

    python3 scripts/neural/train_retrieval.py \
        --train data/retrieval/train.jsonl --dev data/retrieval/dev.jsonl \
        --out model/retrieval.json --version R3
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def load(paths, limit=0):
    """Flatten every list into one tensor plus a list id, which is what the
    vectorised segment-softmax below needs."""
    xs, ys, ws, ids, meta = [], [], [], [], []
    names = None
    nl = 0
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                if names is None:
                    names = r["feature_names"]
                elif r["feature_names"] != names:
                    raise SystemExit(f"{p}: feature names differ from the first file; re-export both")
                # Prefer the token-share weights; fall back to binary
                # labels for exports made before they existed.
                y = torch.tensor(r.get("label_weights") or r["labels"], dtype=torch.float32)
                if y.sum() == 0:
                    continue
                x = torch.tensor(r["features"], dtype=torch.float32)
                xs.append(x)
                ys.append(y)
                ws.append(torch.tensor(r["seg_words"], dtype=torch.float32))
                ids.append(torch.full((len(y),), nl, dtype=torch.long))
                meta.append({"kind": r.get("kind", ""), "n_segments": r.get("n_segments", len(y)),
                             "tier": r.get("tier", ""), "template_ids": r.get("template_ids", [])})
                nl += 1
                if limit and nl >= limit:
                    break
        if limit and nl >= limit:
            break
    if nl == 0:
        return None
    return {
        "x": torch.cat(xs),
        "y": torch.cat(ys),
        "w": torch.cat(ws),
        "list_id": torch.cat(ids),
        "n_lists": nl,
        "names": names,
        "meta": meta,
        "offsets": torch.tensor([0] + [len(v) for v in ys]).cumsum(0),
    }


def segment_logsumexp(scores, list_id, n_lists, mask=None):
    """log sum exp of `scores` within each list, optionally restricted to
    `mask`. Done with index_add so the whole dataset is one tensor op."""
    s = scores if mask is None else scores.masked_fill(~mask, float("-inf"))
    m = torch.full((n_lists,), float("-inf")).index_reduce_(0, list_id, s, "amax", include_self=True)
    e = torch.exp(s - m[list_id])
    e = torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    z = torch.zeros(n_lists).index_add_(0, list_id, e)
    return m + torch.log(z.clamp_min(1e-30))


def listwise_loss(scores, d):
    """Cross-entropy against the token-share target distribution.

    The old form, -log of the total mass on the positives, is indifferent
    between putting that mass on the segment carrying most of the span and
    the one carrying a sliver of it. Weighting the target by token share
    makes the loss care about the same thing the suite metric does."""
    logZ = segment_logsumexp(scores, d["list_id"], d["n_lists"])
    logp = scores - logZ[d["list_id"]]
    tgt_sum = torch.zeros(d["n_lists"]).index_add_(0, d["list_id"], d["y"]).clamp_min(1e-12)
    tgt = d["y"] / tgt_sum[d["list_id"]]
    per_list = torch.zeros(d["n_lists"]).index_add_(0, d["list_id"], -tgt * logp)
    return per_list.mean()


def budget_recall(scores, d, budget):
    """Share of the annotated span that survives a greedy fill of the word
    budget. With token-share labels this tracks the suite's token-level
    recall rather than a count of segments, which is the distinction that
    sank the first fit."""
    off = d["offsets"]
    got_total, top1 = 0.0, 0.0
    for i in range(d["n_lists"]):
        a, b = int(off[i]), int(off[i + 1])
        s, y, w = scores[a:b], d["y"][a:b], d["w"][a:b]
        order = torch.argsort(s, descending=True)
        top1 += float(y[order[0]] > 0)
        used, got = 0.0, 0.0
        for j in order.tolist():
            wj = float(w[j])
            if wj == 0 or used + wj > budget:
                continue
            used += wj
            got += float(y[j])
            if used >= budget:
                break
        total = float(y.sum())
        got_total += got / total if total else 1.0
    n = max(d["n_lists"], 1)
    return got_total / n, top1 / n


def evaluate(w, b, d, budget):
    with torch.no_grad():
        s = d["x"] @ w + b
        rec, top1 = budget_recall(s, d, budget)
        return {"loss": float(listwise_loss(s, d)), "budget_recall": rec, "top1": top1, "n_lists": d["n_lists"]}


def baseline(d, budget, name="pooled_rank"):
    """The heuristic the ranker has to beat: the pooled BM25 ordering."""
    idx = d["names"].index(name)
    with torch.no_grad():
        s = d["x"][:, idx].contiguous()
        rec, top1 = budget_recall(s, d, budget)
        return {"feature": name, "budget_recall": rec, "top1": top1, "n_lists": d["n_lists"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--version", default="")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--budget", type=int, default=140)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    tr = load(a.train, a.train_limit)
    dv = load(a.dev, a.dev_limit)
    if tr is None or dv is None:
        raise SystemExit("no usable lists (every row needs at least one annotated segment)")
    if tr["names"] != dv["names"]:
        raise SystemExit("train and dev were exported with different feature sets")
    names = tr["names"]
    print(f"train lists {tr['n_lists']} segments {len(tr['y'])} | dev lists {dv['n_lists']} | features {len(names)}", file=sys.stderr)

    base = baseline(dv, a.budget)
    print(f"[baseline {base['feature']}] budget_recall {base['budget_recall']:.4f} top1 {base['top1']:.4f}", file=sys.stderr)

    w = torch.zeros(len(names), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=a.lr)
    best = None
    for step in range(1, a.steps + 1):
        s = tr["x"] @ w + b
        loss = listwise_loss(s, tr) + a.l2 * (w * w).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(w.detach(), b.detach(), dv, a.budget)
            print(
                f"[step {step}] train_loss {float(loss):.4f} dev_loss {m['loss']:.4f} "
                f"budget_recall {m['budget_recall']:.4f} top1 {m['top1']:.4f}",
                file=sys.stderr,
            )
            if best is None or m["budget_recall"] > best["metrics"]["budget_recall"]:
                best = {"w": w.detach().clone(), "b": float(b.detach()), "metrics": m, "step": step}

    art = {
        "weights": [float(x) for x in best["w"]],
        "bias": best["b"],
        "version": a.version or "retrieval",
        "feature_names": names,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(art, indent=2) + "\n")
    report = {
        "out": a.out,
        "best_step": best["step"],
        "dev": best["metrics"],
        "baseline": base,
        "budget_words": a.budget,
        "train_lists": tr["n_lists"],
        "weights": dict(zip(names, art["weights"])),
        "bias": art["bias"],
    }
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
