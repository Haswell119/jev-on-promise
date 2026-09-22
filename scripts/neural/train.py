#!/usr/bin/env python3
"""Train the Sextant neural decision scorer (CPU-friendly, resumable).

Usage:
  python3 scripts/neural/train.py --train pairs/train.jsonl --dev pairs/dev.jsonl \
      --out experiments/runs/E1 --encoder sentence-transformers/all-MiniLM-L6-v2 \
      --epochs 2 --max-len 192 --group-batch 4 --max-candidates 8

Groups (one question with its candidates) are the unit of batching; several
groups are packed into a step with dynamic padding and length bucketing so
CPU time is spent on real tokens only. Checkpoints hold model, optimizer,
scheduler, RNG and data-cursor state, so training resumes exactly.
"""
import argparse, json, math, os, random, signal, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Group, encode_group, load_groups, subsample_candidates  # noqa: E402
from losses import group_loss  # noqa: E402
from model import DecisionScorer, freeze_bottom  # noqa: E402

STOP = False


def _sigterm(*_):
    global STOP
    STOP = True
    print("[train] stop requested; checkpointing after this step", flush=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def collate(tok, groups, max_len, device, with_features=False):
    encs, offsets, n = [], [], 0
    for g in groups:
        e = encode_group(tok, g, max_len)
        encs.append(e)
        offsets.append((n, n + g.k))
        n += g.k
    payload = {"input_ids": [x for e in encs for x in e["input_ids"]]}
    if "token_type_ids" in encs[0]:
        # `tok.pad` only pads the keys it is handed: token_type_ids must be
        # passed explicitly, otherwise the encoder silently sees all-zero
        # segment ids and Python/Rust parity breaks.
        payload["token_type_ids"] = [x for e in encs for x in e["token_type_ids"]]
    batch = tok.pad(payload, return_tensors="pt")
    out = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
    if "token_type_ids" in batch:
        out["token_type_ids"] = batch["token_type_ids"].to(device)
    if with_features:
        feats = [f for g in groups for f in (g.features or [[]] * g.k)]
        if feats and feats[0]:
            out["features"] = torch.tensor(feats, dtype=torch.float32, device=device)
        sym = [s for g in groups for s in (g.symbolic_logits or [0.0] * g.k)]
        out["symbolic_logit"] = torch.tensor(sym, dtype=torch.float32, device=device)
    return out, offsets


def buckets(groups, group_batch, rng, tok_estimate):
    """Length-bucketed shuffling: sort by estimated length inside shuffled
    mega-batches so padding stays small while ordering stays stochastic."""
    idx = list(range(len(groups)))
    rng.shuffle(idx)
    mega = group_batch * 32
    batches = []
    for i in range(0, len(idx), mega):
        chunk = sorted(idx[i : i + mega], key=lambda j: tok_estimate[j])
        for s in range(0, len(chunk), group_batch):
            batches.append(chunk[s : s + group_batch])
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(model, tok, groups, max_len, device, args, max_groups=0, batch=8):
    model.eval()
    gs = groups[:max_groups] if max_groups else groups
    tot, correct, nll, brier = 0, 0, 0.0, 0.0
    confs, hits = [], []
    per_tier = {}
    for i in range(0, len(gs), batch):
        chunk = gs[i : i + batch]
        inputs, offsets = collate(tok, chunk, max_len, device, with_features=args.use_features or args.use_symbolic_logit)
        logits = model(**inputs)
        for g, (a, b) in zip(chunk, offsets):
            p = F.softmax(logits[a:b], dim=-1)
            pred = int(torch.argmax(p).item())
            ok = pred == g.gold
            tot += 1
            correct += ok
            nll += -math.log(max(float(p[g.gold]), 1e-12))
            oh = torch.zeros_like(p); oh[g.gold] = 1.0
            brier += float(((p - oh) ** 2).sum())
            confs.append(float(p.max())); hits.append(ok)
            t = per_tier.setdefault(g.tier or g.record_family or "all", [0, 0])
            t[0] += 1; t[1] += ok
    ece = 0.0
    if confs:
        bins = 10
        for bi in range(bins):
            lo, hi = bi / bins, (bi + 1) / bins
            sel = [(c, h) for c, h in zip(confs, hits) if (c > lo or bi == 0) and c <= hi]
            if sel:
                acc = sum(h for _, h in sel) / len(sel)
                conf = sum(c for c, _ in sel) / len(sel)
                ece += len(sel) / len(confs) * abs(acc - conf)
    model.train()
    return {
        "n": tot,
        "accuracy": correct / max(tot, 1),
        "nll": nll / max(tot, 1),
        "brier": brier / max(tot, 1),
        "ece": ece,
        "per_group": {k: {"n": v[0], "accuracy": v[1] / v[0]} for k, v in sorted(per_tier.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--group-batch", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--freeze-layers", type=int, default=0)
    ap.add_argument("--brier-weight", type=float, default=0.0)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--ordinal-smooth", type=float, default=0.08)
    ap.add_argument("--use-features", action="store_true")
    ap.add_argument("--use-symbolic-logit", action="store_true")
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--drop-sources", default="", help="comma-separated source substrings to exclude from training")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder, revision=args.revision)

    need_feats = args.use_features or args.use_symbolic_logit
    t0 = time.time()
    train = load_groups(args.train, limit=args.train_limit, with_features=need_feats)
    drops = [d for d in args.drop_sources.split(",") if d]
    if drops:
        train = [g for g in train if not any(d in g.source for d in drops)]
    dev = load_groups(args.dev, limit=args.dev_limit, with_features=need_feats)
    print(f"[train] {len(train)} train groups, {len(dev)} dev groups, loaded in {time.time()-t0:.1f}s", flush=True)

    n_features = len(train[0].features[0]) if (args.use_features and train and train[0].features and train[0].features[0]) else 0
    model = DecisionScorer(args.encoder, n_features=n_features, use_symbolic_logit=args.use_symbolic_logit, pooling=args.pooling, revision=args.revision).to(device)
    frozen = freeze_bottom(model, args.freeze_layers)
    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params={n_params/1e6:.1f}M trainable={n_train_params/1e6:.1f}M frozen={frozen/1e6:.1f}M features={n_features} device={device}", flush=True)

    head_names = [n for n, _ in model.named_parameters() if n.startswith("head") or n.startswith("feat_norm")]
    groups_params = [
        {"params": [p for n, p in model.named_parameters() if n in head_names and p.requires_grad], "lr": args.head_lr},
        {"params": [p for n, p in model.named_parameters() if n not in head_names and p.requires_grad], "lr": args.lr},
    ]
    opt = torch.optim.AdamW(groups_params, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    tok_estimate = [len(g.question) + len(g.evidence) + max((len(c) for c in g.cand_texts), default=0) for g in train]
    all_batches = []
    for _ in range(max(1, math.ceil(args.epochs))):
        all_batches.extend(buckets(train, args.group_batch, rng, tok_estimate))
    total_steps = int(len(all_batches) * min(1.0, args.epochs / max(1, math.ceil(args.epochs))))
    all_batches = all_batches[:total_steps]
    warmup_steps = max(1, int(total_steps * args.warmup))

    def lr_at(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    start_step = 0
    best = -1.0
    ckpt_path = out / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]; best = ck.get("best", -1.0)
        random.setstate(ck["py_rng"]); torch.set_rng_state(ck["torch_rng"])
        print(f"[train] resumed at step {start_step}/{total_steps}", flush=True)

    cfg = vars(args) | {"n_params": n_params, "total_steps": total_steps, "device": str(device), "n_features": n_features}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log = (out / "train_log.jsonl").open("a")
    t_start = time.time()
    seen = 0
    running = 0.0
    model.train()
    for step in range(start_step, total_steps):
        for pg, lr in zip(opt.param_groups, [args.head_lr, args.lr]):
            pg["lr"] = lr * lr_at(step)
        batch_groups = [subsample_candidates(train[i], args.max_candidates, rng) for i in all_batches[step]]
        inputs, offsets = collate(tok, batch_groups, args.max_len, device, with_features=need_feats)
        logits = model(**inputs)
        loss = 0.0
        for g, (a, b) in zip(batch_groups, offsets):
            soft = torch.tensor(g.soft, device=device) if g.soft else None
            loss = loss + group_loss(logits[a:b], g.gold, g.kind, soft=soft, brier_weight=args.brier_weight, ordinal_smooth=args.ordinal_smooth, label_smooth=args.label_smooth)
        loss = loss / len(batch_groups)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        running += float(loss)
        seen += sum(g.k for g in batch_groups)
        if (step + 1) % 50 == 0:
            el = time.time() - t_start
            msg = {"step": step + 1, "total": total_steps, "loss": running / 50, "seq_per_s": seen / el, "elapsed_s": el, "lr": opt.param_groups[1]["lr"]}
            print(f"[train] step {step+1}/{total_steps} loss {msg['loss']:.4f} {msg['seq_per_s']:.1f} seq/s eta {(total_steps-step-1)*el/max(1,step+1-start_step)/60:.0f}m", flush=True)
            log.write(json.dumps(msg) + "\n"); log.flush()
            running = 0.0
        need_stop = STOP or (args.max_seconds and time.time() - t_start > args.max_seconds)
        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps or need_stop:
            m = evaluate(model, tok, dev, args.max_len, device, args)
            m["step"] = step + 1
            print(f"[eval] step {step+1} acc {m['accuracy']:.4f} nll {m['nll']:.3f} brier {m['brier']:.3f} ece {m['ece']:.3f}", flush=True)
            log.write(json.dumps({"eval": m}) + "\n"); log.flush()
            (out / "metrics.json").write_text(json.dumps(m, indent=2))
            if m["accuracy"] > best:
                best = m["accuracy"]
                model.encoder.save_pretrained(out / "best_encoder", safe_serialization=True)
                tok.save_pretrained(out / "best_encoder")
                torch.save({"head": model.head.state_dict(), "feat_norm": model.feat_norm.state_dict() if hasattr(model, "feat_norm") else None, "config": cfg, "metrics": m}, out / "best_head.pt")
                (out / "best_metrics.json").write_text(json.dumps(m, indent=2))
        if (step + 1) % args.save_every == 0 or need_stop:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step + 1, "best": best, "py_rng": random.getstate(), "torch_rng": torch.get_rng_state(), "config": cfg}, ckpt_path)
        if need_stop:
            print("[train] stopping early (signal or time budget)", flush=True)
            break
    log.close()
    print(f"[train] done in {(time.time()-t_start)/60:.1f} min, best dev accuracy {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
