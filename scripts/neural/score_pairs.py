#!/usr/bin/env python3
"""Score exported pair rows with the Python model (for Python/Rust parity
checks and for evaluating a run before it is exported to Rust)."""
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_groups  # noqa: E402
from model import DecisionScorer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from train import collate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run = Path(a.run)
    head = torch.load(run / "best_head.pt", map_location="cpu", weights_only=False)
    cfg = head["config"]
    tok = AutoTokenizer.from_pretrained(run / "best_encoder")
    model = DecisionScorer(str(run / "best_encoder"), n_features=cfg.get("n_features", 0), use_symbolic_logit=cfg.get("use_symbolic_logit", False), pooling=cfg["pooling"])
    model.head.load_state_dict(head["head"])
    if head.get("feat_norm") and hasattr(model, "feat_norm"):
        model.feat_norm.load_state_dict(head["feat_norm"])
    model.eval()
    need = cfg.get("n_features", 0) > 0 or cfg.get("use_symbolic_logit", False)
    groups = load_groups([a.input], limit=a.limit, with_features=need)
    t0 = time.time()
    with open(a.out, "w") as f, torch.no_grad():
        for i in range(0, len(groups), a.batch):
            chunk = groups[i : i + a.batch]
            inputs, offsets = collate(tok, chunk, cfg["max_len"], torch.device("cpu"), with_features=need)
            logits = model(**inputs)
            for g, (s, e) in zip(chunk, offsets):
                f.write(json.dumps({"id": g.gid, "logits": [round(float(x), 6) for x in logits[s:e]], "gold": g.gold, "kind": g.kind, "tier": g.tier, "record_family": g.record_family, "difficulty": g.extra.get("difficulty")}) + "\n")
    print(f"scored {len(groups)} questions in {time.time()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
