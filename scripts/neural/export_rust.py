#!/usr/bin/env python3
"""Export a trained scorer into the layout the Rust runtime loads:
config.json, model.safetensors, tokenizer.json, head.safetensors, scorer.json.
"""
import argparse, json, shutil, sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="experiments/runs/<id> containing best_encoder/ and best_head.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence-words", type=int, default=140)
    ap.add_argument("--version", default="")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = run / "best_encoder"
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src = enc / name
        if src.exists():
            shutil.copy2(src, out / name)
    head = torch.load(run / "best_head.pt", map_location="cpu", weights_only=False)
    cfg = head["config"]
    sd = head["head"]
    # nn.Sequential(Linear, GELU, Dropout, Linear) -> keys "0.weight","0.bias","3.weight","3.bias"
    tensors = {
        "l1.weight": sd["0.weight"].contiguous(),
        "l1.bias": sd["0.bias"].contiguous(),
        "l2.weight": sd["3.weight"].contiguous(),
        "l2.bias": sd["3.bias"].contiguous(),
    }
    if head.get("feat_norm"):
        tensors["feat_norm.weight"] = head["feat_norm"]["weight"].contiguous()
        tensors["feat_norm.bias"] = head["feat_norm"]["bias"].contiguous()
    save_file(tensors, str(out / "head.safetensors"))
    scorer = {
        "max_len": cfg["max_len"],
        "pooling": cfg["pooling"],
        "evidence_words": a.evidence_words,
        "n_features": cfg.get("n_features", 0),
        "use_symbolic_logit": bool(cfg.get("use_symbolic_logit", False)),
        "encoder": cfg["encoder"],
        "revision": cfg.get("revision") or "",
        "version": a.version or Path(a.run).name,
        "pair_sep": " | ",
    }
    (out / "scorer.json").write_text(json.dumps(scorer, indent=2) + "\n")
    total = sum(t.numel() for t in tensors.values())
    print(json.dumps({"out": str(out), "head_params": total, "metrics": head.get("metrics", {}).get("accuracy"), "scorer": scorer}, indent=2))


if __name__ == "__main__":
    main()
