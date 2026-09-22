"""Training objectives for grouped candidate scoring."""
import torch
import torch.nn.functional as F


def ordinal_targets(k: int, gold: int, smooth: float, device) -> torch.Tensor:
    t = torch.zeros(k, device=device)
    t[gold] = 1.0
    if smooth > 0 and k >= 3:
        nbrs = [i for i in (gold - 1, gold + 1) if 0 <= i < k]
        t[gold] -= smooth * len(nbrs)
        for i in nbrs:
            t[i] += smooth
    return t


def group_loss(logits, gold, kind, soft=None, brier_weight=0.0, ordinal_smooth=0.08, label_smooth=0.0):
    """Cross entropy over one candidate group (+ optional Brier term)."""
    k = logits.numel()
    if soft is not None:
        target = soft
    elif kind == "score":
        target = ordinal_targets(k, gold, ordinal_smooth, logits.device)
    else:
        target = torch.zeros(k, device=logits.device)
        target[gold] = 1.0
        if label_smooth > 0:
            target = target * (1 - label_smooth) + label_smooth / k
    logp = F.log_softmax(logits, dim=-1)
    ce = -(target * logp).sum()
    if brier_weight > 0:
        p = logp.exp()
        ce = ce + brier_weight * ((p - target) ** 2).sum()
    return ce
