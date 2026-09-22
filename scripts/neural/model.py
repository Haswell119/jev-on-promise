"""Sextant neural decision scorer.

A compact pretrained encoder scores one (question, candidate, evidence)
sequence into a scalar compatibility logit. A question's candidates are
normalised with a softmax over the group, so the output is a native
probability distribution over caller-supplied options — no fixed label
vocabulary anywhere.

Optionally the symbolic feature vector and the symbolic fusion logit are
concatenated to the pooled representation (hybrid head), which lets the
exact resolvers and lexical channels keep contributing.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DecisionScorer(nn.Module):
    def __init__(self, encoder_name: str, n_features: int = 0, use_symbolic_logit: bool = False, dropout: float = 0.1, pooling: str = "cls", revision: str | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, revision=revision)
        self.config = self.encoder.config
        h = self.config.hidden_size
        self.pooling = pooling
        self.n_features = n_features
        self.use_symbolic_logit = use_symbolic_logit
        extra = n_features + (1 if use_symbolic_logit else 0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h + extra, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 1))
        if extra:
            self.feat_norm = nn.LayerNorm(extra)

    def pool(self, out, attention_mask):
        hs = out.last_hidden_state
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            return (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return hs[:, 0]

    def forward(self, input_ids, attention_mask, token_type_ids=None, features=None, symbolic_logit=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.config, "type_vocab_size", 1) > 1:
            kw["token_type_ids"] = token_type_ids
        out = self.encoder(**kw)
        pooled = self.dropout(self.pool(out, attention_mask))
        parts = [pooled]
        extra = []
        if self.n_features and features is not None:
            extra.append(features)
        if self.use_symbolic_logit and symbolic_logit is not None:
            extra.append(symbolic_logit.unsqueeze(-1))
        if extra:
            e = torch.cat(extra, dim=-1)
            parts.append(self.feat_norm(e))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def freeze_bottom(model: DecisionScorer, n_layers: int):
    """Freeze embeddings + the first n encoder layers (CPU training speed)."""
    if n_layers <= 0:
        return 0
    frozen = 0
    emb = getattr(model.encoder, "embeddings", None)
    if emb is not None:
        for p in emb.parameters():
            p.requires_grad_(False)
            frozen += p.numel()
    layers = None
    for attr in ("encoder", "transformer"):
        sub = getattr(model.encoder, attr, None)
        if sub is not None:
            layers = getattr(sub, "layer", None) or getattr(sub, "layers", None)
            if layers is not None:
                break
    if layers is not None:
        for layer in list(layers)[:n_layers]:
            for p in layer.parameters():
                p.requires_grad_(False)
                frozen += p.numel()
    return frozen
