"""Dataset plumbing for the Sextant neural decision scorer.

Input = the JSONL produced by `sextant export-pairs` (question text,
retrieved evidence, candidate descriptions, symbolic features, gold index).
Every question becomes a GROUP of sequences, one per candidate; the loss is
a softmax over the group, so the model learns a general
compatibility(state, question, criterion) function rather than a fixed label
vocabulary.
"""
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

NOUL_DEFAULTS = {"true": "yes, the statement holds", "false": "no, the statement does not hold"}


@dataclass
class Group:
    gid: str
    kind: str
    family: str
    question: str
    evidence: str
    cand_texts: List[str]
    cand_keys: List[str]
    gold: int
    soft: Optional[List[float]] = None
    symbolic_logits: Optional[List[float]] = None
    features: Optional[List[List[float]]] = None
    source: str = ""
    tier: str = ""
    record_family: str = ""
    transformation: str = ""
    group_key: Optional[str] = None
    state_words: int = 0
    resolver: Optional[str] = None
    extra: Dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.cand_texts)


def _cand_text(kind: str, key: str, text: str) -> str:
    t = (text or "").strip()
    if kind == "noul":
        base = NOUL_DEFAULTS.get(key, key)
        return base if not t or t == key else f"{base}; {t}"
    if not t:
        return key
    return t


def load_groups(paths, limit: int = 0, kinds=None, with_features: bool = False) -> List[Group]:
    out: List[Group] = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if kinds and r["kind"] not in kinds:
                    continue
                cands = r["candidates"]
                g = Group(
                    gid=r["id"],
                    kind=r["kind"],
                    family=r.get("family", ""),
                    question=r.get("question", "").strip(),
                    evidence=r.get("evidence", "").strip(),
                    cand_texts=[_cand_text(r["kind"], c["key"], c.get("text", "")) for c in cands],
                    cand_keys=[c["key"] for c in cands],
                    gold=r["gold"],
                    soft=r.get("gold_probs"),
                    symbolic_logits=[c.get("symbolic_logit", 0.0) for c in cands],
                    features=[c.get("features") or [] for c in cands] if with_features else None,
                    source=r.get("source", ""),
                    tier=r.get("tier", ""),
                    record_family=r.get("record_family", ""),
                    transformation=r.get("transformation", ""),
                    group_key=r.get("group"),
                    state_words=r.get("state_words", 0),
                    resolver=r.get("path"),
                    extra=r.get("extra") or {},
                )
                if g.k >= 2 and 0 <= g.gold < g.k:
                    out.append(g)
                if limit and len(out) >= limit:
                    break
        if limit and len(out) >= limit:
            break
    return out


def subsample_candidates(g: Group, max_k: int, rng: random.Random) -> Group:
    """Keep the gold plus sampled negatives (training only). Order is
    reshuffled so candidate position can never be a shortcut."""
    if g.k <= max_k:
        idx = list(range(g.k))
    else:
        negs = [i for i in range(g.k) if i != g.gold]
        if g.kind == "score":
            # keep ordinal neighbours so the scale stays learnable
            near = sorted(negs, key=lambda i: abs(i - g.gold))[: max_k - 1]
            idx = sorted(set(near + [g.gold]))
        else:
            idx = rng.sample(negs, max_k - 1) + [g.gold]
    rng.shuffle(idx)
    soft = [g.soft[i] for i in idx] if g.soft and len(g.soft) == g.k else None
    if soft:
        s = sum(soft)
        soft = [x / s for x in soft] if s > 0 else None
    return Group(
        gid=g.gid,
        kind=g.kind,
        family=g.family,
        question=g.question,
        evidence=g.evidence,
        cand_texts=[g.cand_texts[i] for i in idx],
        cand_keys=[g.cand_keys[i] for i in idx],
        gold=idx.index(g.gold),
        soft=soft,
        symbolic_logits=[g.symbolic_logits[i] for i in idx] if g.symbolic_logits else None,
        features=[g.features[i] for i in idx] if g.features else None,
        source=g.source,
        tier=g.tier,
        record_family=g.record_family,
        transformation=g.transformation,
        group_key=g.group_key,
        state_words=g.state_words,
        resolver=g.resolver,
        extra=g.extra,
    )


def encode_group(tok, g: Group, max_len: int):
    """One sequence per candidate: [CLS] question [SEP] candidate [SEP] evidence."""
    first = [f"{g.question} | {c}" for c in g.cand_texts]
    second = [g.evidence] * g.k
    return tok(first, second, truncation="only_second", max_length=max_len, padding=False)


def group_hash(g: Group) -> str:
    return hashlib.sha1((g.question + "\x00" + g.evidence).encode()).hexdigest()[:16]
