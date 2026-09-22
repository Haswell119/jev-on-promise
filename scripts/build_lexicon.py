#!/usr/bin/env python3
"""Build the compact lexical assets consumed by the Rust runtime.

Inputs (downloaded by scripts/fetch_resources.sh, licenses in THIRD_PARTY.md):
  * Open English WordNet 2024, WNDB format (CC BY 4.0)  -> wn_lemmas.tsv, wn_synsets.tsv, wn_antonyms.tsv
  * VADER sentiment lexicon (MIT)                        -> sentiment.tsv
  * Google Books Ngram derived 1-gram list (CC BY 3.0)   -> word_freq.tsv

Everything is deterministic: same inputs -> byte-identical outputs.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict

POS_FILES = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}
WORD_RE = re.compile(r"^[a-z][a-z'\-]*(_[a-z][a-z'\-]*)*$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_lemma(w):
    w = w.lower()
    w = re.sub(r"\([a-z]+\)$", "", w)  # adjective markers (a) (p) (ip)
    return w


def build_wordnet(wn_dir, out_dir):
    synsets = OrderedDict()  # key "n02084071" -> dict(lemmas=[...], hyper=[...], similar=[...])
    antonyms = set()
    for pos, name in POS_FILES.items():
        path = os.path.join(wn_dir, f"data.{name}")
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("  ") or not line.strip():
                    continue
                body = line.split("|", 1)[0].split()
                offset, lexno, ss_type, w_cnt = body[0], body[1], body[2], int(body[3], 16)
                words = []
                i = 4
                for _ in range(w_cnt):
                    words.append(clean_lemma(body[i]))
                    i += 2
                p_cnt = int(body[i])
                i += 1
                hyper, similar, domains = [], [], []
                sid = ss_type.replace("s", "a") + offset
                for _ in range(p_cnt):
                    sym, toff, tpos, st = body[i], body[i + 1], body[i + 2], body[i + 3]
                    i += 4
                    tid = tpos.replace("s", "a") + toff
                    if sym in ("@", "@i"):
                        hyper.append(tid)
                    elif sym in ("&", "+", "\\", "^"):
                        similar.append(tid)
                    elif sym == ";c":
                        domains.append(tid)
                    elif sym == "!":
                        src, tgt = int(st[:2], 16), int(st[2:], 16)
                        if 1 <= src <= len(words):
                            antonyms.add((words[src - 1], tid, tgt))
                synsets[sid] = {"lemmas": words, "hyper": hyper, "similar": similar, "domains": domains}
    # resolve antonym targets to lemma names
    ant_pairs = set()
    for src_word, tid, tgt in antonyms:
        target = synsets.get(tid)
        if target and 1 <= tgt <= len(target["lemmas"]):
            a, b = src_word, target["lemmas"][tgt - 1]
            if WORD_RE.match(a) and WORD_RE.match(b) and a != b:
                ant_pairs.add(tuple(sorted((a, b))))
    # lemma -> synsets in sense order from index files
    lemma_synsets = OrderedDict()
    for pos, name in POS_FILES.items():
        path = os.path.join(wn_dir, f"index.{name}")
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("  ") or not line.strip():
                    continue
                parts = line.split()
                lemma, p = parts[0].lower(), parts[1]
                if not WORD_RE.match(lemma):
                    continue
                p_cnt = int(parts[3])
                offsets = parts[6 + p_cnt:]
                ids = [p.replace("s", "a") + o for o in offsets]
                lemma_synsets.setdefault(lemma, []).extend(ids)
    # dense ids
    dense = {sid: i for i, sid in enumerate(synsets.keys())}
    n_lem, n_syn, n_ant = 0, 0, 0
    with open(os.path.join(out_dir, "wn_lemmas.tsv"), "w", encoding="utf-8") as f:
        f.write("# Open English WordNet 2024 (CC BY 4.0), lemma -> synsets in sense order. See THIRD_PARTY.md\n")
        for lemma, ids in lemma_synsets.items():
            ids = [str(dense[s]) for s in ids if s in dense]
            if not ids:
                continue
            f.write(f"{lemma}\t{','.join(ids)}\n")
            n_lem += 1
    with open(os.path.join(out_dir, "wn_synsets.tsv"), "w", encoding="utf-8") as f:
        f.write("# Open English WordNet 2024 (CC BY 4.0): id, lemmas, hypernyms, similar/derivational, topic domains. See THIRD_PARTY.md\n")
        for sid, s in synsets.items():
            lemmas = [w for w in s["lemmas"] if WORD_RE.match(w)]
            hyper = [str(dense[h]) for h in s["hyper"] if h in dense]
            sim = [str(dense[h]) for h in s["similar"] if h in dense]
            dom = [str(dense[h]) for h in s["domains"] if h in dense]
            f.write(f"{dense[sid]}\t{'|'.join(lemmas)}\t{','.join(hyper)}\t{','.join(sim)}\t{','.join(dom)}\n")
            n_syn += 1
    with open(os.path.join(out_dir, "wn_antonyms.tsv"), "w", encoding="utf-8") as f:
        f.write("# Open English WordNet 2024 (CC BY 4.0): antonym lemma pairs. See THIRD_PARTY.md\n")
        for a, b in sorted(ant_pairs):
            f.write(f"{a}\t{b}\n")
            n_ant += 1
    return {"lemmas": n_lem, "synsets": n_syn, "antonym_pairs": n_ant}


def build_sentiment(vader_path, out_dir):
    n = 0
    with open(vader_path, encoding="utf-8") as f, open(os.path.join(out_dir, "sentiment.tsv"), "w", encoding="utf-8") as out:
        out.write("# VADER sentiment lexicon (MIT, C.J. Hutto): token, mean valence in [-4, 4]. See THIRD_PARTY.md\n")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            tok, val = parts[0].strip().lower(), parts[1].strip()
            if not re.match(r"^[a-z][a-z'\-]*$", tok):
                continue
            try:
                v = float(val)
            except ValueError:
                continue
            out.write(f"{tok}\t{v}\n")
            n += 1
    return {"entries": n}


def build_freq(csv_path, out_dir):
    n = 0
    with open(csv_path, encoding="utf-8") as f, open(os.path.join(out_dir, "word_freq.tsv"), "w", encoding="utf-8") as out:
        out.write("# Derived from the Google Books Ngram Corpus v3 via orgtre/google-books-ngram-frequency (CC BY 3.0). See THIRD_PARTY.md\n")
        r = csv.DictReader(f)
        for row in r:
            w = row["ngram"].strip().lower()
            if not re.match(r"^[a-z][a-z'\-]*$", w):
                continue
            out.write(f"{w}\t{int(float(row['freq']))}\n")
            n += 1
    return {"entries": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wordnet-dir", required=True, help="directory containing data.noun, index.noun, ...")
    ap.add_argument("--vader", required=True)
    ap.add_argument("--freq-csv", required=True)
    ap.add_argument("--out-dir", default="crates/core/assets/lexicon")
    ap.add_argument("--provenance", default="data/provenance/lexicon.json")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    stats = {
        "wordnet": build_wordnet(a.wordnet_dir, a.out_dir),
        "sentiment": build_sentiment(a.vader, a.out_dir),
        "word_freq": build_freq(a.freq_csv, a.out_dir),
    }
    prov = {
        "generated_by": "scripts/build_lexicon.py",
        "inputs": {
            "wordnet_dir": a.wordnet_dir,
            "vader": {"path": a.vader, "sha256": sha256(a.vader)},
            "freq_csv": {"path": a.freq_csv, "sha256": sha256(a.freq_csv)},
        },
        "outputs": {name: {"sha256": sha256(os.path.join(a.out_dir, name)), "bytes": os.path.getsize(os.path.join(a.out_dir, name))} for name in ["wn_lemmas.tsv", "wn_synsets.tsv", "wn_antonyms.tsv", "sentiment.tsv", "word_freq.tsv"]},
        "stats": stats,
    }
    os.makedirs(os.path.dirname(a.provenance), exist_ok=True)
    with open(a.provenance, "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
