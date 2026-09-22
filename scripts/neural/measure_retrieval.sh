#!/usr/bin/env bash
# Measure a fitted retrieval ranker against the heuristic ordering on the
# frozen internal long-context suite.
#
#   scripts/neural/measure_retrieval.sh <retrieval.json> <tag>
#
# The ranker is fitted and selected on the TRAINING pool only, so this
# suite is an untouched measurement rather than the selection criterion.
set -euo pipefail
cd "$(dirname "$0")/../.."
RANKER="${1:?usage: measure_retrieval.sh <retrieval.json> <tag>}"
TAG="${2:?usage: measure_retrieval.sh <retrieval.json> <tag>}"
SUITE="${SEXTANT_LONGCTX_SET:-data/bench_internal/long_context.jsonl}"
BUDGET="${SEXTANT_EVIDENCE_WORDS:-140}"
BIN=./target/release/sextant
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p reports/retrieval

cp model/weights.json model/calibration.json "$WORK/"
cp "$RANKER" "$WORK/retrieval.json"

echo "== heuristic ordering (no ranker)"
$BIN --threads 1 export-pairs "$SUITE" --out "$WORK/base.jsonl" \
  --budget-words "$BUDGET" --no-features --local-idf --q-expand >/dev/null 2>&1
python3 scripts/neural/evidence_recall.py --records "$SUITE" --pairs "$WORK/base.jsonl" \
  --out "reports/retrieval/${TAG}_heuristic.json" > /dev/null

echo "== learned ranker"
$BIN --model-dir "$WORK" --threads 1 export-pairs "$SUITE" --out "$WORK/ranked.jsonl" \
  --budget-words "$BUDGET" --no-features --local-idf --q-expand >/dev/null 2>&1
python3 scripts/neural/evidence_recall.py --records "$SUITE" --pairs "$WORK/ranked.jsonl" \
  --out "reports/retrieval/${TAG}_ranked.json" > /dev/null

python3 - "$TAG" <<'PY'
import json, sys
tag = sys.argv[1]
a = json.load(open(f"reports/retrieval/{tag}_heuristic.json"))
b = json.load(open(f"reports/retrieval/{tag}_ranked.json"))
def sk(x):
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)
print(f"\n{'':<12}{'heuristic':>11}{'ranker':>9}{'delta':>9}")
for k in ("full", "partial", "lost"):
    print(f"{k:<12}{a[k]:>11.3f}{b[k]:>9.3f}{b[k]-a[k]:>+9.3f}")
for field, label in (("by_bucket", "tokens"), ("by_position", "position")):
    print(f"\n{label:<12}{'heuristic':>11}{'ranker':>9}{'delta':>9}")
    for k in sorted(a[field], key=sk):
        print(f"{k:<12}{a[field][k]:>11.3f}{b[field][k]:>9.3f}{b[field][k]-a[field][k]:>+9.3f}")
json.dump({"heuristic": a, "ranked": b}, open(f"reports/retrieval/{tag}_comparison.json", "w"), indent=2)
PY
