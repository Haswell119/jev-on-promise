#!/usr/bin/env bash
# Reproduce the model artifact (model/weights.json + model/calibration.json)
# from allowed training data. Deterministic: fixed seeds, full-batch optimizers.
#
# Usage: scripts/train.sh [OUT_DIR]        (default: model)
# Env:   SEXTANT_SKIP_DATA=1  to skip dataset preparation (use existing data/processed)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-model}"

echo "== 1/5 build (release)"
cargo build --release --quiet

echo "== 2/5 synthetic recipes (seed 20260922)"
python3 scripts/synth/generate.py --seed 20260922 --out data/synthetic >/dev/null

if [ "${SEXTANT_SKIP_DATA:-0}" != "1" ]; then
  echo "== 3/5 permissive public datasets (downloads into data/raw, converts into data/processed)"
  python3 scripts/prepare_data.py
else
  echo "== 3/5 skipping dataset preparation (SEXTANT_SKIP_DATA=1)"
fi

# HWU64 is evaluation-only: its utterances overlap with MASSIVE (a jev-bench source), see data/README.md.
TRAIN=(data/synthetic/train.jsonl)
CALIB=(data/synthetic/calib.jsonl)
if [ -d data/processed ]; then
  while IFS= read -r f; do TRAIN+=("$f"); done < <(find data/processed -name train.jsonl -not -path "*/hwu64/*" | sort)
  while IFS= read -r f; do CALIB+=("$f"); done < <(find data/processed -name calib.jsonl -not -path "*/hwu64/*" | sort)
fi

echo "== contamination pre-scan: training records whose state matches an external eval set are excluded"
EVAL=(data/synthetic/dev.jsonl)
[ -d benchmarks/jevbench/harness/datasets/public ] && EVAL+=(benchmarks/jevbench/harness/datasets/public)
[ -d benchmarks/jev_bench_hf/data ] && EVAL+=(benchmarks/jev_bench_hf/data)
./target/release/sextant leakage --train "${TRAIN[@]}" "${CALIB[@]}" --eval "${EVAL[@]}" --out reports/leakage_prescan.json --exclusions-out data/exclusions.json || true

echo "== 4/5 train fusion weights → $OUT/weights.json"
./target/release/sextant train "${TRAIN[@]}" --out-dir "$OUT" --seed 42 --exclude-ids data/exclusions.json

echo "== 5/5 calibrate on the disjoint calibration split → $OUT/calibration.json"
./target/release/sextant calibrate "${CALIB[@]}" --out-dir "$OUT" --compare-isotonic

echo "== contamination check of the data actually used (after exclusions)"
python3 - "${TRAIN[@]}" "${CALIB[@]}" <<'PY'
import json, sys, os
ids = set(json.load(open("data/exclusions.json"))["exclude_ids"]) if os.path.exists("data/exclusions.json") else set()
os.makedirs("reports/raw", exist_ok=True)
with open("reports/raw/training_used.jsonl", "w") as out:
    for f in sys.argv[1:]:
        for line in open(f):
            if line.strip() and json.loads(line)["id"] not in ids:
                out.write(line)
PY
./target/release/sextant leakage --train reports/raw/training_used.jsonl --eval "${EVAL[@]}" --out reports/leakage.json || echo "leakage check reported findings; see reports/leakage.json"

echo "== internal dev evaluation"
./target/release/sextant --model-dir "$OUT" eval data/synthetic/dev.jsonl --group-by family --out reports/internal_dev.json
if [ -d data/processed ]; then
  DEV=()
  while IFS= read -r f; do DEV+=("$f"); done < <(find data/processed -name dev.jsonl | sort)
  ./target/release/sextant --model-dir "$OUT" eval "${DEV[@]}" --group-by source --out reports/external_dev.json
fi
echo "done: $OUT/weights.json $OUT/calibration.json (rebuild the binary to embed them: cargo build --release)"
