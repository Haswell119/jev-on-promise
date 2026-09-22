#!/usr/bin/env bash
# Evaluate a trained neural run end to end through the Rust engine and score
# it against the champion.
#
#   scripts/evaluate_challenger.sh E1 [--shadow]
#
# Steps: export the run to the Rust artifact layout, fit the fusion block on
# the internal calibration split, run `sextant eval` on the internal dev set
# (or the shadow set at a promotion boundary), compute the weighted dev score
# and print the PROMOTE / REJECT verdict.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ID="${1:?usage: evaluate_challenger.sh <experiment_id> [--shadow]}"
shift || true
EVAL_SET="${SEXTANT_EVAL_SET:-data/bench_internal/dev.jsonl}"
PAIRS_SET="${SEXTANT_EVAL_PAIRS:-data/neural/bench_dev.jsonl}"
CALIB_PAIRS="${SEXTANT_CALIB_PAIRS:-data/neural/bench_calib.jsonl}"
LONGCTX_SET="${SEXTANT_LONGCTX_SET:-data/bench_internal/long_context.jsonl}"
EVIDENCE_WORDS="${SEXTANT_EVIDENCE_WORDS:-140}"
EVIDENCE_CAP="${SEXTANT_EVIDENCE_CAP:-0}"
EVIDENCE_STRATEGY="${SEXTANT_EVIDENCE_STRATEGY:-quota}"
TAG="dev"
for a in "$@"; do
  if [ "$a" = "--shadow" ]; then EVAL_SET="data/bench_internal/shadow.jsonl"; PAIRS_SET="data/neural/bench_shadow.jsonl"; TAG="shadow"; fi
done
RUN="experiments/runs/$RUN_ID"
MODEL="models/$RUN_ID"
BIN=./target/release/sextant
mkdir -p "$MODEL" reports/challengers

echo "== 1/5 export $RUN -> $MODEL/neural"
python3 scripts/neural/export_rust.py --run "$RUN" --out "$MODEL/neural" --version "$RUN_ID" \
  --evidence-words "$EVIDENCE_WORDS" --evidence-adaptive-cap "$EVIDENCE_CAP" --evidence-strategy "$EVIDENCE_STRATEGY" >/dev/null
cp model/weights.json "$MODEL/weights.json"

echo "== 2/5 neural probe on the calibration split"
$BIN neural-probe --neural-dir "$MODEL/neural" --input "$CALIB_PAIRS" --out "reports/challengers/$RUN_ID.calib_probe.jsonl"

echo "== 3/5 fit the fusion block"
python3 scripts/neural/fit_fusion.py --pairs "$CALIB_PAIRS" --probe "reports/challengers/$RUN_ID.calib_probe.jsonl" \
  --base-calibration model/calibration.json --out "$MODEL/calibration.json"

echo "== 4/5 evaluate on the internal $TAG set through the engine"
$BIN --model-dir "$MODEL" eval "$EVAL_SET" --group-by difficulty --out "reports/challengers/$RUN_ID.$TAG.json"
if [ "${SEXTANT_SKIP_LONGCTX:-0}" != "1" ]; then
  $BIN --model-dir "$MODEL" eval "$LONGCTX_SET" --group-by state_bucket --out "reports/challengers/$RUN_ID.longctx.json" | tail -12
fi

echo "== 5/5 dev score and verdict"
python3 scripts/dev_score.py --eval "reports/challengers/$RUN_ID.$TAG.json" --bench reports/bench.json --memory-mb "${SEXTANT_MEM_MB:-900}" --out "reports/challengers/$RUN_ID.score.json" >/dev/null
CHAMP_SCORE=$(python3 -c "import json;print(json.load(open('experiments/CURRENT_CHAMPION.json'))['dev_metrics'] and 1)" 2>/dev/null || echo 1)
python3 - "$RUN_ID" <<'PY'
import json, sys
rid = sys.argv[1]
champ = json.load(open("experiments/CURRENT_CHAMPION.json"))
cs = champ.get("dev_metrics") or {}
ch_file = "reports/dev_score_%s.json" % champ["experiment_id"]
try:
    cs = json.load(open(ch_file))
except Exception:
    pass
json.dump(cs, open("reports/challengers/_champion_score.json", "w"), indent=2)
PY
python3 scripts/experiment.py compare --champion reports/challengers/_champion_score.json --challenger "reports/challengers/$RUN_ID.score.json" || true
