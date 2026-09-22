#!/usr/bin/env bash
# Resumable research loop.
#
#   scripts/research_loop.sh status         show champion, queue, checkpoints
#   scripts/research_loop.sh next           print the next pending hypothesis
#   scripts/research_loop.sh run <id>       train + evaluate + record one queued experiment
#   scripts/research_loop.sh resume         resume the most recent unfinished training run
#
# The queue lives in research/queue.jsonl (one experiment spec per line):
#   {"id":"E3","hypothesis":"...","status":"pending","train_args":"--epochs 1 ..."}
# Nothing is promoted automatically: `run` prints the verdict and records the
# experiment; promotion is an explicit `scripts/experiment.py promote`.
set -euo pipefail
cd "$(dirname "$0")/.."
QUEUE=research/queue.jsonl
mkdir -p research experiments/runs reports/challengers

case "${1:-status}" in
  status)
    python3 scripts/experiment.py show --tail 12
    echo
    echo "== queue"
    [ -f "$QUEUE" ] && python3 - <<'PY' || echo "(empty)"
import json
for l in open("research/queue.jsonl"):
    if l.strip():
        r = json.loads(l)
        print(f"  {r['id']:<8} {r.get('status','pending'):<10} {r['hypothesis'][:80]}")
PY
    echo
    echo "== checkpoints"
    for d in experiments/runs/*/; do
      [ -f "$d/checkpoint.pt" ] || continue
      step=$(python3 -c "import torch,sys;print(torch.load('$d/checkpoint.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo "?")
      echo "  $(basename "$d"): step $step $( [ -f "$d/best_metrics.json" ] && python3 -c "import json;print('best dev acc', round(json.load(open('$d/best_metrics.json'))['accuracy'],4))" )"
    done
    echo
    echo "== datasets"
    for f in data/neural/*.jsonl; do printf "  %-40s %s rows\n" "$f" "$(wc -l < "$f")"; done
    ;;
  next)
    python3 - <<'PY'
import json, os
q = "research/queue.jsonl"
rows = [json.loads(l) for l in open(q)] if os.path.exists(q) else []
nxt = next((r for r in rows if r.get("status", "pending") == "pending"), None)
print(json.dumps(nxt, indent=2) if nxt else "queue empty; add hypotheses from research/BACKLOG.md")
PY
    ;;
  run)
    ID="${2:?usage: research_loop.sh run <experiment_id>}"
    SPEC=$(python3 - "$ID" <<'PY'
import json, sys
rid = sys.argv[1]
for l in open("research/queue.jsonl"):
    if l.strip():
        r = json.loads(l)
        if r["id"] == rid:
            print(json.dumps(r)); break
else:
    raise SystemExit(f"{rid} not in queue")
PY
)
    TRAIN_ARGS=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('train_args',''))" "$SPEC")
    HYP=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['hypothesis'])" "$SPEC")
    mkdir -p "experiments/runs/$ID"
    echo "== training $ID: $HYP"
    # shellcheck disable=SC2086
    python3 scripts/neural/train.py --out "experiments/runs/$ID" --resume $TRAIN_ARGS 2>&1 | tee -a "experiments/runs/$ID/stdout.log" | grep -E "^\[(train|eval)\]" | tail -5
    scripts/evaluate_challenger.sh "$ID"
    python3 - "$ID" <<'PY'
import json, sys
rid = sys.argv[1]
rows = [json.loads(l) for l in open("research/queue.jsonl") if l.strip()]
for r in rows:
    if r["id"] == rid:
        r["status"] = "evaluated"
with open("research/queue.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
PY
    ;;
  resume)
    LAST=$(ls -td experiments/runs/*/ 2>/dev/null | head -1)
    [ -n "$LAST" ] || { echo "no runs to resume"; exit 1; }
    ID=$(basename "$LAST")
    echo "resuming $ID"
    exec "$0" run "$ID"
    ;;
  *)
    echo "usage: $0 {status|next|run <id>|resume}"; exit 2 ;;
esac
