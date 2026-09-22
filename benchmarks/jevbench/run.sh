#!/usr/bin/env bash
# Run JevBench's public tiers against a local Sextant server using the harness's own
# `typesafe` adapter (native /v1/systemone wire format) and the harness's own scoring,
# then write tier/family/calibration tables with summarize.py.
#
#   benchmarks/jevbench/run.sh                         all three public tiers (231 decisions)
#   benchmarks/jevbench/run.sh --tiers easy,hard       a subset of tiers
#   benchmarks/jevbench/run.sh --limit 5               first 5 tasks of each tier (smoke)
#   benchmarks/jevbench/run.sh --label "..."           label printed at the top of report.md
#   benchmarks/jevbench/run.sh --out DIR               output directory (must not exist yet)
#   SEXTANT_ENDPOINT=http://127.0.0.1:8080 ...         use a running server instead of starting one
#   SEXTANT_BIN=path/to/sextant ...                    binary to serve (default target/release/sextant)
#
# Tiers: easy = datasets/public/easy.jsonl (48 of 72), standard = original.jsonl (72 of 96),
# hard = hard.jsonl (111 of 220). The judge tier (146 imported decisions) is not public.
#
# Output: reports/jevbench/<git-short-sha>-<YYYYMMDD>/
#   results.jsonl          per-item harness records, all tiers   results.<tier>.jsonl  per tier
#   summary.json           harness public export, all tiers      summary.<tier>.json   per tier
#   manifest.json          this run's settings + harness manifests per tier
#   report.md, metrics.json   summarize.py tables (accuracy, schema validity, Brier, ECE, ...)
#   environment.json       CPU / cores / RAM / rustc / commits / dataset hashes
#   ledger.jsonl           harness budget ledger (all zero: local server, no tariff)
#   raw/                   raw request + response per item (gitignored)   server.log
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
source "$HERE/../lib.sh"
ROOT="$(bench_root)"
HARNESS_DIR="${JEVBENCH_HARNESS_DIR:-$HERE/harness}"
MODEL="${SEXTANT_MODEL:-sextant-1}"

tiers="easy,original,hard"
limit=""
label=""
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tiers) tiers="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) bench_die "unknown argument: $1" ;;
  esac
done

[ -f "$HARNESS_DIR/jevbench/cli.py" ] || bench_die "harness not fetched; run benchmarks/jevbench/fetch.sh"
"$HERE/fetch.sh" --check >/dev/null

run_id="$(bench_run_id)"
if [ -z "$out" ]; then
  out="$(bench_fresh_dir "$ROOT/reports/jevbench/$run_id")"
fi
[ ! -e "$out" ] || bench_die "output directory exists: $out (the harness never overwrites a run)"
mkdir -p "$out/raw"
echo "[jevbench] run $run_id -> $out"

trap bench_stop_server EXIT
bench_start_server "$out/server.log"

py="$(command -v python3)"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
status=0
IFS=',' read -r -a tier_list <<<"$tiers"
for tier in "${tier_list[@]}"; do
  tasks="$HARNESS_DIR/datasets/public/$tier.jsonl"
  [ -f "$tasks" ] || bench_die "unknown tier '$tier' (expected easy, original or hard)"
  echo "[jevbench] tier $tier: $(wc -l <"$tasks") public tasks"
  # --key-env '' sends no Authorization header; --reserve-usd 0 / --cost-basis local: a
  # local CPU server has no tariff, so the harness reports cost as null, basis "local".
  "$py" -m jevbench.cli run \
    --tasks "$tasks" \
    --adapter typesafe \
    --endpoint "$BENCH_ENDPOINT" \
    --key-env '' \
    --model "$MODEL" \
    --run-label "sextant-$run_id" \
    --cost-basis local \
    --reserve-usd 0 \
    ${limit:+--limit "$limit"} \
    --results "$out/results.$tier.jsonl" \
    --raw-dir "$out/raw" \
    --ledger "$out/ledger.jsonl" \
    --manifest "$out/manifest.$tier.json" || status=$?
  # exit 3 = some tasks unattempted (the harness stops after 3 consecutive infrastructure errors)
  [ "$status" -eq 0 ] || echo "[jevbench] harness exit $status on tier $tier" >&2
  "$py" -m jevbench.cli summarize \
    --tasks "$tasks" \
    --results "$out/results.$tier.jsonl" \
    --ledger "$out/ledger.jsonl" \
    --public-export "$out/summary.$tier.json" >/dev/null
done
bench_stop_server
trap - EXIT

# Pooled files over the tiers that ran (task ids are unique across tiers).
all_tasks=""
: >"$out/results.jsonl"
for tier in "${tier_list[@]}"; do
  cat "$out/results.$tier.jsonl" >>"$out/results.jsonl"
  all_tasks="${all_tasks:+$all_tasks,}$HARNESS_DIR/datasets/public/$tier.jsonl"
done
"$py" -m jevbench.cli summarize \
  --tasks "$all_tasks" \
  --results "$out/results.jsonl" \
  --ledger "$out/ledger.jsonl" \
  --public-export "$out/summary.json" >/dev/null

harness_commit="$(git -C "$HARNESS_DIR" rev-parse HEAD)"
dataset_hashes="$(cd "$HARNESS_DIR/datasets/public" && "$py" -c '
import hashlib, json, sys
print(json.dumps({t + ".jsonl": hashlib.sha256(open(t + ".jsonl", "rb").read()).hexdigest() for t in sys.argv[1:]}))
' "${tier_list[@]}")"

"$py" - "$out" "$run_id" "$tiers" "$limit" "$label" "$MODEL" "$BENCH_ENDPOINT" "$harness_commit" "$dataset_hashes" "$started" <<'PY'
import datetime, json, os, sys
out, run_id, tiers, limit, label, model, endpoint, harness_commit, hashes, started = sys.argv[1:]
manifests = {}
for tier in tiers.split(","):
    p = os.path.join(out, f"manifest.{tier}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            manifests[tier] = json.load(fh)
doc = {
    "benchmark": "JevBench (Benchmark Heaven) public tiers",
    "harness_repo": "https://github.com/fstandhartinger/jevbench",
    "harness_commit": harness_commit,
    "harness_license": "MIT",
    "adapter": "typesafe",
    "scoring": "harness (jevbench.scoring / jevbench.summarize), unmodified",
    "run_id": run_id,
    "label": label,
    "model": model,
    "endpoint": endpoint,
    "tiers": tiers.split(","),
    "limit_per_tier": int(limit) if limit else None,
    "dataset_sha256": json.loads(hashes),
    "started_utc": started,
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "harness_manifests": manifests,
}
with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

bin="${SEXTANT_BIN:-$ROOT/target/release/sextant}"
bench_environment_json "$out/environment.json" \
  "sextant_version=$("$bin" --version 2>/dev/null || echo unknown)" \
  "sextant_binary=$bin" \
  "server_endpoint=$BENCH_ENDPOINT" \
  "harness=jevbench" \
  "harness_commit=$harness_commit" \
  "dataset_sha256=json:$dataset_hashes" \
  "python=$("$py" --version 2>&1 | cut -d' ' -f2)"

"$py" "$HERE/summarize.py" \
  --tasks-dir "$HARNESS_DIR/datasets/public" \
  --tiers "$tiers" \
  --results "$out/results.jsonl" \
  --manifest "$out/manifest.json" \
  --environment "$out/environment.json" \
  ${label:+--label "$label"} \
  --md "$out/report.md" \
  --json "$out/metrics.json"

echo "[jevbench] report: $out/report.md"
exit "$status"
