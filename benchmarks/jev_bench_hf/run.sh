#!/usr/bin/env bash
# Run jev-bench test splits through Jevify's own HTTP runner (`jevify-run api`) against a
# local Sextant server, score them with the harness's own metrics (`jevify-run report`),
# and add macro / per-primitive rows with summarize.py.
#
#   benchmarks/jev_bench_hf/run.sh                          all 22 configs (22,773 records)
#   benchmarks/jev_bench_hf/run.sh --sources boolq,sst5     subset of configs
#   benchmarks/jev_bench_hf/run.sh --limit 20               first 20 records per config (smoke)
#   benchmarks/jev_bench_hf/run.sh --concurrency 4          parallel requests (default 4)
#   benchmarks/jev_bench_hf/run.sh --label "..."            label printed at the top of summary.md
#   benchmarks/jev_bench_hf/run.sh --out DIR                output directory (must not exist yet)
#   SEXTANT_ENDPOINT=http://127.0.0.1:8080 ...              use a running server instead of starting one
#
# Output: reports/jev_bench_hf/<git-short-sha>-<YYYYMMDD>/
#   preds.jsonl       harness predictions (one per record, with latency_ms)
#   report.md         harness per-config table      metrics.json  harness per-config metrics + reliability bins
#   summary.md / summary.json   summarize.py: label, macro and per-primitive rows, errors, latency
#   manifest.json     settings, dataset revision + checksums, harness commit    environment.json    server.log
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
source "$HERE/../lib.sh"
ROOT="$(bench_root)"
HARNESS_DIR="${JEVIFY_HARNESS_DIR:-$HERE/harness}"
VENV="${BENCH_VENV:-$HERE/../.venv}"
MODEL="${SEXTANT_MODEL:-sextant-1}"

sources=""
limit=""
concurrency=4
label=""
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sources) sources="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --concurrency) concurrency="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) bench_die "unknown argument: $1" ;;
  esac
done

"$HERE/fetch.sh" --check >/dev/null || bench_die "run benchmarks/jev_bench_hf/fetch.sh first"
jevify_run="$VENV/bin/jevify-run"

run_id="$(bench_run_id)"
if [ -z "$out" ]; then
  out="$(bench_fresh_dir "$ROOT/reports/jev_bench_hf/$run_id")"
fi
[ ! -e "$out" ] || bench_die "output directory exists: $out"
mkdir -p "$out"
echo "[jev-bench] run $run_id -> $out"

trap bench_stop_server EXIT
bench_start_server "$out/server.log"

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# The runner sends "Authorization: Bearer $TYPESAFE_API_KEY" (empty here); Sextant
# ignores it. --records is a root holding data/<source>/test.jsonl.
TYPESAFE_API_KEY="" "$jevify_run" api \
  --records "$HERE" \
  --out "$out/preds.jsonl" \
  --split test \
  --model "$MODEL" \
  --base-url "$BENCH_ENDPOINT" \
  --concurrency "$concurrency" \
  ${sources:+--sources "$sources"} \
  ${limit:+--limit "$limit"}
bench_stop_server
trap - EXIT

"$jevify_run" report \
  --records "$HERE" \
  --preds "$out/preds.jsonl" \
  --split test \
  --md "$out/report.md" \
  --json "$out/metrics.json" >/dev/null

harness_commit="$(git -C "$HARNESS_DIR" rev-parse HEAD)"
"$VENV/bin/python" - "$out" "$run_id" "$sources" "$limit" "$concurrency" "$label" "$MODEL" "$BENCH_ENDPOINT" "$harness_commit" "$HERE/checksums.txt" "$started" <<'PY'
import datetime, json, os, sys
out, run_id, sources, limit, concurrency, label, model, endpoint, harness_commit, checksums, started = sys.argv[1:]
sha = {}
with open(checksums, encoding="utf-8") as fh:
    for line in fh:
        if line.strip() and not line.startswith("#"):
            digest, path = line.split()
            sha[path] = digest
doc = {
    "benchmark": "jev-bench (Praveenrajus/jev-bench) test splits",
    "dataset": "Praveenrajus/jev-bench",
    "dataset_revision": "002ad22de8db2df5e0eb898b3da8072dbd4af4de",
    "dataset_sha256": sha,
    "harness_repo": "https://github.com/uspraveen/Jevify",
    "harness_commit": harness_commit,
    "harness_license": "Apache-2.0",
    "runner": "jevify-run api (SystemOneAPIRunner) + jevify-run report, unmodified",
    "run_id": run_id,
    "label": label,
    "model": model,
    "endpoint": endpoint,
    "sources": sources.split(",") if sources else "all",
    "limit_per_source": int(limit) if limit else None,
    "concurrency": int(concurrency),
    "started_utc": started,
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
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
  "harness=jevify" \
  "harness_commit=$harness_commit" \
  "dataset_revision=002ad22de8db2df5e0eb898b3da8072dbd4af4de" \
  "python=$("$VENV/bin/python" --version 2>&1 | cut -d' ' -f2)" \
  "concurrency=$concurrency"

"$VENV/bin/python" "$HERE/summarize.py" \
  --metrics "$out/metrics.json" \
  --preds "$out/preds.jsonl" \
  --dataset-manifest "$HERE/data/manifest.json" \
  --manifest "$out/manifest.json" \
  --environment "$out/environment.json" \
  ${label:+--label "$label"} \
  --md "$out/summary.md" \
  --json "$out/summary.json"

echo "[jev-bench] harness report: $out/report.md"
echo "[jev-bench] summary:        $out/summary.md"
