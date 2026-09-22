#!/usr/bin/env bash
# Fetch the JevBench harness (Benchmark Heaven, MIT) at its pinned commit into
# benchmarks/jevbench/harness/ (gitignored) and verify the three public task files.
#
#   benchmarks/jevbench/fetch.sh            clone or update to the pinned commit, then verify
#   benchmarks/jevbench/fetch.sh --check    verify an existing checkout only (no network)
#
# The harness's HTTP adapter needs only the Python 3.10+ standard library, so nothing
# is installed: run.sh puts the checkout on PYTHONPATH. Set JEVBENCH_HARNESS_DIR to
# use a checkout kept elsewhere.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="${JEVBENCH_HARNESS_DIR:-$HERE/harness}"

JEVBENCH_REPO="https://github.com/fstandhartinger/jevbench"
# v1.3.0 "score intelligence above chance" (2026-09-21); public task files frozen 2026-09-19.
JEVBENCH_COMMIT="75e6224ed8103bbc3485ca74820a2eaf7ce8abe0"

# sha256 of datasets/public/*.jsonl at that commit (also listed in the harness's
# datasets/manifest.json). A mismatch means the tasks are not the frozen ones.
EXPECTED_EASY="231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"
EXPECTED_ORIGINAL="5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"
EXPECTED_HARD="89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"

check_only=0
for arg in "$@"; do
  case "$arg" in
    --check) check_only=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }

if [ "$check_only" -eq 0 ]; then
  if [ ! -d "$HARNESS_DIR/.git" ]; then
    echo "[fetch] cloning $JEVBENCH_REPO into $HARNESS_DIR"
    git init -q "$HARNESS_DIR"
    git -C "$HARNESS_DIR" remote add origin "$JEVBENCH_REPO"
  fi
  if [ "$(git -C "$HARNESS_DIR" rev-parse HEAD 2>/dev/null || true)" != "$JEVBENCH_COMMIT" ]; then
    echo "[fetch] fetching pinned commit $JEVBENCH_COMMIT"
    # GitHub serves single commits by hash; fall back to a full fetch for other hosts.
    git -C "$HARNESS_DIR" fetch -q --depth 1 origin "$JEVBENCH_COMMIT" \
      || git -C "$HARNESS_DIR" fetch -q origin
    git -C "$HARNESS_DIR" checkout -q --detach "$JEVBENCH_COMMIT"
  fi
fi

[ -d "$HARNESS_DIR/.git" ] || die "no harness checkout at $HARNESS_DIR (run without --check)"
head="$(git -C "$HARNESS_DIR" rev-parse HEAD)"
[ "$head" = "$JEVBENCH_COMMIT" ] || die "harness is at $head, expected $JEVBENCH_COMMIT"
echo "[fetch] harness commit $head (ok)"

sha_of() { sha256sum "$1" | cut -d' ' -f1; }
verify() {
  local name="$1" expected="$2" path="$HARNESS_DIR/datasets/public/$1" actual
  [ -f "$path" ] || die "missing $path"
  actual="$(sha_of "$path")"
  printf '%s  %s (%s lines)\n' "$actual" "datasets/public/$name" "$(wc -l <"$path")"
  [ "$actual" = "$expected" ] || die "sha256 mismatch for $name: expected $expected"
}
verify easy.jsonl "$EXPECTED_EASY"
verify original.jsonl "$EXPECTED_ORIGINAL"
verify hard.jsonl "$EXPECTED_HARD"

# The harness import must work with the plain interpreter: no third-party packages.
py="$(command -v python3 || true)"
[ -n "$py" ] || die "python3 not found (the harness needs Python 3.10+)"
PYTHONPATH="$HARNESS_DIR" "$py" - <<'PY' || die "harness import failed"
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"python {sys.version.split()[0]} is too old; JevBench needs 3.10+")
import jevbench.cli  # noqa: F401  (stdlib only for the typesafe adapter)
print(f"[fetch] python {sys.version.split()[0]}: jevbench.cli imports (ok)")
PY
echo "[fetch] ready: $HARNESS_DIR"
