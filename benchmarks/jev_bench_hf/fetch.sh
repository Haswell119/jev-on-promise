#!/usr/bin/env bash
# Fetch the jev-bench test splits (Praveenrajus/jev-bench on the Hugging Face Hub) at a
# pinned dataset revision into benchmarks/jev_bench_hf/data/<config>/test.jsonl
# (gitignored), verify them against the committed checksums.txt, and install the
# Jevify harness (Apache-2.0) at its pinned commit into benchmarks/jev_bench_hf/harness/
# (gitignored) inside the virtualenv benchmarks/.venv.
#
#   benchmarks/jev_bench_hf/fetch.sh                    download + verify + install harness
#   benchmarks/jev_bench_hf/fetch.sh --check            verify existing files only (no network)
#   benchmarks/jev_bench_hf/fetch.sh --data-only        skip the harness / virtualenv
#   benchmarks/jev_bench_hf/fetch.sh --write-checksums  (maintainers) regenerate checksums.txt
#                                                       after changing DATASET_REVISION
#
# Only the 22 test.jsonl files (about 30 MB) are downloaded, never train/validation.
# Records carry per-source upstream licenses; see the dataset's manifest.json (also
# downloaded) and https://github.com/uspraveen/Jevify/blob/main/docs/DATASETS.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$HERE/data"
HARNESS_DIR="${JEVIFY_HARNESS_DIR:-$HERE/harness}"
VENV="${BENCH_VENV:-$HERE/../.venv}"
CHECKSUMS="$HERE/checksums.txt"

DATASET="Praveenrajus/jev-bench"
DATASET_REVISION="002ad22de8db2df5e0eb898b3da8072dbd4af4de"   # jev-bench v0.1.1
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}/datasets/$DATASET/resolve/$DATASET_REVISION"

JEVIFY_REPO="https://github.com/uspraveen/Jevify"
JEVIFY_COMMIT="2891025b8a4520d0eb639f2cd1b3fddaf4b922b3"

# The 22 configs of the dataset card (22,773 test records in total).
CONFIGS=(
  banking77 clinc150 massive ledgar go_emotions mmlu arc_challenge mnli chaosnli
  sst5 yelp5 helpsteer2_helpfulness helpsteer2_verbosity stsb measuring_hate_speech
  boolq fever_evidence paws civil_comments sms_spam strategyqa_closed strategyqa_grounded
)

check_only=0
data_only=0
write_checksums=0
for arg in "$@"; do
  case "$arg" in
    --check) check_only=1 ;;
    --data-only) data_only=1 ;;
    --write-checksums) write_checksums=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }

expected_sha() {  # expected_sha <relative path> -> sha or empty
  [ -f "$CHECKSUMS" ] || return 0
  awk -v p="$1" '$2 == p { print $1 }' "$CHECKSUMS"
}

# Local paths are relative to this directory (data/<config>/test.jsonl, data/manifest.json);
# on the Hub the test files live under data/ and manifest.json at the repository root.
remote_of() {
  case "$1" in
    data/manifest.json) echo "manifest.json" ;;
    *) echo "$1" ;;
  esac
}

download() {  # download <local relative path>
  local rel="$1" url="$HF_BASE/$(remote_of "$1")" dest="$HERE/$1"
  mkdir -p "$(dirname "$dest")"
  echo "[fetch] $url"
  curl -fsSL --retry 3 --retry-delay 2 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

files=(data/manifest.json)
for cfg in "${CONFIGS[@]}"; do
  files+=("data/$cfg/test.jsonl")
done

if [ "$check_only" -eq 0 ]; then
  for rel in "${files[@]}"; do
    dest="$HERE/$rel"
    want="$(expected_sha "$rel")"
    if [ -f "$dest" ] && [ -n "$want" ] && [ "$(sha_of "$dest")" = "$want" ]; then
      continue  # already present and verified
    fi
    download "$rel"
  done
fi

if [ "$write_checksums" -eq 1 ]; then
  {
    echo "# sha256 of $DATASET at revision $DATASET_REVISION; paths relative to benchmarks/jev_bench_hf"
    echo "# (verify: cd benchmarks/jev_bench_hf && sha256sum -c checksums.txt)"
    for rel in "${files[@]}"; do
      printf '%s  %s\n' "$(sha_of "$HERE/$rel")" "$rel"
    done
  } >"$CHECKSUMS"
  echo "[fetch] wrote $CHECKSUMS"
fi

[ -f "$CHECKSUMS" ] || die "no $CHECKSUMS to verify against (maintainers: --write-checksums)"
missing=0
for rel in "${files[@]}"; do
  [ -f "$HERE/$rel" ] || { echo "missing: $rel" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || die "dataset files missing (run without --check)"
(cd "$HERE" && sha256sum --quiet -c <(grep -v '^#' "$CHECKSUMS")) || die "checksum mismatch; see $CHECKSUMS"
total=0
for cfg in "${CONFIGS[@]}"; do
  total=$((total + $(wc -l <"$DATA_DIR/$cfg/test.jsonl")))
done
echo "[fetch] dataset ok: ${#CONFIGS[@]} configs, $total test records, revision $DATASET_REVISION"

[ "$data_only" -eq 0 ] || exit 0

if [ "$check_only" -eq 0 ]; then
  if [ ! -d "$HARNESS_DIR/.git" ]; then
    echo "[fetch] cloning $JEVIFY_REPO into $HARNESS_DIR"
    git init -q "$HARNESS_DIR"
    git -C "$HARNESS_DIR" remote add origin "$JEVIFY_REPO"
  fi
  if [ "$(git -C "$HARNESS_DIR" rev-parse HEAD 2>/dev/null || true)" != "$JEVIFY_COMMIT" ]; then
    echo "[fetch] fetching pinned commit $JEVIFY_COMMIT"
    git -C "$HARNESS_DIR" fetch -q --depth 1 origin "$JEVIFY_COMMIT" \
      || git -C "$HARNESS_DIR" fetch -q origin
    git -C "$HARNESS_DIR" checkout -q --detach "$JEVIFY_COMMIT"
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    echo "[fetch] creating virtualenv $VENV"
    python3 -m venv "$VENV"
  fi
  # `pip install -e harness` brings the runner's real dependencies (httpx, numpy,
  # pydantic). The package's __init__ also imports its torch-backed engine
  # (jevify/__init__.py -> load.py -> engine/readout.py), so `jevify-run` cannot start
  # without torch even though the HTTP runner never uses it; a CPU-only build from the
  # PyTorch index keeps that to ~200 MB. Set JEVIFY_SKIP_TORCH=1 if torch is already there.
  "$VENV/bin/pip" install --quiet -e "$HARNESS_DIR"
  if [ -z "${JEVIFY_SKIP_TORCH:-}" ] && ! "$VENV/bin/python" -c 'import torch' 2>/dev/null; then
    echo "[fetch] installing CPU-only torch (harness import dependency)"
    "$VENV/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

[ -d "$HARNESS_DIR/.git" ] || die "no harness checkout at $HARNESS_DIR (run without --check)"
head="$(git -C "$HARNESS_DIR" rev-parse HEAD)"
[ "$head" = "$JEVIFY_COMMIT" ] || die "harness is at $head, expected $JEVIFY_COMMIT"
[ -x "$VENV/bin/jevify-run" ] || die "jevify-run not installed in $VENV (run without --check)"
"$VENV/bin/python" -c 'import jevify.runners.cli, jevify.bench.metrics' || die "harness import failed"
echo "[fetch] harness ok: Jevify @ $head, $("$VENV/bin/python" --version), $VENV/bin/jevify-run"
