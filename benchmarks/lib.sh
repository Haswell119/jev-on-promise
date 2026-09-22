#!/usr/bin/env bash
# Shared helpers for benchmarks/*/run.sh. Source this file; do not execute it.
#
#   bench_root                       absolute path of the repository
#   bench_ensure_binary              prints the release binary path (builds it when missing)
#   bench_run_id                     <git-short-sha>-<YYYYMMDD>
#   bench_fresh_dir BASE             prints BASE, or BASE-<HHMMSS> when BASE already exists
#   bench_start_server LOGFILE       starts `sextant serve` on a free loopback port unless
#                                    SEXTANT_ENDPOINT points at a running server; sets
#                                    BENCH_ENDPOINT (and BENCH_SERVER_PID when it started one)
#   bench_stop_server                stops the server this script started (no-op otherwise)
#   bench_environment_json OUT [k=v ...]
#                                    writes CPU model / cores / RAM / rustc / commits and any
#                                    extra key=value pairs (value `json:<literal>` is parsed)
#   bench_python                     prints the interpreter to use (benchmarks/.venv when present)

bench_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

bench_die() {
  echo "error: $*" >&2
  exit 1
}

bench_python() {
  local root
  root="$(bench_root)"
  if [ -x "$root/benchmarks/.venv/bin/python" ]; then
    echo "$root/benchmarks/.venv/bin/python"
  else
    command -v python3 || bench_die "python3 not found"
  fi
}

bench_ensure_binary() {
  local root bin
  root="$(bench_root)"
  bin="${SEXTANT_BIN:-$root/target/release/sextant}"
  if [ ! -x "$bin" ]; then
    echo "[bench] building release binary ($bin)" >&2
    (cd "$root" && cargo build --release --locked --bin sextant >&2) || bench_die "cargo build failed"
  fi
  [ -x "$bin" ] || bench_die "binary not found: $bin"
  echo "$bin"
}

bench_run_id() {
  local root sha
  root="$(bench_root)"
  sha="$(git -C "$root" rev-parse --short=8 HEAD 2>/dev/null || echo nogit)"
  echo "${sha}-$(date -u +%Y%m%d)"
}

bench_fresh_dir() {
  local base="$1"
  if [ -e "$base" ]; then
    echo "${base}-$(date -u +%H%M%S)"
  else
    echo "$base"
  fi
}

bench_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

BENCH_SERVER_PID=""
BENCH_ENDPOINT=""

bench_start_server() {
  local log="$1" bin port i
  if [ -n "${SEXTANT_ENDPOINT:-}" ]; then
    BENCH_ENDPOINT="${SEXTANT_ENDPOINT%/}"
    curl -fsS --max-time 5 "$BENCH_ENDPOINT/health" >/dev/null \
      || bench_die "no Sextant server answering at $BENCH_ENDPOINT/health"
    echo "[bench] using running server at $BENCH_ENDPOINT" >&2
    return 0
  fi
  bin="$(bench_ensure_binary)"
  port="$(bench_free_port)"
  "$bin" serve --addr "127.0.0.1:$port" >"$log" 2>&1 &
  BENCH_SERVER_PID=$!
  BENCH_ENDPOINT="http://127.0.0.1:$port"
  # The engine loads its embedded artifact in well under a second; allow 60 s anyway.
  for i in $(seq 1 600); do
    if curl -fsS --max-time 2 "$BENCH_ENDPOINT/health" >/dev/null 2>&1; then
      echo "[bench] started $bin serve on $BENCH_ENDPOINT (pid $BENCH_SERVER_PID)" >&2
      return 0
    fi
    if ! kill -0 "$BENCH_SERVER_PID" 2>/dev/null; then
      cat "$log" >&2
      bench_die "server exited before becoming healthy"
    fi
    sleep 0.1
  done
  bench_stop_server
  bench_die "server did not become healthy within 60 s (see $log)"
}

bench_stop_server() {
  if [ -n "${BENCH_SERVER_PID:-}" ]; then
    kill -INT "$BENCH_SERVER_PID" 2>/dev/null || true
    wait "$BENCH_SERVER_PID" 2>/dev/null || true
    echo "[bench] stopped server (pid $BENCH_SERVER_PID)" >&2
    BENCH_SERVER_PID=""
  fi
}

bench_environment_json() {
  local out="$1"
  shift
  BENCH_ROOT="$(bench_root)" python3 - "$out" "$@" <<'PY'
import datetime, json, os, platform, subprocess, sys

out, extra = sys.argv[1], sys.argv[2:]
root = os.environ["BENCH_ROOT"]

def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=root).stdout.strip()
    except Exception:
        return ""

def proc_field(path, prefix):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None

mem_kib = proc_field("/proc/meminfo", "memtotal")
env = {
    "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "platform": platform.platform(),
    "cpu_model": proc_field("/proc/cpuinfo", "model name") or platform.processor() or None,
    "cpu_count": os.cpu_count(),
    "ram_gib": round(int(mem_kib.split()[0]) / 1048576, 1) if mem_kib else None,
    "python": platform.python_version(),
    "rustc": sh("rustc", "--version"),
    "cargo": sh("cargo", "--version"),
    "sextant_commit": sh("git", "rev-parse", "HEAD"),
    "sextant_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
    "sextant_worktree_dirty": bool(sh("git", "status", "--porcelain", "--untracked-files=no")),
}
for item in extra:
    key, _, value = item.partition("=")
    if value.startswith("json:"):
        value = json.loads(value[5:])
    env[key] = value
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(env, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}
