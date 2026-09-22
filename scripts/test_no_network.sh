#!/usr/bin/env bash
#
# Proves that Sextant inference works with networking completely disabled:
# the CLI (`sextant decide`) and the HTTP server (`sextant serve`) are run
# inside a sandbox that has no network interfaces at all, an outbound TCP
# connection is shown to fail there, and both entry points still answer
# examples/request_basic.json.
#
# Mechanisms, tried in this order:
#   1. docker   containers started with `--network none`. When the repo root has
#               a Dockerfile the image is built from it and exercised directly
#               (its runtime image ships curl); otherwise the release binary is
#               bind-mounted into python:3.12-slim.
#   2. unshare  `unshare -rn`: a fresh user + network namespace on this host.
#   3. neither  print SKIPPED with the reason and exit 0.
#
# Usage: scripts/test_no_network.sh
#   SEXTANT_NONET_MECHANISM=auto|docker|unshare   force a mechanism (default auto)
#   SEXTANT_TEST_PORT=18080                       loopback port used by the server check
#   SEXTANT_REBUILD=1                             rebuild target/release/sextant first
#
# Exit status: 0 on PASS or SKIPPED, 1 on FAIL. The last line always starts
# with PASS, FAIL or SKIPPED.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/target/release/sextant"
EXAMPLE="$REPO/examples/request_basic.json"
PORT="${SEXTANT_TEST_PORT:-18080}"
MECH="${SEXTANT_NONET_MECHANISM:-auto}"
IMAGE_TAG="sextant-no-network-test:local"

log() { printf '[no-network] %s\n' "$*"; }
pass() { printf 'PASS: %s\n' "$*"; exit 0; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }
skip() { printf 'SKIPPED: %s\n' "$*"; exit 0; }

[ -f "$EXAMPLE" ] || fail "missing $EXAMPLE"
if [ ! -x "$BIN" ] || [ "${SEXTANT_REBUILD:-0}" = "1" ]; then
    command -v cargo >/dev/null 2>&1 || fail "no release binary at $BIN and cargo is not installed"
    log "building the release binary (cargo build --release -p sextant)"
    (cd "$REPO" && cargo build --release -p sextant) || fail "cargo build --release failed"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ----------------------------------------------------------------------------
# Inner script: runs entirely INSIDE the network-less sandbox.
#   args: <sextant binary> <request.json> <port>
# ----------------------------------------------------------------------------
INNER="$WORK/inner.sh"
cat > "$INNER" <<'INNER_EOF'
#!/usr/bin/env bash
set -uo pipefail
SX="$1"; REQ="$2"; PORT="$3"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export no_proxy='*' NO_PROXY='*'
say() { printf '[no-network:inner] %s\n' "$*"; }
die() { say "FAIL: $*"; exit 1; }
PY="$(command -v python3 || true)"

# 1. Prove the outside world is unreachable (direct IP: no DNS involved).
if [ -n "$PY" ]; then
    "$PY" - <<'PY' || die "an outbound TCP connection to 1.1.1.1:80 SUCCEEDED: networking is not disabled"
import socket, sys
try:
    socket.create_connection(("1.1.1.1", 80), timeout=3)
except OSError as e:
    print(f"[no-network:inner] outbound connection blocked as expected: {e}")
    sys.exit(0)
sys.exit(1)
PY
else
    say "python3 is not available in the sandbox; skipping the outbound-connection proof"
fi

# 2. Offline inference through the CLI.
OUT="$("$SX" decide "$REQ" 2>/dev/null)" || die "'sextant decide' exited non-zero"
case "$OUT" in
    *'"answers"'*) say "CLI decide answered offline: ${OUT:0:120}..." ;;
    *) die "'sextant decide' output has no \"answers\": $OUT" ;;
esac

# 3. Offline HTTP server on loopback. A fresh namespace starts with `lo` DOWN;
#    bring it up with iproute2 when present, otherwise with a plain ioctl.
if command -v ip >/dev/null 2>&1; then
    ip link set lo up 2>/dev/null || true
elif [ -n "$PY" ]; then
    "$PY" - <<'PY' || say "could not bring loopback up; the server check may fail"
import fcntl, socket, struct
SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ifr = struct.pack("16sH14s", b"lo", 0, b"\0" * 14)
flags = struct.unpack("16sH14s", fcntl.ioctl(s.fileno(), SIOCGIFFLAGS, ifr))[1]
if not flags & IFF_UP:
    fcntl.ioctl(s.fileno(), SIOCSIFFLAGS, struct.pack("16sH14s", b"lo", flags | IFF_UP, b"\0" * 14))
print("[no-network:inner] loopback interface is up")
PY
fi

if [ -z "$PY" ] && ! command -v curl >/dev/null 2>&1; then
    say "neither python3 nor curl is available in the sandbox; HTTP server check skipped"
    say "all available checks passed"
    exit 0
fi

"$SX" serve --addr "127.0.0.1:$PORT" >/dev/null 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT

if [ -n "$PY" ]; then
    "$PY" - "$PORT" "$REQ" <<'PY' || die "HTTP server check failed"
import json, sys, time, urllib.request
port, req_path = int(sys.argv[1]), sys.argv[2]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = f"http://127.0.0.1:{port}"
health = None
for _ in range(150):
    try:
        health = json.load(opener.open(base + "/health", timeout=2))
        break
    except Exception:
        time.sleep(0.2)
if not health or health.get("status") != "ok":
    print("[no-network:inner] the server never became healthy"); sys.exit(1)
if health.get("neural") is not False or health.get("network_required") is not False:
    print("[no-network:inner] unexpected health payload:", health); sys.exit(1)
body = open(req_path, "rb").read()
req = urllib.request.Request(base + "/v1/systemone", data=body, headers={"content-type": "application/json"})
resp = json.load(opener.open(req, timeout=60))
if "answers" not in resp:
    print("[no-network:inner] no answers in the HTTP response:", resp); sys.exit(1)
print("[no-network:inner] HTTP /v1/systemone answered offline:",
      ", ".join(f"{k}={v['type']}" for k, v in resp["answers"].items()))
PY
else
    ok=0
    for _ in $(seq 1 150); do
        curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        sleep 0.2
    done
    [ "$ok" = 1 ] || die "the server never became healthy"
    RESP="$(curl -fsS --max-time 60 -H 'content-type: application/json' --data-binary "@$REQ" "http://127.0.0.1:$PORT/v1/systemone")" \
        || die "POST /v1/systemone failed"
    case "$RESP" in
        *'"answers"'*) say "HTTP /v1/systemone answered offline" ;;
        *) die "no answers in the HTTP response: $RESP" ;;
    esac
fi
say "all checks passed"
INNER_EOF
chmod +x "$INNER"

# ----------------------------------------------------------------------------
# Mechanism selection
# ----------------------------------------------------------------------------
docker_usable() {
    command -v docker >/dev/null 2>&1 || return 1
    docker info >/dev/null 2>&1 || return 1
    return 0
}

unshare_usable() {
    command -v unshare >/dev/null 2>&1 || return 1
    unshare -rn true >/dev/null 2>&1 || return 1
    return 0
}

# Docker path A: the repo's own Dockerfile (entrypoint = the binary, curl inside).
run_docker_image() {
    log "building $IMAGE_TAG from $REPO/Dockerfile"
    if ! docker build -q -t "$IMAGE_TAG" "$REPO" > "$WORK/build.log" 2>&1; then
        tail -n 3 "$WORK/build.log" | sed 's/^/[no-network]   docker build: /'
        return 1
    fi

    log "docker run --network none $IMAGE_TAG decide < examples/request_basic.json"
    local out
    out="$(docker run --rm --network none -i "$IMAGE_TAG" decide < "$EXAMPLE" 2>/dev/null)" \
        || fail "'decide' failed inside the image with --network none"
    case "$out" in
        *'"answers"'*) log "CLI decide answered offline inside the image" ;;
        *) fail "'decide' output inside the image has no \"answers\": $out" ;;
    esac

    log "docker run -d --network none $IMAGE_TAG serve --addr 127.0.0.1:$PORT"
    local cid
    cid="$(docker run -d --rm --network none "$IMAGE_TAG" serve --addr "127.0.0.1:$PORT")" || fail "could not start the server container"
    # shellcheck disable=SC2064
    trap "docker rm -f '$cid' >/dev/null 2>&1; rm -rf '$WORK'" EXIT

    # Outbound proof from inside the same network namespace.
    if docker exec "$cid" curl -sS --max-time 3 http://1.1.1.1/ >/dev/null 2>&1; then
        fail "an outbound HTTP request SUCCEEDED inside the container: networking is not disabled"
    fi
    log "outbound HTTP blocked inside the container as expected"

    local ok=0 i
    for i in $(seq 1 100); do
        docker exec "$cid" curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        sleep 0.2
    done
    [ "$ok" = 1 ] || fail "the server inside the container never became healthy"
    local resp
    resp="$(docker exec -i "$cid" curl -fsS --max-time 60 -H 'content-type: application/json' --data-binary @- "http://127.0.0.1:$PORT/v1/systemone" < "$EXAMPLE")" \
        || fail "POST /v1/systemone failed inside the container"
    case "$resp" in
        *'"answers"'*) log "HTTP /v1/systemone answered offline inside the container" ;;
        *) fail "no answers in the HTTP response inside the container: $resp" ;;
    esac
    docker rm -f "$cid" >/dev/null 2>&1 || true
    trap 'rm -rf "$WORK"' EXIT
    return 0
}

# Docker path B: stock python image with the release binary bind-mounted.
run_docker_mounted() {
    log "docker run --network none python:3.12-slim with $BIN bind-mounted"
    docker run --rm --network none \
        -v "$BIN:/sextant:ro" -v "$EXAMPLE:/request.json:ro" -v "$INNER:/inner.sh:ro" \
        python:3.12-slim bash /inner.sh /sextant /request.json "$PORT"
}

run_docker() {
    if [ -f "$REPO/Dockerfile" ]; then
        if run_docker_image; then
            return 0
        fi
        log "could not build the image from the Dockerfile (offline builder?); falling back to a bind-mounted binary"
    fi
    run_docker_mounted || fail "offline checks failed inside docker (--network none)"
}

run_unshare() {
    log "unshare -rn (fresh user + network namespace) running $BIN"
    unshare -rn bash "$INNER" "$BIN" "$EXAMPLE" "$PORT" || fail "offline checks failed inside 'unshare -rn'"
}

case "$MECH" in
    auto)
        if docker_usable; then
            MECH=docker
        elif unshare_usable; then
            MECH=unshare
        else
            skip "no network-less sandbox available: docker daemon not reachable and 'unshare -rn' not permitted on this host"
        fi
        ;;
    docker)  docker_usable  || skip "SEXTANT_NONET_MECHANISM=docker but the docker daemon is not reachable" ;;
    unshare) unshare_usable || skip "SEXTANT_NONET_MECHANISM=unshare but 'unshare -rn' is not permitted here" ;;
    *) fail "unknown SEXTANT_NONET_MECHANISM='$MECH' (use auto, docker or unshare)" ;;
esac

log "mechanism: $MECH"
case "$MECH" in
    docker)  run_docker ;;
    unshare) run_unshare ;;
esac
pass "Sextant inference (CLI decide + HTTP serve) works with networking disabled (mechanism: $MECH)"
