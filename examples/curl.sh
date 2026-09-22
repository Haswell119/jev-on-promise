#!/usr/bin/env bash
# Three calls against a running Sextant server (start one with `cargo run --release -- serve`
# or `docker compose up`). SEXTANT_URL overrides the base URL.
set -euo pipefail
URL="${SEXTANT_URL:-http://127.0.0.1:8080}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pretty() { if command -v jq >/dev/null 2>&1; then jq .; else cat; echo; fi; }

echo "# 1. basic: three typed questions about one state (examples/request_basic.json)"
curl -sS -X POST "$URL/v1/systemone" \
  -H 'Content-Type: application/json' \
  --data-binary @"$HERE/request_basic.json" | pretty

echo
echo "# 2. explain=true attaches an inspection block to every answer (inference is unchanged)"
curl -sS -X POST "$URL/v1/systemone?explain=true" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "sextant-1",
        "state": {"subject": "Invoice 4821 charged twice", "body": "Please refund the duplicate charge."},
        "questions": {
          "refund_request": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
        }
      }' | pretty

echo
echo "# 3. error case: a Choice question needs at least two options -> 422 with a structured error body"
curl -sS -w '\nHTTP %{http_code}\n' -X POST "$URL/v1/systemone" \
  -H 'Content-Type: application/json' \
  -d '{"model": "sextant-1", "state": "hi", "questions": {"q": {"type": "choice", "criteria": {"only": "one option"}}}}'
