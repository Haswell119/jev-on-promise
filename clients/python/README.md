# Sextant Python client

`sextant_client.py` is a single-file, dependency-free client (standard library `urllib`,
Python 3.8+) for the Sextant HTTP API (`POST /v1/systemone`, `GET /health`,
`GET /v1/models`). Copy the file into your project or add this directory to `PYTHONPATH`.

```python
from sextant_client import SextantClient, SextantError, choice, score, noul

client = SextantClient("http://127.0.0.1:8080")          # timeout=30.0 by default

reply = client.systemone(
    state="Help! My payouts have been failing for 3 days.",
    questions={
        "department": choice("Which team should handle this?", {
            "billing": "Payments, invoicing, refunds, payouts",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        }),
        "is_urgent": noul("Does this convey urgency?"),
        "frustration": score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
    },
    model="sextant-1",      # default
    explain=False,          # True attaches an `explain` block to every answer
)

reply["answers"]["department"]      # {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": 0.88}
reply["answers"]["is_urgent"]       # {"type": "noul", "noul": 0.38}
reply["answers"]["frustration"]     # {"type": "score", "score": 1.38, "legend": {...}, "probabilities": {...}, "confidence": 0.23}
reply["usage"]                      # {"input_tokens": 56, "output_tokens": 12}
```

## Questions

| helper | wire shape | answer |
|---|---|---|
| `choice(instructions, {key: description, ...})` | `{"type": "choice", "criteria": {...}}` (2..255 options) | `choice` (argmax key), `probabilities` over exactly the supplied keys, `confidence` |
| `score(instructions, [level0, level1, ...])` | `{"type": "score", "criteria": [...]}` (2..10 ordered levels) | `score` = Σ index × P(level), `legend`, `probabilities` keyed `"0".."K-1"`, `confidence` |
| `noul(instructions, criteria=None)` | `{"type": "noul"}` (optional `{"true": ..., "false": ...}`) | `noul` = calibrated P(true) |

`state` and every `instructions` / criteria entry may be a string, a JSON object or an
array. Question ids are yours and never influence inference.

## Errors

Every non-2xx response raises `SextantError`, whose `status`, `code`, `message` and
`field` mirror the server's `{"error": {...}}` body (`body` keeps it verbatim):

```python
try:
    client.systemone("hi", {"q": choice("?", {"only": "one option"})})
except SextantError as e:
    print(e.status, e.code, e.field)     # 422 invalid_request questions.q.criteria
```

Connection failures raise `SextantError` with `status == 0` and `code == "connection_error"`.

## Other endpoints

```python
client.health()   # {"status": "ok", "version": "0.1.0", "model": "sextant-1", "neural": False, "network_required": False, ...}
client.models()   # {"object": "list", "models": [{"name": "sextant-1", "aliases": [...], ...}]}
```

`example.py` runs the snippet above against `$SEXTANT_URL` (default `http://127.0.0.1:8080`).
The full wire format is in `docs/openapi.yaml` (also served at `GET /openapi.yaml`).
