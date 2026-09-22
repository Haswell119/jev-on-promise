# HTTP API

Base URL: `http://localhost:8080` (configurable with `sextant serve --addr`).
The machine-readable contract is `docs/openapi.yaml` (also served at
`GET /openapi.yaml`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness; reports version, model id, `neural: false`, `network_required: false` |
| `GET` | `/v1/models` | Model cards (`sextant-1`, aliases `sextant-latest`, `sextant`) |
| `POST` | `/v1/systemone` | Evaluate typed questions against a state (`?explain=true` for inspection) |
| `GET` | `/openapi.yaml` | OpenAPI 3.1 document |

## Request

```json
{
  "model": "sextant-1",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

* `state`: string, object or array. Objects are flattened with dotted paths
  (`ticket.messages[0].text`); instructions may reference paths in backticks.
* `questions`: caller-chosen ids → questions. Ids are metadata only.
* `instructions`: string, object or array. In an object, question-like fields
  (`question`, `instructions`, `focus`, …, or any string ending in `?`) form
  the question; other fields are reference data that the question may cite in
  backticks ("Does `extracted_value` match the `field` in `source_text`?").
* Choice `criteria`: option key → description (string | object | array |
  null), 2–255 options. Object descriptions keep field semantics: fields whose
  names carry negation morphemes (`not_for`, `excludes`, `negative_examples`,
  …) become negative evidence; example-like fields (`examples`, `phrases`,
  `keywords`, …) are matched individually.
* Score `criteria`: ordered array of 2–10 level descriptions (index 0 =
  lowest).
* Noul `criteria` (optional): `{"true": …, "false": …}`.
* `explain` (optional boolean, or `?explain=true`): attach explanations.

## Response

```json
{
  "model": "sextant-1",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.91, "technical": 0.05, "sales": 0.04 },
      "confidence": 0.86
    },
    "is_urgent": { "type": "noul", "noul": 0.71 },
    "frustration": {
      "type": "score",
      "score": 1.32,
      "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
      "probabilities": { "0": 0.11, "1": 0.46, "2": 0.43 },
      "confidence": 0.41
    }
  },
  "usage": { "input_tokens": 61, "output_tokens": 12 }
}
```

Guarantees:

* `probabilities` contain exactly the supplied keys, every value in [0, 1],
  and sum to 1 (the residual of floating-point normalisation is pushed onto
  the largest entry, so the sum is exactly 1.0 in f64 arithmetic).
* `choice` is the argmax; ties are broken by lexicographic key order, which
  makes the result independent of option order.
* `score` = Σ level_index × P(level) ∈ [0, K−1].
* `noul` ∈ [0, 1] is a calibrated probability, not a similarity score.
* `confidence` ∈ [0, 1] is a calibrated function of the distribution's shape
  (see `docs/CALIBRATION.md`), not `max(p)`.
* Answers are deterministic and independent of other questions in the request.

## Explain block

With `explain=true` each answer carries:

```json
"explain": {
  "family": "routing",
  "path": "semantic",                      // or "symbolic:<resolver>"
  "top_evidence": [{"path": "", "text": "…", "score": 1.83}],
  "raw_scores": {"billing": 2.1, "technical": 0.4, "sales": 0.3},
  "features": {"billing": {"cov_w": 0.62, "neg_conflict": 0.0, …}, …},
  "calibration": {"method": "temperature+platt", "temperature": 0.57, "bucket": "3-4", "family": "routing"},
  "notes": ["value 1240 falls in level `2`"]
}
```

Explain mode never changes inference; it only reports it.

## Errors

| Status | `error.code` | When |
|---|---|---|
| 404 | `unknown_model` | `model` is not `sextant-1` / an alias |
| 404 | `not_found` | unknown route |
| 413 | `payload_too_large` | body over the limit (default 4 MiB) or state over its limits |
| 422 | `invalid_request` | malformed JSON, missing fields, wrong cardinalities, nesting too deep, empty ids… (`error.field` names the offending field) |
| 500 | `internal_error` | worker failure (should never happen; please report) |

Body shape: `{"error": {"status": 422, "code": "invalid_request", "message": "…", "field": "questions.q.criteria"}}`.

## Limits (defaults)

| Limit | Value |
|---|---|
| request body | 4 MiB (`--max-body`) |
| state size | 2 MiB, 50,000 leaves, depth 64 |
| questions per request | 1024 |
| choice options | 2–255 |
| score levels | 2–10 |
| option key / question id length | 512 bytes |
| single question size | 256 KiB |

## CLI equivalents

```bash
sextant decide request.json --pretty          # or: cat request.json | sextant decide -
sextant decide request.json --explain
sextant validate request.json
sextant serve --addr 0.0.0.0:8080
sextant bench --iters 200 --out reports/bench.json
```
