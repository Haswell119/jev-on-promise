"""Thin Python client for the Sextant decision engine (System One compatible API).

Standard library only (urllib); Python 3.8+. One file: copy it next to your code or
``pip install -e clients/python``-style vendor it.

    from sextant_client import SextantClient, choice, score, noul

    client = SextantClient("http://127.0.0.1:8080")
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
    )
    reply["answers"]["department"]["choice"]          # -> "billing"
    reply["answers"]["is_urgent"]["noul"]             # -> P(true), 0..1
    reply["answers"]["frustration"]["score"]          # -> expected level index, 0..2

Any non-2xx response raises ``SextantError`` carrying the server's structured error
body (``status``, ``code``, ``message``, ``field``). Wire format: docs/openapi.yaml.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Mapping, Optional

__all__ = ["SextantClient", "SextantError", "choice", "score", "noul", "DEFAULT_MODEL", "__version__"]
__version__ = "0.1.0"

DEFAULT_MODEL = "sextant-1"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"

Entry = Any  # string | object | array | None


class SextantError(Exception):
    """A failed request. ``status`` is the HTTP status (0 when no response arrived);
    ``code``, ``message`` and ``field`` mirror the server's ``{"error": {...}}`` body,
    which is also kept verbatim in ``body``."""

    def __init__(self, status: int, code: str, message: str, field: Optional[str] = None, body: Any = None):
        detail = f"HTTP {status} {code}: {message}" if status else f"{code}: {message}"
        if field:
            detail += f" (field: {field})"
        super().__init__(detail)
        self.status = status
        self.code = code
        self.message = message
        self.field = field
        self.body = body

    @classmethod
    def from_response(cls, status: int, payload: bytes) -> "SextantError":
        text = payload.decode("utf-8", errors="replace")
        try:
            body = json.loads(text) if text else None
        except ValueError:
            body = text
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            return cls(status, str(err.get("code") or "error"), str(err.get("message") or text or "request failed"),
                       err.get("field"), body)
        return cls(status, "http_error", text[:500] or "request failed", None, body)


def choice(instructions: Entry, criteria: Mapping[str, Entry]) -> Dict[str, Any]:
    """A Choice question: pick one of 2..255 options (key -> description)."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: Entry, criteria: Iterable[Entry]) -> Dict[str, Any]:
    """A Score question: place the state on 2..10 ordered levels (index 0 = lowest)."""
    return {"type": "score", "instructions": instructions, "criteria": list(criteria)}


def noul(instructions: Entry, criteria: Optional[Mapping[str, Entry]] = None) -> Dict[str, Any]:
    """A Noul question: calibrated P(true). ``criteria`` may describe the ``true`` / ``false`` sides."""
    question: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = dict(criteria)
    return question


class SextantClient:
    """Minimal HTTP client. Safe to share between threads (no state beyond configuration)."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0,
                 headers: Optional[Mapping[str, str]] = None, user_agent: str = f"sextant-client-python/{__version__}"):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Accept": "application/json", "User-Agent": user_agent, **(headers or {})}

    # ------------------------------------------------------------------ endpoints

    def health(self) -> Dict[str, Any]:
        """GET /health -> {"status": "ok", "version", "model", "uptime_seconds", "neural", "network_required"}."""
        return self._request("GET", "/health")

    def models(self) -> Dict[str, Any]:
        """GET /v1/models -> {"object": "list", "models": [...]}."""
        return self._request("GET", "/v1/models")

    def systemone(self, state: Any, questions: Mapping[str, Mapping[str, Any]],
                  model: str = DEFAULT_MODEL, explain: bool = False) -> Dict[str, Any]:
        """POST /v1/systemone. ``state`` is a string, object or array; ``questions`` maps
        caller-chosen ids to questions (see ``choice`` / ``score`` / ``noul``). With
        ``explain=True`` every answer carries an ``explain`` block (inference is unchanged).
        Returns {"model", "answers": {id: answer}, "usage"}."""
        body: Dict[str, Any] = {"model": model, "state": state, "questions": dict(questions)}
        if explain:
            body["explain"] = True
        return self._request("POST", "/v1/systemone", body)

    # ------------------------------------------------------------------ transport

    def _request(self, method: str, path: str, body: Any = None) -> Dict[str, Any]:
        data = None
        headers = dict(self.headers)
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            raise SextantError.from_response(e.code, e.read()) from None
        except urllib.error.URLError as e:
            raise SextantError(0, "connection_error", f"{self.base_url}: {e.reason}") from e
        try:
            return json.loads(payload.decode("utf-8"))
        except ValueError as e:
            raise SextantError(0, "malformed_response", f"non-JSON body from {path}: {e}", None, payload[:500]) from None
