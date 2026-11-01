"""Budgeted, disk-cached Messages API client.

Every request is keyed by a hash of its full parameters. A cache hit returns the
stored response at zero cost, so reruns are free and results reproducible even for
models that cannot be run deterministically. Every call, live or cached, is logged to
the budget ledger with tokens, cost and latency.

Neither the cache nor the ledger stores credentials or HTTP headers: the API key
lives only inside the SDK client, request params never contain it, and only a
whitelist of response-body fields is written to disk.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from certharness.llm.budget import Budget
from certharness.llm.pricing import PRICES, cost_usd, rates, worst_case_usd

Transport = Callable[[dict[str, Any]], dict[str, Any]]

# Response-body fields written to the disk cache; anything else is dropped.
STORED_RESPONSE_FIELDS = (
    "id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "stop_details",
    "usage",
)  # fmt: skip

# Conservative chars-per-token for the pre-call worst-case estimate.
_CHARS_PER_TOKEN = 2.5


def request_key(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()


def anthropic_transport() -> Transport:
    import anthropic  # imported lazily so offline use and tests need no SDK client or key

    client = anthropic.Anthropic(max_retries=4)

    def send(params: dict[str, Any]) -> dict[str, Any]:
        api = client.beta.messages if "betas" in params else client.messages
        return api.create(**split_params(api.create, params)).to_dict()

    return send


def split_params(create: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
    """Pass API fields the installed SDK has no named argument for (e.g. `temperature`
    in anthropic 2.x) through `extra_body`, so the wire request is unchanged."""
    accepted = inspect.signature(create).parameters
    kwargs = {k: v for k, v in params.items() if k in accepted}
    extra = {k: v for k, v in params.items() if k not in accepted}
    if extra:
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra}
    return kwargs


class LLMClient:
    def __init__(self, budget: Budget, cache_dir: Path, transport: Transport | None = None):
        self.budget = budget
        self.cache_dir = Path(cache_dir)
        self._transport = transport

    def create(self, purpose: str, **params: Any) -> dict[str, Any]:
        model = params["model"]
        rates(model)  # fail fast on unpriced models, before anything is sent
        key = request_key(params)
        path = self.cache_dir / key[:2] / f"{key}.json"
        entry = {"ts": datetime.now(UTC).isoformat(), "purpose": purpose, "model": model,
                 "request_hash": key}  # fmt: skip

        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.budget.log({**entry, "cached": True, "cost_usd": 0.0, **stored["usage_summary"]})
            return stored["response"]

        est_in = int(len(json.dumps(params, default=str)) / _CHARS_PER_TOKEN)
        reserved = worst_case_usd(model, est_in, params.get("max_tokens", 4096))
        self.budget.reserve(reserved)
        t0 = time.perf_counter()
        try:
            if self._transport is None:
                self._transport = anthropic_transport()
            raw = self._transport(params)
            response = {k: raw[k] for k in STORED_RESPONSE_FIELDS if k in raw}
        except BaseException:
            self.budget.release(reserved)
            raise
        latency = time.perf_counter() - t0

        usage = response.get("usage") or {}
        served = response.get("model") or model
        # A server-side fallback may serve the request on another model; price at the
        # higher of the two rates so the cap stays conservative.
        price_model = served if served in PRICES and rates(served)[1] > rates(model)[1] else model
        summary = {
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens") or 0,
            "cache_read_input_tokens": usage.get("cache_read_input_tokens") or 0,
            "latency_s": round(latency, 3),
            "stop_reason": response.get("stop_reason"),
            "served_model": served,
        }
        cost = cost_usd(price_model, usage)
        self.budget.settle(reserved, {**entry, "cached": False, "cost_usd": cost, **summary})

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(
            json.dumps({"response": response, "usage_summary": summary, "cost_usd": cost}),
            encoding="utf-8",
        )
        tmp.replace(path)
        return response
