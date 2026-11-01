"""Per-token prices for spend accounting. Update PRICES_AS_OF when you change them.

Cache writes (5-minute TTL) cost 1.25x the input rate; cache reads cost 0.1x.
An unknown model raises, because the spend cap cannot be enforced without a price.
"""

from __future__ import annotations

from typing import Any

PRICES_AS_OF = "2026-06-24"

# model -> (input $/MTok, output $/MTok)
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


def rates(model: str) -> tuple[float, float]:
    try:
        return PRICES[model]
    except KeyError:
        raise KeyError(f"no price for {model!r}; add it to certharness.llm.pricing") from None


def cost_usd(model: str, usage: dict[str, Any]) -> float:
    inp, out = rates(model)
    tokens_in = usage.get("input_tokens") or 0
    tokens_out = usage.get("output_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    return (
        tokens_in * inp
        + cache_write * inp * CACHE_WRITE_MULT
        + cache_read * inp * CACHE_READ_MULT
        + tokens_out * out
    ) / 1e6


def worst_case_usd(model: str, est_input_tokens: int, max_tokens: int) -> float:
    """Upper bound for one call: all input uncached at the cache-write rate, full output."""
    inp, out = rates(model)
    return (est_input_tokens * inp * CACHE_WRITE_MULT + max_tokens * out) / 1e6
