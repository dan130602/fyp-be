"""Dated price table and cost estimates (USD per million tokens). Estimates only; re-check before paid runs."""

PRICE_TABLE_VERSION = "2026-10-09"
PRICES = {
    "gpt-6.1-sol": {
        "standard": {"input": 2.0, "cached_input": 0.10, "output": 10.0},
        "batch": {"input": 1.0, "cached_input": 0.05, "output": 5.0},
    },
}
CACHE_WRITE_MULTIPLIER = 1.25


def usage_summary(usage):
    """Normalise a provider usage object (Responses API shape) into flat integers."""
    usage = usage or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_tokens": int(input_details.get("cached_tokens") or 0),
        "cache_write_tokens": int(input_details.get("cache_write_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
    }


def estimate_cost(model, usage, tier):
    """Estimated USD for one request, or None when the model/tier is not in the price table."""
    prices = PRICES.get(model, {}).get(tier)
    if prices is None:
        return None
    uncached = max(usage["input_tokens"] - usage["cached_tokens"], 0)
    cost = uncached * prices["input"] + usage["cached_tokens"] * prices["cached_input"]
    cost += usage["cache_write_tokens"] * prices["input"] * (CACHE_WRITE_MULTIPLIER - 1)
    cost += usage["output_tokens"] * prices["output"]  # reasoning tokens are billed as output
    return round(cost / 1_000_000, 6)
