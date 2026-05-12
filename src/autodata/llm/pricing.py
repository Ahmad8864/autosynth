"""Per-million-token pricing for cost accounting.

Conservative defaults; users override via :class:`LLMConfig.prices`. Unknown
models return ``None`` so cost stays unbilled rather than guessed.
"""
from __future__ import annotations

_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4o":                 (2.50, 10.00),
    "openai/gpt-4o-mini":            (0.15, 0.60),
    "openai/gpt-4.1":                (2.00, 8.00),
    "openai/gpt-4.1-mini":           (0.40, 1.60),
    "anthropic/claude-opus-4-7":     (15.00, 75.00),
    "anthropic/claude-sonnet-4-6":   (3.00, 15.00),
    "anthropic/claude-haiku-4-5":    (0.80, 4.00),
}


def price_for(model_key: str, *, overrides: dict[str, list[float]] | None = None) -> tuple[float, float] | None:
    if overrides and model_key in overrides:
        p = overrides[model_key]
        if len(p) >= 2:
            return float(p[0]), float(p[1])
    return _DEFAULT_PRICES.get(model_key)


def compute_cost(model_key: str, usage: dict[str, int],
                 overrides: dict[str, list[float]] | None = None) -> float | None:
    p = price_for(model_key, overrides=overrides)
    if not p:
        return None
    pin, pout = p
    return (usage.get("prompt_tokens", 0) * pin
            + usage.get("completion_tokens", 0) * pout) / 1_000_000
