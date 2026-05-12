"""LLM client for the event-sourced pipeline.

One class, ~250 LOC. Handles:
  - provider routing (LiteLLM for real models, in-process mock for `mock/*`)
  - per-(provider, model) RPM rate limiting via a simple token bucket
  - retries via tenacity (exponential backoff)
  - cost accounting from LiteLLM `usage` and a default price table

The pipeline emits :py:class:`LLMRequest` objects; the dispatcher passes them
to :py:meth:`LLMClient.complete`, which returns :py:class:`Response`. Mock
scenarios are registered via :py:func:`register_mock` and share the same
registry as the legacy ``models.LLMClient`` so existing test scenarios
(``mock/happy``, ``mock/reject``, ``mock/metaopt``) keep working during the
migration.

See MIGRATION_PLAN.md §5.
"""
from __future__ import annotations

import fnmatch
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Share the mock registry with the legacy client so a single
# `register_mock("happy", handler)` call works for both code paths.
from autodata.models import _MOCK_REGISTRY, MockHandler, register_mock as _register_mock
from autodata.utils import extract_json

__all__ = [
    "LLMRequest",
    "Response",
    "LLMClient",
    "LLMConfig",
    "RateLimitSpec",
    "Message",
    "TokenBucket",
    "register_mock",
    "price_for",
]


Message = dict[str, str]  # {"role": "system|user|assistant", "content": "..."}

Role = Literal["challenger", "quality", "weak", "strong", "judge", "reflector", "meta_mutator"]


@dataclass(frozen=True)
class LLMRequest:
    """One LLM call. Constructed by the pipeline, fulfilled by the dispatcher."""

    request_id: str
    item_id: str
    round_n: int
    role: str  # one of `Role`; typed loosely to avoid runtime Literal validation
    model_key: str  # provider model string, e.g. "openai/gpt-4o-mini" or "mock/happy"
    messages: list[Message]
    json_mode: bool = False
    attempt: int = 0
    parent_response_id: Optional[str] = None
    # Per-call overrides (rarely needed; usually use LLMConfig defaults).
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class Response:
    """Provider response in the shape the store and pipeline both consume."""

    request_id: str
    model: str
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    duration_ms: int = 0

    def parse_json(self) -> dict[str, Any]:
        return extract_json(self.text)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RateLimitSpec(BaseModel):
    """RPM only; TPM is logged but not enforced (no pre-call token counting)."""

    rpm: Optional[int] = None  # None = no limit
    burst: Optional[int] = None  # default = rpm / 4, min 1


class LLMConfig(BaseModel):
    """Top-level LLM settings, separate from per-role ModelConfig defaults."""

    rate_limits: dict[str, RateLimitSpec] = Field(default_factory=dict)
    max_retries: int = 3
    request_timeout_s: int = 60
    # Per-model price overrides: { "openai/gpt-4o-mini": [input_per_million, output_per_million] }
    prices: dict[str, list[float]] = Field(default_factory=dict)
    # Default sampling params applied when LLMRequest leaves them None.
    default_temperature: float = 0.7
    default_max_tokens: int = 2048


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# Per-million-token prices (input, output). Conservative defaults; users
# override via LLMConfig.prices. Unknown models → cost_usd = None.
_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4o":                 (2.50, 10.00),
    "openai/gpt-4o-mini":            (0.15, 0.60),
    "openai/gpt-4.1":                (2.00, 8.00),
    "openai/gpt-4.1-mini":           (0.40, 1.60),
    "anthropic/claude-opus-4-7":     (15.00, 75.00),
    "anthropic/claude-sonnet-4-6":   (3.00, 15.00),
    "anthropic/claude-haiku-4-5":    (0.80, 4.00),
}


def price_for(model_key: str, *, overrides: Optional[dict[str, list[float]]] = None) -> Optional[tuple[float, float]]:
    if overrides and model_key in overrides:
        p = overrides[model_key]
        if len(p) >= 2:
            return float(p[0]), float(p[1])
    return _DEFAULT_PRICES.get(model_key)


def _compute_cost(model_key: str, usage: dict[str, int],
                  overrides: Optional[dict[str, list[float]]] = None) -> Optional[float]:
    p = price_for(model_key, overrides=overrides)
    if not p:
        return None
    pin, pout = p
    return (usage.get("prompt_tokens", 0) * pin
            + usage.get("completion_tokens", 0) * pout) / 1_000_000


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class TokenBucket:
    """Thread-safe RPM token bucket. `acquire()` blocks until a slot is free.

    The clock and sleep functions are injectable for deterministic tests.
    """

    def __init__(self, rate_per_sec: float, burst: int,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self.rate = rate_per_sec
        self.burst = burst
        self._tokens = float(burst)
        self._last = clock()
        self._lock = threading.Lock()
        self._clock = clock
        self._sleep = sleep

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> None:
        deadline = (self._clock() + timeout) if timeout is not None else None
        while True:
            with self._lock:
                now = self._clock()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                need = tokens - self._tokens
                wait = need / self.rate
            if deadline is not None and self._clock() + wait > deadline:
                raise TimeoutError("token bucket acquire timeout")
            self._sleep(wait)


# ---------------------------------------------------------------------------
# Mock registry (shared with legacy autodata.models)
# ---------------------------------------------------------------------------

def register_mock(scenario: str, handler: MockHandler) -> None:
    """Register a scripted mock handler. Same registry as autodata.models."""
    _register_mock(scenario, handler)


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Single entry point for completions across the run."""

    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        self._buckets: dict[str, TokenBucket] = {}
        self._buckets_lock = threading.Lock()

    # ---- entry point ----------------------------------------------------

    def complete(self, req: LLMRequest) -> Response:
        bucket = self._limiter_for(req.model_key)
        if bucket is not None:
            bucket.acquire()
        if req.model_key.startswith("mock/"):
            return self._call_mock(req)
        return self._call_real(req)

    # ---- rate limiting --------------------------------------------------

    def _limiter_for(self, model_key: str) -> Optional[TokenBucket]:
        with self._buckets_lock:
            if model_key in self._buckets:
                return self._buckets[model_key]
            spec = self._match_rate_limit(model_key)
            if spec is None or spec.rpm is None:
                return None
            rate = spec.rpm / 60.0
            burst = spec.burst or max(1, spec.rpm // 4)
            bucket = TokenBucket(rate_per_sec=rate, burst=burst)
            self._buckets[model_key] = bucket
            return bucket

    def _match_rate_limit(self, model_key: str) -> Optional[RateLimitSpec]:
        # Exact match wins; otherwise first matching glob in insertion order.
        if model_key in self.cfg.rate_limits:
            return self.cfg.rate_limits[model_key]
        for pattern, spec in self.cfg.rate_limits.items():
            if fnmatch.fnmatchcase(model_key, pattern):
                return spec
        return None

    # ---- mock dispatch --------------------------------------------------

    def _call_mock(self, req: LLMRequest) -> Response:
        t0 = time.monotonic()
        raw = _MOCK_REGISTRY.dispatch(req.model_key, req.role, req.messages)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return Response(
            request_id=req.request_id,
            model=req.model_key,
            text=raw.text,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            duration_ms=elapsed_ms,
        )

    # ---- real (LiteLLM) dispatch ---------------------------------------

    def _call_real(self, req: LLMRequest) -> Response:
        import litellm  # lazy import

        @retry(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> Any:
            kwargs: dict[str, Any] = dict(
                model=req.model_key,
                messages=req.messages,
                temperature=(req.temperature if req.temperature is not None
                             else self.cfg.default_temperature),
                max_tokens=(req.max_tokens if req.max_tokens is not None
                            else self.cfg.default_max_tokens),
                timeout=self.cfg.request_timeout_s,
            )
            if req.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return litellm.completion(**kwargs)

        t0 = time.monotonic()
        try:
            resp = _call()
        except Exception as e:  # pragma: no cover - real network
            logger.warning("LLM call failed after retries (role={}, model={}): {}",
                           req.role, req.model_key, e)
            raise
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage_obj = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        }
        cost = _compute_cost(req.model_key, usage, overrides=self.cfg.prices)
        return Response(
            request_id=req.request_id,
            model=req.model_key,
            text=text,
            prompt_tokens=usage["prompt_tokens"] or None,
            completion_tokens=usage["completion_tokens"] or None,
            cost_usd=cost,
            duration_ms=elapsed_ms,
        )
