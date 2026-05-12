"""LLMClient: provider routing, rate limiting, retries, cost accounting.

Real models go through LiteLLM; ``mock/*`` model strings dispatch to the
in-process mock registry. Per-(provider, model) RPM is enforced via a token
bucket; retries use tenacity with exponential backoff.
"""
from __future__ import annotations

import fnmatch
import threading
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from autodata.llm.mock import dispatch_mock
from autodata.llm.pricing import compute_cost
from autodata.llm.rate_limit import RateLimitSpec, TokenBucket
from autodata.llm.types import LLMRequest, Response


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


class LLMClient:
    """Single entry point for completions across the run."""

    def __init__(self, cfg: LLMConfig | None = None):
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

    def _limiter_for(self, model_key: str) -> TokenBucket | None:
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

    def _match_rate_limit(self, model_key: str) -> RateLimitSpec | None:
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
        text = dispatch_mock(req.model_key, req.role, req.messages)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return Response(
            request_id=req.request_id,
            model=req.model_key,
            text=text,
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
        cost = compute_cost(req.model_key, usage, overrides=self.cfg.prices)
        return Response(
            request_id=req.request_id,
            model=req.model_key,
            text=text,
            prompt_tokens=usage["prompt_tokens"] or None,
            completion_tokens=usage["completion_tokens"] or None,
            cost_usd=cost,
            duration_ms=elapsed_ms,
        )
