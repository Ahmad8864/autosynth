"""LLM client for the event-sourced pipeline.

One class. Handles:

  - provider routing (LiteLLM for real models, in-process mock for ``mock/*``)
  - per-(provider, model) RPM rate limiting via a token bucket
  - retries via tenacity (exponential backoff with jitter)
  - cost accounting from LiteLLM ``usage`` and a default price table

The pipeline emits :class:`LLMRequest` objects; the dispatcher passes them
to :meth:`LLMClient.complete`, which returns a :class:`Response`. Mock
scenarios are registered via :func:`register_mock` and addressable as the
``mock/<scenario>`` model string.
"""
from __future__ import annotations

import fnmatch
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

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

# Agent roles understood by the pipeline; the field is typed `str` for forward
# compatibility (e.g. user-defined roles in custom domains).
ROLES = ("challenger", "quality", "weak", "strong", "judge", "reflector", "meta_mutator")


@dataclass(frozen=True)
class LLMRequest:
    """One LLM call. Constructed by the pipeline, fulfilled by the dispatcher."""

    request_id: str
    item_id: str
    round_n: int
    role: str
    model_key: str  # LiteLLM model string, e.g. "openai/gpt-4o-mini", "mock/happy"
    messages: list[Message]
    json_mode: bool = False
    attempt: int = 0
    parent_response_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class Response:
    """Provider response in the shape the store and pipeline both consume."""

    request_id: str
    model: str
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: int = 0

    def parse_json(self) -> dict[str, Any]:
        return extract_json(self.text)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RateLimitSpec(BaseModel):
    """RPM only; TPM is logged but not enforced (no pre-call token counting)."""

    rpm: int | None = None  # None = no limit
    burst: int | None = None  # default = rpm / 4, min 1


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


def price_for(model_key: str, *, overrides: dict[str, list[float]] | None = None) -> tuple[float, float] | None:
    if overrides and model_key in overrides:
        p = overrides[model_key]
        if len(p) >= 2:
            return float(p[0]), float(p[1])
    return _DEFAULT_PRICES.get(model_key)


def _compute_cost(model_key: str, usage: dict[str, int],
                  overrides: dict[str, list[float]] | None = None) -> float | None:
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

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> None:
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
# Mock provider — scripted, role-aware, deterministic
# ---------------------------------------------------------------------------

MockHandler = Callable[[str, list[Message]], str]
"""(role, messages) -> raw text. Should usually return JSON."""


_ROLE_TAGS = (
    ("challenger", "ROLE:CHALLENGER"),
    ("reflector", "ROLE:REFLECTION"),
    ("quality", "ROLE:QUALITY"),
    ("judge", "ROLE:JUDGE"),
    ("weak", "ROLE:WEAK"),
    ("strong", "ROLE:STRONG"),
    ("meta_mutator", "ROLE:META_MUTATOR"),
)

_ROUND_RE = re.compile(r"ROUND[=:\s]+(\d+)")


def _canonical_role(role: str, all_text: str) -> str:
    """Map (LLMClient role, prompt text) → canonical role for mock dispatch.

    A single role label can serve multiple agents (e.g. the judge client serves
    both quality_check and score), so mock handlers need to inspect the prompt.
    """
    for canon, tag in _ROLE_TAGS:
        if role == canon or tag in all_text:
            return canon
    return role


def _peek_round(text: str) -> int:
    m = _ROUND_RE.search(text)
    return int(m.group(1)) if m else 1


def _join_messages(messages: list[Message]) -> str:
    return " ".join(m.get("content", "") for m in messages)


class _MockRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, MockHandler] = {}
        self._lock = threading.Lock()

    def register(self, scenario: str, handler: MockHandler) -> None:
        with self._lock:
            self._handlers[scenario] = handler

    def dispatch(self, provider_model: str, role: str, messages: list[Message]) -> str:
        scenario = provider_model.split("/", 1)[1] if "/" in provider_model else "default"
        handler = self._handlers.get(scenario)
        if handler is None:
            if scenario not in ("default", "scripted"):
                logger.warning(
                    "mock scenario {!r} not registered; falling back to 'default'", scenario,
                )
            handler = self._handlers.get("default") or _default_mock_handler
        return handler(role, messages)


_MOCK_REGISTRY = _MockRegistry()


def register_mock(scenario: str, handler: MockHandler) -> None:
    """Register a scripted mock handler. Reachable via `mock/<scenario>` model strings."""
    _MOCK_REGISTRY.register(scenario, handler)


# ---------------------------------------------------------------------------
# Default mock scenarios (preserved from legacy autodata.models)
# ---------------------------------------------------------------------------

def _default_mock_handler(role: str, messages: list[Message]) -> str:
    """Built-in mock simulating a realistic generation→accept trajectory."""
    all_text = _join_messages(messages)
    canon = _canonical_role(role, all_text)

    if canon == "challenger":
        round_n = _peek_round(all_text)
        return json.dumps({
            "payload": {
                "question": f"What is the main contribution of the source, as understood at round {round_n}?",
                "context": "Synthetic context snippet for mock run.",
                "reasoning_skills": ["comprehension", "synthesis"],
            },
            "reference_output": "A concise synthesis of the source's main contribution.",
            "rubric": [
                {"id": "c1", "description": "Names the main contribution", "weight": 5},
                {"id": "c2", "description": "Cites at least one supporting detail", "weight": 3},
                {"id": "c3", "description": "Avoids generic boilerplate", "weight": 2},
            ],
        })
    if canon == "reflector":
        return json.dumps({
            "feedback": [
                "Make the question depend on a source-specific detail.",
                "Avoid framings answerable from generic knowledge.",
            ],
            "new_angle": "Target a quantitative claim or design choice unique to the source.",
        })
    if canon == "quality":
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if canon == "judge":
        solver_tag = "weak" if "[solver=weak]" in all_text else "strong"
        if solver_tag == "weak":
            return json.dumps({
                "per_criterion": {"c1": 0.2, "c2": 0.0, "c3": 0.1},
                "total": 0.13,
                "failure_modes": ["generic_response"],
            })
        return json.dumps({
            "per_criterion": {"c1": 0.95, "c2": 0.85, "c3": 0.8},
            "total": 0.88,
            "failure_modes": [],
        })
    if canon == "weak":
        return "The source seems to be about general AI topics. It probably contributes something useful."
    if canon == "strong":
        return (
            "The source's main contribution is a method to iteratively generate training data "
            "using weak/strong solver disagreement, supported by a quality verifier and reflective "
            "recipe updates. Specifically it shows wide weak-strong gaps on accepted examples."
        )
    return "{}"


register_mock("scripted", _default_mock_handler)
register_mock("default", _default_mock_handler)


_MARKER_RULE = "Target a quantitative or design-specific claim unique to the source."


def _metaopt_handler(role: str, messages: list[Message]) -> str:
    """Mock scenario used by the meta-optimization demo + tests."""
    all_text = _join_messages(messages)
    canon = _canonical_role(role, all_text)

    if canon == "meta_mutator":
        return json.dumps({
            "rationale": "Add source-specificity rule to widen weak/strong gap.",
            "challenger_rules_add": [_MARKER_RULE],
        })

    if canon == "challenger":
        marker = _MARKER_RULE in all_text
        c1 = "[SPECIFIC] Names a source-specific contribution" if marker else "Names contribution"
        return json.dumps({
            "payload": {
                "question": "What specific contribution does this source make?",
                "context": "Synthetic context.",
                "reasoning_skills": ["synthesis"],
            },
            "reference_output": "The contribution as named in the source.",
            "rubric": [
                {"id": "c1", "description": c1, "weight": 5},
                {"id": "c2", "description": "Cites a detail from the source", "weight": 3},
            ],
        })
    if canon == "reflector":
        return json.dumps({"feedback": ["push for source-specificity"], "new_angle": "quantitative claim"})
    if canon == "quality":
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if canon == "judge":
        specific = "[SPECIFIC]" in all_text
        if "[solver=weak]" in all_text:
            return json.dumps({"per_criterion": {"c1": 0.25, "c2": 0.10}, "total": 0.20, "failure_modes": []})
        return json.dumps({
            "per_criterion": {"c1": 0.95 if specific else 0.55, "c2": 0.7 if specific else 0.55},
            "total": 0.85 if specific else 0.55,
            "failure_modes": [] if specific else ["generic_response"],
        })
    if canon == "weak":
        return "generic weak attempt"
    if canon == "strong":
        return "specific strong attempt"
    return "{}"


register_mock("metaopt", _metaopt_handler)


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

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
        text = _MOCK_REGISTRY.dispatch(req.model_key, req.role, req.messages)
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
