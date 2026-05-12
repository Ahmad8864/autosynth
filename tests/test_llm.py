"""Tests for the new event-sourced LLMClient (autodata.llm).

Covers: token-bucket math, rate-limit glob matching, mock dispatch
preserves the legacy register_mock contract, cost accounting from a known
price table, retry on transient errors. The legacy autodata.models is
left untouched and not tested here — its tests live in test_mock_provider.py
until commit 6 deletes it.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from autodata.llm import (
    LLMClient,
    LLMConfig,
    LLMRequest,
    Message,
    RateLimitSpec,
    Response,
    TokenBucket,
    price_for,
    register_mock,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeSleep:
    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, dt: float) -> None:
        self.calls.append(dt)
        self.clock.advance(dt)


def _req(model_key: str = "mock/happy", role: str = "weak",
         messages: list[Message] | None = None,
         request_id: str = "req-1") -> LLMRequest:
    return LLMRequest(
        request_id=request_id,
        item_id="item-1",
        round_n=1,
        role=role,
        model_key=model_key,
        messages=messages or [{"role": "user", "content": "hello"}],
    )


# ---------------------------------------------------------------------------
# TokenBucket math
# ---------------------------------------------------------------------------

def test_bucket_serves_burst_immediately():
    clock = FakeClock()
    sleep = FakeSleep(clock)
    bucket = TokenBucket(rate_per_sec=1.0, burst=3, clock=clock, sleep=sleep)
    bucket.acquire()
    bucket.acquire()
    bucket.acquire()
    assert sleep.calls == []                  # no sleep within burst


def test_bucket_throttles_after_burst():
    clock = FakeClock()
    sleep = FakeSleep(clock)
    bucket = TokenBucket(rate_per_sec=2.0, burst=2, clock=clock, sleep=sleep)
    bucket.acquire()       # burst 2 → 1
    bucket.acquire()       # burst 1 → 0
    bucket.acquire()       # must wait 0.5s for one refill
    assert len(sleep.calls) == 1
    assert sleep.calls[0] == pytest.approx(0.5, rel=1e-6)


def test_bucket_refills_continuously():
    clock = FakeClock()
    sleep = FakeSleep(clock)
    bucket = TokenBucket(rate_per_sec=1.0, burst=1, clock=clock, sleep=sleep)
    bucket.acquire()
    clock.advance(2.0)      # 2 tokens accumulated but capped at burst=1
    bucket.acquire()
    assert sleep.calls == []


def test_bucket_validates_positive_rate():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=0, burst=1)
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=1, burst=0)


# ---------------------------------------------------------------------------
# Rate-limit glob matching
# ---------------------------------------------------------------------------

def test_rate_limit_exact_match_wins():
    cfg = LLMConfig(rate_limits={
        "openai/gpt-4o-mini": RateLimitSpec(rpm=10),
        "openai/*": RateLimitSpec(rpm=1000),
    })
    client = LLMClient(cfg)
    bucket = client._limiter_for("openai/gpt-4o-mini")
    assert bucket is not None
    assert bucket.rate == pytest.approx(10 / 60.0)


def test_rate_limit_glob_fallback():
    cfg = LLMConfig(rate_limits={"openai/*": RateLimitSpec(rpm=600)})
    client = LLMClient(cfg)
    bucket = client._limiter_for("openai/gpt-4o")
    assert bucket is not None
    assert bucket.rate == pytest.approx(10.0)
    # Other providers unaffected.
    assert client._limiter_for("anthropic/whatever") is None


def test_rate_limit_none_means_unlimited():
    cfg = LLMConfig(rate_limits={"mock/*": RateLimitSpec(rpm=None)})
    client = LLMClient(cfg)
    assert client._limiter_for("mock/happy") is None


# ---------------------------------------------------------------------------
# Mock dispatch
# ---------------------------------------------------------------------------

def test_mock_dispatch_routes_to_registered_handler():
    register_mock("llm_test_echo", lambda role, msgs: f"echo:{role}")
    client = LLMClient()
    resp = client.complete(_req(model_key="mock/llm_test_echo", role="weak"))
    assert resp.text == "echo:weak"
    assert resp.model == "mock/llm_test_echo"
    assert resp.cost_usd == 0.0
    assert resp.request_id == "req-1"


def test_mock_dispatch_shares_registry_with_legacy_models():
    # Mocks registered via autodata.models.register_mock must be reachable.
    from autodata.models import register_mock as legacy_register_mock
    legacy_register_mock("llm_legacy_echo", lambda role, msgs: "from-legacy")
    client = LLMClient()
    resp = client.complete(_req(model_key="mock/llm_legacy_echo"))
    assert resp.text == "from-legacy"


def test_mock_response_parses_json():
    register_mock("llm_test_json", lambda role, msgs: '{"ok": true}')
    resp = LLMClient().complete(_req(model_key="mock/llm_test_json"))
    parsed = resp.parse_json()
    assert parsed == {"ok": True}


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def test_price_for_known_model():
    p = price_for("openai/gpt-4o-mini")
    assert p == (0.15, 0.60)


def test_price_for_unknown_model_returns_none():
    assert price_for("brand-new-provider/foo") is None


def test_price_override():
    p = price_for("custom/foo", overrides={"custom/foo": [1.0, 2.0]})
    assert p == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Real (LiteLLM) dispatch — mocked
# ---------------------------------------------------------------------------

class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})()


class _FakeResp:
    def __init__(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.choices = [_FakeChoice(content)]
        self.usage = type("U", (), {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })()


def test_real_dispatch_computes_cost(monkeypatch):
    fake = _FakeResp("the answer")
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)

    client = LLMClient()
    req = LLMRequest(request_id="r-1", item_id="i", round_n=1, role="weak",
                     model_key="openai/gpt-4o-mini",
                     messages=[{"role": "user", "content": "x"}])
    resp = client.complete(req)
    assert resp.text == "the answer"
    assert resp.prompt_tokens == 100
    assert resp.completion_tokens == 50
    # (100 * 0.15 + 50 * 0.60) / 1e6 = 0.000045
    assert resp.cost_usd == pytest.approx(0.000045, rel=1e-6)
    assert calls[0]["model"] == "openai/gpt-4o-mini"
    assert calls[0]["temperature"] == 0.7  # default


def test_real_dispatch_passes_json_mode_and_overrides(monkeypatch):
    fake = _FakeResp('{"k":1}')
    captured: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return fake

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)

    client = LLMClient(LLMConfig(default_temperature=0.5, default_max_tokens=1024))
    req = LLMRequest(request_id="r-2", item_id="i", round_n=1, role="judge",
                     model_key="openai/gpt-4o", messages=[{"role": "user", "content": "x"}],
                     json_mode=True, temperature=0.0, max_tokens=512)
    client.complete(req)
    kwargs = captured[0]
    assert kwargs["temperature"] == 0.0          # request override wins
    assert kwargs["max_tokens"] == 512
    assert kwargs["response_format"] == {"type": "json_object"}


def test_real_dispatch_retries_on_transient_failure(monkeypatch):
    attempts = {"n": 0}

    def flaky_completion(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return _FakeResp("ok")

    import litellm
    monkeypatch.setattr(litellm, "completion", flaky_completion)

    client = LLMClient(LLMConfig(max_retries=4))
    # Use a deterministic-but-fast wait by monkey-patching tenacity's sleep.
    import tenacity
    monkeypatch.setattr(tenacity.nap, "sleep", lambda *_: None)

    req = LLMRequest(request_id="r-3", item_id="i", round_n=1, role="weak",
                     model_key="openai/gpt-4o-mini",
                     messages=[{"role": "user", "content": "x"}])
    resp = client.complete(req)
    assert resp.text == "ok"
    assert attempts["n"] == 3


def test_real_dispatch_gives_up_after_max_retries(monkeypatch):
    def always_fails(**kwargs):
        raise RuntimeError("permanent")

    import litellm
    monkeypatch.setattr(litellm, "completion", always_fails)
    import tenacity
    monkeypatch.setattr(tenacity.nap, "sleep", lambda *_: None)

    client = LLMClient(LLMConfig(max_retries=2))
    req = LLMRequest(request_id="r-4", item_id="i", round_n=1, role="weak",
                     model_key="openai/gpt-4o-mini",
                     messages=[{"role": "user", "content": "x"}])
    with pytest.raises(RuntimeError, match="permanent"):
        client.complete(req)


# ---------------------------------------------------------------------------
# Rate limiter integration with complete()
# ---------------------------------------------------------------------------

def test_complete_respects_rate_limit():
    register_mock("llm_test_slow", lambda role, msgs: "ok")
    cfg = LLMConfig(rate_limits={"mock/llm_test_slow": RateLimitSpec(rpm=60, burst=1)})
    client = LLMClient(cfg)
    # 2 calls; the second waits ~1s for refill. Use a tight time budget.
    t0 = time.monotonic()
    client.complete(_req(model_key="mock/llm_test_slow", request_id="a"))
    client.complete(_req(model_key="mock/llm_test_slow", request_id="b"))
    elapsed = time.monotonic() - t0
    # 60 rpm = 1 per second; burst=1 → second call waits ~1s
    assert 0.8 <= elapsed <= 1.4
