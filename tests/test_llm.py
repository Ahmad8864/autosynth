"""Tests for the new event-sourced LLMClient (autosynth.llm).

Covers: token-bucket math, rate-limit glob matching, mock dispatch
preserves the legacy register_mock contract, cost accounting delegated
to ``litellm.completion_cost``, retry on transient errors.
"""

from __future__ import annotations

from typing import Any

import pytest

from autosynth.llm import (
    LLMClient,
    LLMConfig,
    LLMRequest,
    Message,
    RateLimitSpec,
    TokenBucket,
    register_mock,
)
from autosynth.llm.response_format import (
    ChallengerEnvelope,
    JudgeOutput,
    LoopJudgeOutput,
    MutatorOutput,
    QualityOutput,
    ReflectorOutput,
    _strict_compatible,
    challenger_schema_for,
    response_format_for,
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


def _req(
    model_key: str = "mock/happy",
    role: str = "weak",
    messages: list[Message] | None = None,
    request_id: str = "req-1",
    response_schema: Any = None,
) -> LLMRequest:
    return LLMRequest(
        request_id=request_id,
        item_id="item-1",
        round_n=1,
        role=role,
        model_key=model_key,
        messages=messages or [{"role": "user", "content": "hello"}],
        response_schema=response_schema,
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
    assert sleep.calls == []  # no sleep within burst


def test_bucket_throttles_after_burst():
    clock = FakeClock()
    sleep = FakeSleep(clock)
    bucket = TokenBucket(rate_per_sec=2.0, burst=2, clock=clock, sleep=sleep)
    bucket.acquire()  # burst 2 → 1
    bucket.acquire()  # burst 1 → 0
    bucket.acquire()  # must wait 0.5s for one refill
    assert len(sleep.calls) == 1
    assert sleep.calls[0] == pytest.approx(0.5, rel=1e-6)


def test_bucket_refills_continuously():
    clock = FakeClock()
    sleep = FakeSleep(clock)
    bucket = TokenBucket(rate_per_sec=1.0, burst=1, clock=clock, sleep=sleep)
    bucket.acquire()
    clock.advance(2.0)  # 2 tokens accumulated but capped at burst=1
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
    cfg = LLMConfig(
        rate_limits={
            "openai/gpt-4o-mini": RateLimitSpec(rpm=10),
            "openai/*": RateLimitSpec(rpm=1000),
        }
    )
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


def test_mock_dispatch_falls_back_to_default_on_unknown_scenario():
    client = LLMClient()
    resp = client.complete(_req(model_key="mock/this_scenario_doesnt_exist", role="weak"))
    # Default scripted handler returns the weak solver's canned text.
    assert "general AI topics" in resp.text or resp.text == "{}"


def test_mock_response_parses_json():
    register_mock("llm_test_json", lambda role, msgs: '{"ok": true}')
    resp = LLMClient().complete(_req(model_key="mock/llm_test_json"))
    parsed = resp.parse_json()
    assert parsed == {"ok": True}


# ---------------------------------------------------------------------------
# Real (LiteLLM) dispatch — mocked
# ---------------------------------------------------------------------------


class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})()


class _FakeResp:
    def __init__(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.choices = [_FakeChoice(content)]
        self.usage = type(
            "U",
            (),
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )()


def test_real_dispatch_computes_cost(monkeypatch):
    fake = _FakeResp("the answer")
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake

    cost_calls: list[Any] = []

    def fake_cost(*, completion_response):
        cost_calls.append(completion_response)
        return 0.000045

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", fake_cost)

    client = LLMClient()
    req = LLMRequest(
        request_id="r-1",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
    )
    resp = client.complete(req)
    assert resp.text == "the answer"
    assert resp.prompt_tokens == 100
    assert resp.completion_tokens == 50
    assert cost_calls == [fake]
    assert resp.cost_usd == pytest.approx(0.000045, rel=1e-6)
    assert calls[0]["model"] == "openai/gpt-4o-mini"
    # Request didn't carry a temperature, so the client omits it and the
    # provider's own default applies.
    assert "temperature" not in calls[0]
    assert "max_tokens" not in calls[0]


def test_real_dispatch_cost_none_when_litellm_raises(monkeypatch):
    fake = _FakeResp("hi")

    def fake_completion(**kwargs):
        return fake

    def boom(*, completion_response):
        raise RuntimeError("unknown model")

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", boom)

    client = LLMClient()
    req = LLMRequest(
        request_id="r-cost-none",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="brand-new-provider/foo",
        messages=[{"role": "user", "content": "x"}],
    )
    resp = client.complete(req)
    assert resp.cost_usd is None


def test_price_override_registers_with_litellm(monkeypatch):
    fake = _FakeResp("hi")
    registered: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        return fake

    def fake_register(reg):
        registered.append(reg)

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "register_model", fake_register)
    monkeypatch.setattr(litellm, "completion_cost", lambda *, completion_response: 0.0)

    client = LLMClient(LLMConfig(prices={"custom/foo": [1.0, 2.0]}))
    req = LLMRequest(
        request_id="r-reg",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="custom/foo",
        messages=[{"role": "user", "content": "x"}],
    )
    client.complete(req)
    # And again — registration should be a one-shot.
    client.complete(req)
    assert registered == [
        {
            "custom/foo": {
                "input_cost_per_token": 1.0 / 1_000_000,
                "output_cost_per_token": 2.0 / 1_000_000,
            }
        }
    ]


def test_real_dispatch_passes_json_mode_and_overrides(monkeypatch):
    fake = _FakeResp('{"k":1}')
    captured: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return fake

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)

    client = LLMClient()
    # A role with no strict schema exercises the json_object fallback branch
    # deterministically (no dependency on litellm's per-model schema support).
    req = LLMRequest(
        request_id="r-2",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="openai/gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        json_mode=True,
        temperature=0.0,
        max_tokens=512,
    )
    client.complete(req)
    kwargs = captured[0]
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 512
    assert kwargs["response_format"] == {"type": "json_object"}


def test_real_dispatch_spreads_model_extras(monkeypatch):
    fake = _FakeResp("hi")
    captured: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return fake

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda *, completion_response: 0.0)

    client = LLMClient(
        LLMConfig(
            model_extras={
                "azure/my-deployment": {
                    "api_base": "https://example.openai.azure.com",
                    "api_version": "2024-02-01",
                    # Should NOT override the explicit per-call value:
                    "temperature": 0.99,
                },
                "openai/gpt-4o": {"api_base": "should-not-leak"},
            }
        )
    )
    req = LLMRequest(
        request_id="r-extras",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="azure/my-deployment",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.0,
    )
    client.complete(req)

    kwargs = captured[0]
    assert kwargs["api_base"] == "https://example.openai.azure.com"
    assert kwargs["api_version"] == "2024-02-01"
    assert kwargs["temperature"] == 0.0  # explicit kwarg wins over extras
    assert kwargs["model"] == "azure/my-deployment"
    # Unrelated model's extras must not bleed in.
    assert "should-not-leak" not in kwargs.values()


def test_real_dispatch_unset_model_extras_is_noop(monkeypatch):
    fake = _FakeResp("hi")
    captured: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return fake

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda *, completion_response: 0.0)

    client = LLMClient()  # no model_extras configured
    req = LLMRequest(
        request_id="r-noextras",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
    )
    client.complete(req)
    # No sampling params on the request → none in kwargs. Provider defaults apply.
    assert set(captured[0].keys()) == {"model", "messages", "timeout"}


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

    req = LLMRequest(
        request_id="r-3",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
    )
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
    req = LLMRequest(
        request_id="r-4",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
    )
    with pytest.raises(RuntimeError, match="permanent"):
        client.complete(req)


# ---------------------------------------------------------------------------
# Rate limiter integration with complete()
# ---------------------------------------------------------------------------


def test_complete_respects_rate_limit():
    """complete() acquires from the model's bucket before dispatching. We inject
    a fake-clock bucket so the throttle is asserted deterministically rather than
    by sleeping ~1s in real time (the spec→bucket wiring is covered separately by
    the _limiter_for tests above)."""
    register_mock("llm_test_slow", lambda role, msgs: "ok")
    clock = FakeClock()
    sleep = FakeSleep(clock)
    client = LLMClient()
    # rpm=60 → 1 token/sec, burst=1: the second call must wait one full second.
    client._buckets["mock/llm_test_slow"] = TokenBucket(rate_per_sec=1.0, burst=1, clock=clock, sleep=sleep)
    client.complete(_req(model_key="mock/llm_test_slow", request_id="a"))  # burst slot
    client.complete(_req(model_key="mock/llm_test_slow", request_id="b"))  # waits for refill
    assert sleep.calls == [pytest.approx(1.0, rel=1e-6)]


# ---------------------------------------------------------------------------
# structured-outputs response_format gating
# ---------------------------------------------------------------------------


class _Litellm:
    def __init__(self, supported: bool | Exception):
        self._supported = supported

    def supports_response_schema(self, model: str) -> bool:
        if isinstance(self._supported, Exception):
            raise self._supported
        return self._supported


def test_response_format_uses_strict_schema_when_role_and_provider_support_it():
    lite = _Litellm(supported=True)
    assert response_format_for(lite, _req(role="quality")) is QualityOutput
    assert response_format_for(lite, _req(role="loop_judge")) is LoopJudgeOutput
    assert response_format_for(lite, _req(role="reflector")) is ReflectorOutput
    assert response_format_for(lite, _req(role="meta_mutator")) is MutatorOutput


def test_response_format_falls_back_to_json_object_when_unsupported():
    lite = _Litellm(supported=False)
    assert response_format_for(lite, _req(role="quality")) == {"type": "json_object"}
    # Even judge/meta_mutator drop back to plain JSON mode on old providers.
    assert response_format_for(lite, _req(role="judge")) == {"type": "json_object"}
    assert response_format_for(lite, _req(role="meta_mutator")) == {"type": "json_object"}


def test_response_format_challenger_without_schema_stays_json_object():
    # A challenger request with no attached response_schema (role alone carries
    # no fixed shape) falls back to plain JSON mode.
    lite = _Litellm(supported=True)
    assert response_format_for(lite, _req(role="challenger")) == {"type": "json_object"}


def test_strict_compatible_classifies_open_map_schemas():
    from autosynth.domains.math_word_problems import MathProblemPayload

    # Open-keyed dicts (judge per_criterion, generic challenger payload) are not
    # strict-compatible; every fixed-shape schema is.
    assert _strict_compatible(JudgeOutput) is False
    assert _strict_compatible(ChallengerEnvelope) is False
    assert _strict_compatible(QualityOutput) is True
    assert _strict_compatible(LoopJudgeOutput) is True
    assert _strict_compatible(ReflectorOutput) is True
    assert _strict_compatible(MutatorOutput) is True
    assert _strict_compatible(challenger_schema_for(MathProblemPayload)) is True


def test_response_format_open_map_schema_degrades_to_json_object_even_when_supported():
    # Schemas with an open map (judge per_criterion; the generic challenger
    # envelope's free-form payload) are outside the provider strict subset and
    # 400 on OpenAI/Azure, so the guard falls them back to json_object even when
    # the provider reports schema support.
    lite = _Litellm(supported=True)
    assert challenger_schema_for(None) is ChallengerEnvelope
    assert response_format_for(lite, _req(role="judge")) == {"type": "json_object"}
    assert response_format_for(lite, _req(role="challenger", response_schema=ChallengerEnvelope)) == {
        "type": "json_object"
    }


def test_response_format_challenger_uses_dynamic_schema_from_payload_model():
    from autosynth.domains.math_word_problems import MathProblemPayload

    lite = _Litellm(supported=True)
    schema = challenger_schema_for(MathProblemPayload)
    assert schema is not ChallengerEnvelope
    # payload tightened to the domain model's required fields.
    fields = schema.model_fields["payload"].annotation
    assert fields is MathProblemPayload
    assert response_format_for(lite, _req(role="challenger", response_schema=schema)) is schema


def test_challenger_schema_for_is_cached():
    from autosynth.domains.math_word_problems import MathProblemPayload

    assert challenger_schema_for(MathProblemPayload) is challenger_schema_for(MathProblemPayload)
    assert challenger_schema_for(None) is challenger_schema_for(None)


def test_response_format_explicit_request_schema_wins_over_role():
    # An attached (strict-compatible) response_schema takes precedence over the
    # role-based default.
    from autosynth.domains.math_word_problems import MathProblemPayload

    lite = _Litellm(supported=True)
    schema = challenger_schema_for(MathProblemPayload)
    assert response_format_for(lite, _req(role="quality", response_schema=schema)) is schema


def test_response_format_falls_back_when_support_check_raises():
    from autosynth.domains.math_word_problems import MathProblemPayload

    lite = _Litellm(supported=RuntimeError("old litellm"))
    assert response_format_for(lite, _req(role="quality")) == {"type": "json_object"}
    # A strict-compatible challenger schema still degrades when the support probe
    # itself raises (old litellm) — exercises the except path, not the guard.
    assert response_format_for(
        lite, _req(role="challenger", response_schema=challenger_schema_for(MathProblemPayload))
    ) == {"type": "json_object"}
