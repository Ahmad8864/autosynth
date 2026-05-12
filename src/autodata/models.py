"""LLM client abstraction.

Real calls dispatch through LiteLLM by model string. A `MockLLMProvider`
returns scripted JSON responses so tests and demos run with zero API keys.

Model strings:
  - "openai/gpt-4o-mini"
  - "anthropic/claude-haiku-4-5"
  - "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
  - "ollama/llama3"
  - "openrouter/anthropic/claude-haiku-4-5"
  - "mock/<scenario>"   — handled in-process
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from autodata.config import ModelConfig
from autodata.utils import extract_json


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]
    model: str
    usage: dict[str, int] = field(default_factory=dict)


Message = dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


class LLMClient:
    """Provider-agnostic chat client.

    Routes "mock/*" model strings to an in-process MockLLMProvider registry,
    everything else through LiteLLM.
    """

    def __init__(self, cfg: ModelConfig, role: str, timeout_s: int = 60, max_retries: int = 3):
        self.cfg = cfg
        self.role = role
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @property
    def is_mock(self) -> bool:
        return self.cfg.provider_model.startswith("mock/")

    def complete(self, messages: list[Message], *, json_mode: bool = False) -> LLMResponse:
        if self.is_mock:
            return _MOCK_REGISTRY.dispatch(self.cfg.provider_model, self.role, messages)
        return self._litellm_complete(messages, json_mode=json_mode)

    def complete_json(self, messages: list[Message]) -> dict[str, Any]:
        resp = self.complete(messages, json_mode=True)
        try:
            return extract_json(resp.text)
        except ValueError as e:
            logger.warning(
                "json parse failure (role={}): {} | raw={!r}",
                self.role,
                e,
                resp.text[:300],
            )
            raise

    # ---- real provider path -------------------------------------------------

    def _litellm_complete(self, messages: list[Message], *, json_mode: bool) -> LLMResponse:
        import litellm  # lazy import; not needed for mock-only runs

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=1, max=20),
            reraise=True,
        )
        def _call() -> Any:
            kwargs: dict[str, Any] = dict(
                model=self.cfg.provider_model,
                messages=messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                top_p=self.cfg.top_p,
                timeout=self.timeout_s,
            )
            if json_mode:
                # LiteLLM passes this through to providers that support it.
                kwargs["response_format"] = {"type": "json_object"}
            kwargs.update(self.cfg.extra or {})
            return litellm.completion(**kwargs)

        resp = _call()
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        usage_dict = (
            {"prompt_tokens": getattr(usage, "prompt_tokens", 0),
             "completion_tokens": getattr(usage, "completion_tokens", 0),
             "total_tokens": getattr(usage, "total_tokens", 0)}
            if usage else {}
        )
        return LLMResponse(text=text, raw=resp.model_dump() if hasattr(resp, "model_dump") else {}, model=self.cfg.provider_model, usage=usage_dict)


# ---------------------------------------------------------------------------
# Mock provider — scripted, role-aware, deterministic
# ---------------------------------------------------------------------------

MockHandler = Callable[[str, list[Message]], str]
"""(scenario, messages) -> raw text. Should usually be JSON."""


class _MockRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, MockHandler] = {}
        self._lock = threading.Lock()

    def register(self, scenario: str, handler: MockHandler) -> None:
        with self._lock:
            self._handlers[scenario] = handler

    def dispatch(self, provider_model: str, role: str, messages: list[Message]) -> LLMResponse:
        # provider_model = "mock/<scenario>"
        scenario = provider_model.split("/", 1)[1] if "/" in provider_model else "default"
        handler = self._handlers.get(scenario) or self._handlers.get("default") or _default_handler
        text = handler(role, messages)
        return LLMResponse(text=text, raw={"mock": True, "scenario": scenario}, model=provider_model)


_MOCK_REGISTRY = _MockRegistry()


def register_mock(scenario: str, handler: MockHandler) -> None:
    """Register a scripted mock handler for a given scenario."""
    _MOCK_REGISTRY.register(scenario, handler)


def _default_handler(role: str, messages: list[Message]) -> str:
    """Built-in mock that simulates a realistic generation→accept trajectory.

    Useful for tests and the bundled demo. It returns role-appropriate JSON.
    The scripted behavior:
      - challenger emits a candidate (round-aware, gets harder over rounds)
      - quality verifier passes after round 1
      - weak solver answers shallowly (low rubric scores)
      - strong solver answers thoroughly (high rubric scores)
      - judge scores per-rubric-criterion
      - reflector emits feedback bullets
    """
    all_text = " ".join(m.get("content", "") for m in messages)
    if role == "challenger" or "ROLE:CHALLENGER" in all_text:
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
    if role == "reflector" or "ROLE:REFLECTION" in all_text:
        return json.dumps({
            "feedback": [
                "Make the question depend on a source-specific detail.",
                "Avoid framings answerable from generic knowledge.",
            ],
            "new_angle": "Target a quantitative claim or design choice unique to the source.",
        })
    # Quality and judging share the same LLMClient role="judge", so probe by content first.
    if "ROLE:QUALITY" in all_text:
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if "ROLE:JUDGE" in all_text or role == "judge":
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
    if role == "weak" or "ROLE:WEAK_SOLVER" in all_text:
        return "The source seems to be about general AI topics. It probably contributes something useful."
    if role == "strong" or "ROLE:STRONG_SOLVER" in all_text:
        return ("The source's main contribution is a method to iteratively generate training data "
                "using weak/strong solver disagreement, supported by a quality verifier and reflective "
                "recipe updates. Specifically it shows wide weak-strong gaps on accepted examples.")
    return "{}"


def _peek_round(text: str) -> int:
    import re
    m = re.search(r"ROUND[=:\s]+(\d+)", text)
    return int(m.group(1)) if m else 1


# Register the default scripted scenario.
register_mock("scripted", _default_handler)
register_mock("default", _default_handler)


# ---------------------------------------------------------------------------
# Meta-optimization demo scenario.
#
# Behavior:
#   - mutator deterministically proposes the marker rule on the FIRST iteration
#     and a different rule on later iterations (no-op refinements).
#   - judge gives the strong solver +0.20 if the marker rule appears in the
#     challenger system prompt. So once meta-opt accepts the marker, accept
#     rates improve and the loop progresses.
# ---------------------------------------------------------------------------

_MARKER_RULE = "Target a quantitative or design-specific claim unique to the source."


def _metaopt_handler(role: str, messages: list[Message]) -> str:
    import json as _json

    all_text = " ".join(m.get("content", "") for m in messages)

    if role == "meta_mutator" or "ROLE:META_MUTATOR" in all_text:
        # On every call propose the marker rule; the metaopt loop will detect a
        # repeat fingerprint and skip subsequent duplicates.
        return _json.dumps({
            "rationale": "Add source-specificity rule to widen weak/strong gap.",
            "challenger_rules_add": [_MARKER_RULE],
        })

    if role == "challenger" or "ROLE:CHALLENGER" in all_text:
        # When the marker rule is in OUR system prompt, emit a rubric tagged
        # [SPECIFIC] — that tag flows downstream into the judge's prompt and
        # changes its scoring. This simulates "better instructions → better
        # candidate → judge can tell".
        marker = _MARKER_RULE in all_text
        c1 = "[SPECIFIC] Names a source-specific contribution" if marker else "Names contribution"
        return _json.dumps({
            "payload": {"question": "What specific contribution does this source make?",
                        "context": "Synthetic context.", "reasoning_skills": ["synthesis"]},
            "reference_output": "The contribution as named in the source.",
            "rubric": [
                {"id": "c1", "description": c1, "weight": 5},
                {"id": "c2", "description": "Cites a detail from the source", "weight": 3},
            ],
        })
    if role == "reflector" or "ROLE:REFLECTION" in all_text:
        return _json.dumps({"feedback": ["push for source-specificity"], "new_angle": "quantitative claim"})
    if "ROLE:QUALITY" in all_text:
        return _json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if "ROLE:JUDGE" in all_text or role == "judge":
        # Detect [SPECIFIC] in the rubric (the judge prompt embeds the rubric).
        specific = "[SPECIFIC]" in all_text
        if "[solver=weak]" in all_text:
            return _json.dumps({"per_criterion": {"c1": 0.25, "c2": 0.10}, "total": 0.20, "failure_modes": []})
        return _json.dumps({
            "per_criterion": {"c1": 0.95 if specific else 0.55,
                              "c2": 0.7 if specific else 0.55},
            "total": 0.85 if specific else 0.55,
            "failure_modes": [] if specific else ["generic_response"],
        })
    if role == "weak" or "ROLE:WEAK" in all_text:
        return "generic weak attempt"
    if role == "strong" or "ROLE:STRONG" in all_text:
        return "specific strong attempt"
    return "{}"


register_mock("metaopt", _metaopt_handler)
