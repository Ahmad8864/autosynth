"""Scripted, role-aware mock provider for tests and demos.

Mock scenarios register a handler keyed by the second segment of a ``mock/*``
model string. Two scenarios ship by default:

  - ``mock/scripted`` / ``mock/default`` — a realistic generation→accept trajectory
  - ``mock/metaopt`` — used by the meta-optimization demo and tests
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable

from loguru import logger

from autodata.llm.types import Message

__all__ = [
    "MockHandler",
    "register_mock",
    "dispatch_mock",
    "_canonical_role",
    "_join_messages",
]


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
                    "mock scenario {!r} not registered; falling back to 'default'",
                    scenario,
                )
            handler = self._handlers.get("default") or _default_mock_handler
        return handler(role, messages)


_MOCK_REGISTRY = _MockRegistry()


def register_mock(scenario: str, handler: MockHandler) -> None:
    """Register a scripted mock handler. Reachable via `mock/<scenario>` model strings."""
    _MOCK_REGISTRY.register(scenario, handler)


def dispatch_mock(provider_model: str, role: str, messages: list[Message]) -> str:
    return _MOCK_REGISTRY.dispatch(provider_model, role, messages)


# ---------------------------------------------------------------------------
# Default mock scenarios
# ---------------------------------------------------------------------------


def _default_mock_handler(role: str, messages: list[Message]) -> str:
    """Built-in mock simulating a realistic generation→accept trajectory."""
    all_text = _join_messages(messages)
    canon = _canonical_role(role, all_text)

    if canon == "challenger":
        round_n = _peek_round(all_text)
        return json.dumps(
            {
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
            }
        )
    if canon == "reflector":
        return json.dumps(
            {
                "feedback": [
                    "Make the question depend on a source-specific detail.",
                    "Avoid framings answerable from generic knowledge.",
                ],
                "new_angle": "Target a quantitative claim or design choice unique to the source.",
            }
        )
    if canon == "quality":
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if canon == "judge":
        solver_tag = "weak" if "[solver=weak]" in all_text else "strong"
        if solver_tag == "weak":
            return json.dumps(
                {
                    "per_criterion": {"c1": 0.2, "c2": 0.0, "c3": 0.1},
                    "total": 0.13,
                    "failure_modes": ["generic_response"],
                }
            )
        return json.dumps(
            {
                "per_criterion": {"c1": 0.95, "c2": 0.85, "c3": 0.8},
                "total": 0.88,
                "failure_modes": [],
            }
        )
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
        return json.dumps(
            {
                "rationale": "Add source-specificity rule to widen weak/strong gap.",
                "challenger_rules_add": [_MARKER_RULE],
            }
        )

    if canon == "challenger":
        marker = _MARKER_RULE in all_text
        c1 = "[SPECIFIC] Names a source-specific contribution" if marker else "Names contribution"
        return json.dumps(
            {
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
            }
        )
    if canon == "reflector":
        return json.dumps({"feedback": ["push for source-specificity"], "new_angle": "quantitative claim"})
    if canon == "quality":
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if canon == "judge":
        specific = "[SPECIFIC]" in all_text
        if "[solver=weak]" in all_text:
            return json.dumps({"per_criterion": {"c1": 0.25, "c2": 0.10}, "total": 0.20, "failure_modes": []})
        return json.dumps(
            {
                "per_criterion": {"c1": 0.95 if specific else 0.55, "c2": 0.7 if specific else 0.55},
                "total": 0.85 if specific else 0.55,
                "failure_modes": [] if specific else ["generic_response"],
            }
        )
    if canon == "weak":
        return "generic weak attempt"
    if canon == "strong":
        return "specific strong attempt"
    return "{}"


register_mock("metaopt", _metaopt_handler)
