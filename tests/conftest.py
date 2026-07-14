"""Shared fixtures: mock LLM scenarios, sample docs, isolated output dirs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from autosynth.llm import register_mock


@pytest.fixture
def sample_docs(tmp_path: Path) -> Path:
    """A tiny doc directory used by qa_from_documents."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# Doc A\nThis document talks about topic A in detail.")
    (d / "b.md").write_text("# Doc B\nThis document covers topic B with examples.")
    return d


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "outputs"


@pytest.fixture
def script_mock() -> Callable[[str, Callable], None]:
    """Helper: register a fresh mock scenario for a single test."""

    def _register(scenario: str, handler):
        register_mock(scenario, handler)

    return _register


# Mock that accepts on the first round


def _happy_handler(role: str, messages):
    all_text = " ".join(m.get("content", "") for m in messages)
    if role == "challenger" or "ROLE:CHALLENGER" in all_text:
        return json.dumps(
            {
                "payload": {
                    "question": "What is the contribution?",
                    "context": "ctx",
                    "reasoning_skills": ["x"],
                },
                "reference_output": "the contribution",
                "rubric": [
                    {"id": "c1", "description": "Names contribution", "weight": 5},
                    {"id": "c2", "description": "Cites detail", "weight": 3},
                ],
            }
        )
    if role == "reflector" or "ROLE:REFLECTION" in all_text:
        return json.dumps({"feedback": ["try harder"], "new_angle": "x"})
    # Quality and scoring share the judge client role.
    if "ROLE:QUALITY" in all_text:
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if "ROLE:JUDGE" in all_text or role == "judge":
        if "vague answer" in all_text:
            return json.dumps({"per_criterion": {"c1": 0.2, "c2": 0.1}, "total": 0.16, "failure_modes": []})
        return json.dumps({"per_criterion": {"c1": 0.9, "c2": 0.85}, "total": 0.88, "failure_modes": []})
    if role == "weak":
        return "vague answer"
    if role == "strong":
        return "specific, source-grounded answer"
    return "{}"


register_mock("happy", _happy_handler)


# Mock that always rejects


def _reject_handler(role: str, messages):
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if "ROLE:CHALLENGER" in last_user:
        return json.dumps(
            {
                "payload": {"question": "q", "context": "c"},
                "reference_output": "r",
                "rubric": [{"id": "c1", "description": "d", "weight": 3}],
            }
        )
    if "ROLE:QUALITY" in last_user:
        return json.dumps({"passed": False, "failures": ["context_leaks_answer"], "notes": "bad"})
    if "ROLE:REFLECTION" in last_user:
        return json.dumps({"feedback": ["fix leakage"], "new_angle": "different"})
    return "{}"


register_mock("reject", _reject_handler)
