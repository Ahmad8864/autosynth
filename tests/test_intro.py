"""Run intro banner."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from autosynth._intro import render_run_intro
from autosynth.config import DomainConfig, ModelConfig, RunConfig


def _render(panel) -> str:
    con = Console(width=200)
    with con.capture() as cap:
        con.print(panel)
    return cap.get()


def test_intro_summarizes_params():
    cfg = RunConfig(
        domain=DomainConfig(name="qa_from_documents"),
        strong_solver=ModelConfig(provider_model="openai/gpt-4o"),
        budget_usd=5.0,
    )
    out = _render(render_run_intro(cfg, run_id="run-abc", run_dir=Path("outputs/run-abc")))
    assert "autosynth" in out
    assert "agentic synthetic data" in out
    assert "run-abc" in out
    assert "qa_from_documents" in out
    assert "openai/gpt-4o" in out  # strong solver differs → per-role list
    assert "mock/scripted" in out  # the other roles' default
    assert "$5.00" in out
    assert "resuming" not in out


def test_intro_collapses_shared_model_and_flags_resume():
    cfg = RunConfig(domain=DomainConfig(name="qa_from_documents"))
    out = _render(render_run_intro(cfg, run_id="r", run_dir=Path("outputs/r"), resume=True))
    assert out.count("mock/scripted") == 1  # all roles share one model → single line
    assert "unlimited" in out  # budget_usd is None
    assert "resuming" in out
