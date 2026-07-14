"""End-to-end runner tests using in-process providers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from autosynth.config import (
    AcceptanceConfig,
    DispatcherConfig,
    DomainConfig,
    LoopConfig,
    MetaOptConfig,
    ModelConfig,
    RunConfig,
)
from autosynth.llm import register_mock
from autosynth.runner import Runner, _build_llm_config
from autosynth.store import Store


def _cfg(docs_dir: Path, output_dir: Path, scenario: str, *, forbid_weak_zero: bool = False) -> RunConfig:
    return RunConfig(
        run_id="test-run",
        output_dir=str(output_dir),
        max_examples=2,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(docs_dir)}),
        loop=LoopConfig(max_rounds=2, weak_samples=2, strong_samples=2),
        acceptance=AcceptanceConfig(forbid_weak_zero=forbid_weak_zero),
        orchestrator=ModelConfig(provider_model=f"mock/{scenario}"),
        challenger=ModelConfig(provider_model=f"mock/{scenario}"),
        weak_solver=ModelConfig(provider_model=f"mock/{scenario}"),
        strong_solver=ModelConfig(provider_model=f"mock/{scenario}"),
        judge=ModelConfig(provider_model=f"mock/{scenario}"),
        dispatcher=DispatcherConfig(concurrency=4, items_per_advance=10, poll_interval_s=0.0),
    )


def test_happy_path_accepts(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    runner = Runner(cfg)
    summary = runner.run()
    assert summary.accepted == 2
    assert summary.rejected == 0

    store = Store(runner.run_dir / "run.db")
    assert store.count_accepted("test-run") == 2


def test_runner_closes_store_after_run(sample_docs: Path, output_dir: Path):
    """A completed runner closes its store connection."""
    runner = Runner(_cfg(sample_docs, output_dir, "happy"))
    runner.run()
    with pytest.raises(sqlite3.ProgrammingError):
        runner.store.conn.execute("SELECT 1")
    runner.close()


def test_reject_path_exhausts_rounds(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "reject")
    runner = Runner(cfg)
    summary = runner.run()
    assert summary.accepted == 0
    assert summary.rejected == 2


def test_accepted_records_have_metadata(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    runner = Runner(cfg)
    runner.run()

    store = Store(runner.run_dir / "run.db")
    records = list(store.accepted_records("test-run"))
    assert len(records) == 2
    for rec in records:
        assert rec["domain"] == "qa_from_documents"
        assert rec["gap"] is not None and rec["gap"] > 0.2
        assert rec["acceptance_rationale"]
        assert rec["weak_avg"] is not None
        assert rec["strong_avg"] is not None


def test_config_snapshot_written(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    runner = Runner(cfg)
    runner.run()
    snap = runner.run_dir / "config.snapshot.yaml"
    assert snap.exists()
    assert "qa_from_documents" in snap.read_text()


def test_resume_skips_completed(sample_docs: Path, output_dir: Path, monkeypatch):
    """Reopening a completed run should do no work."""
    cfg = _cfg(sample_docs, output_dir, "happy")
    Runner(cfg).run()

    db = output_dir / "test-run" / "run.db"
    store = Store(db)
    before = tuple(
        store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("requests", "responses", "rounds", "accepted")
    )
    store.close()

    def unexpected_completion(*_args, **_kwargs):
        raise AssertionError("a completed run called the provider during resume")

    monkeypatch.setattr("autosynth.llm.client.LLMClient.complete", unexpected_completion)
    cfg.resume = True
    summary2 = Runner(cfg, run_id="test-run").run()
    assert summary2.accepted == 2

    store = Store(db)
    after = tuple(
        store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("requests", "responses", "rounds", "accepted")
    )
    store.close()
    assert after == before


def test_build_llm_config_collects_model_extras(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    cfg.weak_solver = ModelConfig(
        provider_model="azure/weak-dep",
        extra={"api_base": "https://weak.example", "api_version": "2024-02-01"},
    )
    cfg.strong_solver = ModelConfig(
        provider_model="azure/strong-dep",
        extra={"api_base": "https://strong.example"},
    )
    # Roles sharing a model contribute to the same provider options.
    cfg.judge = ModelConfig(
        provider_model="azure/shared",
        extra={"api_base": "https://shared.example"},
    )
    cfg.challenger = ModelConfig(
        provider_model="azure/shared",
        extra={"api_version": "2024-02-01"},
    )
    cfg.metaopt = MetaOptConfig(
        mutator=ModelConfig(provider_model="azure/mutator", extra={"api_base": "https://mutator.example"}),
    )

    llm_cfg = _build_llm_config(cfg)
    assert llm_cfg.model_extras["azure/weak-dep"] == {
        "api_base": "https://weak.example",
        "api_version": "2024-02-01",
    }
    assert llm_cfg.model_extras["azure/strong-dep"] == {"api_base": "https://strong.example"}
    assert llm_cfg.model_extras["azure/shared"] == {
        "api_base": "https://shared.example",
        "api_version": "2024-02-01",
    }
    assert llm_cfg.model_extras["azure/mutator"] == {"api_base": "https://mutator.example"}


def test_build_llm_config_omits_models_with_no_extras(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    llm_cfg = _build_llm_config(cfg)
    assert llm_cfg.model_extras == {}


def test_each_role_request_carries_its_configured_temperature(
    sample_docs: Path,
    output_dir: Path,
):
    """Persist each role's own temperature without a challenger fallback."""
    cfg = _cfg(sample_docs, output_dir, "happy")
    # Unique values make role cross-wiring visible.
    cfg.challenger = ModelConfig(provider_model="mock/happy", temperature=0.81)
    cfg.weak_solver = ModelConfig(provider_model="mock/happy", temperature=0.72)
    cfg.strong_solver = ModelConfig(provider_model="mock/happy", temperature=0.33)
    cfg.judge = ModelConfig(provider_model="mock/happy", temperature=0.04)
    cfg.orchestrator = ModelConfig(provider_model="mock/happy", temperature=0.55)

    Runner(cfg).run()

    store = Store(output_dir / "test-run" / "run.db")
    rows = store.conn.execute("SELECT role, temperature FROM requests").fetchall()
    by_role: dict[str, set[float | None]] = {}
    for row in rows:
        by_role.setdefault(row["role"], set()).add(row["temperature"])

    # Quality uses judge settings; reflection uses orchestrator settings.
    expected = {
        "challenger": 0.81,
        "quality": 0.04,
        "weak": 0.72,
        "strong": 0.33,
        "judge": 0.04,
        "reflector": 0.55,
    }
    # The first-round happy path never emits a reflector request.
    fired = set(by_role)
    assert {"challenger", "quality", "weak", "strong", "judge"}.issubset(fired)
    for role in fired:
        want = expected[role]
        got = by_role[role]
        assert got == {want}, f"role={role!r} expected temperature {want}, got {got}"


# Verifiable mode end-to-end (math, programmatic verify(), no judge)


def _verifiable_math_handler(role: str, messages):
    all_text = " ".join(m.get("content", "") for m in messages)
    if "ROLE:CHALLENGER" in all_text or role == "challenger":
        return json.dumps(
            {
                "payload": {"problem": "What is 6 times 7?", "topic": "arithmetic", "difficulty": "easy"},
                "reference_output": "42",
                "rubric": [{"id": "c1", "description": "correct final answer", "weight": 7}],
            }
        )
    if "ROLE:QUALITY" in all_text:
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if role == "weak":
        return "I'll guess. ANSWER: 41"
    if role == "strong":
        return "6 * 7 = 42. ANSWER: 42"
    return "{}"


register_mock("verifiable_math", _verifiable_math_handler)


def _math_cfg(output_dir: Path) -> RunConfig:
    return RunConfig(
        run_id="math-run",
        output_dir=str(output_dir),
        max_examples=2,
        domain=DomainConfig(
            name="math_word_problems",
            params={
                "topics": [
                    {"topic": "arithmetic", "difficulty": "easy"},
                    {"topic": "more", "difficulty": "easy"},
                ]
            },
        ),
        loop=LoopConfig(max_rounds=2, weak_samples=4, strong_samples=4),
        acceptance=AcceptanceConfig(
            mode="verifiable", verifiable_weak_max_correct=1, verifiable_strong_min_correct=3
        ),
        orchestrator=ModelConfig(provider_model="mock/verifiable_math"),
        challenger=ModelConfig(provider_model="mock/verifiable_math"),
        weak_solver=ModelConfig(provider_model="mock/verifiable_math"),
        strong_solver=ModelConfig(provider_model="mock/verifiable_math"),
        judge=ModelConfig(provider_model="mock/verifiable_math"),
        dispatcher=DispatcherConfig(concurrency=4, items_per_advance=10, poll_interval_s=0.0),
    )


def test_verifiable_math_e2e(output_dir: Path):
    runner = Runner(_math_cfg(output_dir))
    summary = runner.run()
    assert summary.accepted == 2 and summary.rejected == 0

    store = Store(runner.run_dir / "run.db")
    by_solver: dict[str, list] = {}
    for r in store.conn.execute("SELECT solver, correct, total FROM solver_scores"):
        by_solver.setdefault(r["solver"], []).append((r["correct"], r["total"]))
    assert by_solver["strong"] and all(c == 1 and t == 1.0 for c, t in by_solver["strong"])
    assert by_solver["weak"] and all(c == 0 and t == 0.0 for c, t in by_solver["weak"])
    assert store.conn.execute("SELECT COUNT(*) FROM requests WHERE role='judge'").fetchone()[0] == 0


# Judge-decided mode end-to-end (loop-judge accept, NEED_SCORES → NEED_DECISION)


def _judge_loop_handler(role: str, messages):
    all_text = " ".join(m.get("content", "") for m in messages)
    if "ROLE:CHALLENGER" in all_text or role == "challenger":
        return json.dumps(
            {
                "payload": {"question": "What is the contribution?", "context": "ctx"},
                "reference_output": "the contribution",
                "rubric": [{"id": "c1", "description": "names it", "weight": 5}],
            }
        )
    if "ROLE:QUALITY" in all_text:
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if "ROLE:LOOP_JUDGE" in all_text or role == "loop_judge":
        return json.dumps(
            {"verdict": "accept", "grpo_suitability": "high", "reason": "clean gap", "suggestion": ""}
        )
    if "ROLE:JUDGE" in all_text or role == "judge":
        total = 0.2 if "SOLVER_RESPONSE: surface-level answer" in all_text else 0.9
        return json.dumps({"per_criterion": {"c1": total}, "total": total, "failure_modes": []})
    if role == "weak":
        return "surface-level answer"
    if role == "strong":
        return "detailed, source-grounded answer"
    return "{}"


register_mock("judge_loop", _judge_loop_handler)


def test_judge_policy_e2e(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "judge_loop")
    cfg.acceptance = AcceptanceConfig(mode="judge")
    runner = Runner(cfg, run_id="test-run")
    summary = runner.run()
    assert summary.accepted == 2 and summary.rejected == 0

    store = Store(runner.run_dir / "run.db")
    # Each accepted item passed through the asynchronous loop judge.
    assert store.conn.execute("SELECT COUNT(*) FROM requests WHERE role='loop_judge'").fetchone()[0] == 2
