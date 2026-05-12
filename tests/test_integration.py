"""End-to-end mocked integration tests against the Runner.

Replaces the legacy ``test_full_loop.py``; same assertions, new architecture.
The full pipeline runs against the in-process mock provider — zero API keys.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodata.config import (
    AcceptanceConfig,
    DispatcherConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autodata.runner import Runner
from autodata.store import Store


def _cfg(docs_dir: Path, output_dir: Path, scenario: str, *,
         forbid_weak_zero: bool = False) -> RunConfig:
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


def test_resume_skips_completed(sample_docs: Path, output_dir: Path):
    """Second Runner with same run_id finds all items already terminal — no work."""
    cfg = _cfg(sample_docs, output_dir, "happy")
    Runner(cfg).run()
    # Reuse the same run dir and id.
    cfg.resume = True
    summary2 = Runner(cfg, run_id="test-run").run()
    assert summary2.accepted == 2


def test_export_jsonl_via_store(sample_docs: Path, output_dir: Path, tmp_path: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    runner = Runner(cfg)
    runner.run()
    store = Store(runner.run_dir / "run.db")
    out = tmp_path / "out.jsonl"
    n = store.export_jsonl("test-run", out)
    assert n == 2
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["gap"] is not None for r in records)
