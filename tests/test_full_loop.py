"""End-to-end mocked loop: happy path, reject path, resume."""

from __future__ import annotations

import json
from pathlib import Path

from autodata.config import (
    AcceptanceConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autodata.orchestrator import Orchestrator


def _cfg(
    sample_docs: Path, output_dir: Path, scenario: str, *, max_rounds: int = 2, weak_zero: bool = False
) -> RunConfig:
    return RunConfig(
        run_id="test-run",
        output_dir=str(output_dir),
        max_examples=2,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(sample_docs)}),
        loop=LoopConfig(max_rounds=max_rounds, weak_samples=2, strong_samples=2),
        acceptance=AcceptanceConfig(forbid_weak_zero=weak_zero),
        orchestrator=ModelConfig(provider_model=f"mock/{scenario}"),
        challenger=ModelConfig(provider_model=f"mock/{scenario}"),
        weak_solver=ModelConfig(provider_model=f"mock/{scenario}"),
        strong_solver=ModelConfig(provider_model=f"mock/{scenario}"),
        judge=ModelConfig(provider_model=f"mock/{scenario}"),
    )


def test_happy_path_accepts(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    summary = Orchestrator(cfg).run()
    assert summary["accepted"] == 2
    assert summary.get("rejected", 0) == 0

    accepted = (output_dir / "test-run" / "accepted.jsonl").read_text().splitlines()
    assert len(accepted) == 2
    rec = json.loads(accepted[0])
    assert rec["domain"] == "qa_from_documents"
    assert rec["gap"] is not None and rec["gap"] > 0.2
    assert rec["acceptance_rationale"]


def test_reject_path_exhausts_rounds(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "reject", max_rounds=2)
    summary = Orchestrator(cfg).run()
    assert summary.get("accepted", 0) == 0
    assert summary["rejected"] == 2


def test_trajectory_written(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    Orchestrator(cfg).run()
    traj_dir = output_dir / "test-run" / "trajectories"
    files = list(traj_dir.glob("*.json"))
    assert len(files) == 2
    data = json.loads(files[0].read_text())
    assert data["domain"] == "qa_from_documents"
    assert data["total_rounds"] >= 1
    assert data["final_accepted_round"] is not None


def test_config_snapshot_written(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    Orchestrator(cfg).run()
    snap = output_dir / "test-run" / "config.snapshot.yaml"
    assert snap.exists()
    assert "qa_from_documents" in snap.read_text()


def test_resume_skips_completed(sample_docs: Path, output_dir: Path):
    cfg = _cfg(sample_docs, output_dir, "happy")
    Orchestrator(cfg).run()
    # Second run with the SAME run_id and resume=True should add no new accepted
    cfg.resume = True
    summary2 = Orchestrator(cfg).run()
    # Accepted count never decreases; previously accepted items are skipped, so it stays the same.
    assert summary2["accepted"] == 2
