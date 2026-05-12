import json
from pathlib import Path

from autodata.config import DomainConfig, RunConfig
from autodata.schemas import (
    Candidate,
    EvalReport,
    QualityCheck,
    Round,
    RubricCriterion,
    SolverScore,
    Trajectory,
)
from autodata.writer import RunWriter, build_accepted_record


def _trajectory():
    cand = Candidate(
        candidate_id="c1", domain="qa_from_documents", source_id="s1", payload={"q": "?"},
        rubric=[RubricCriterion(id="c1", description="x", weight=3)],
        reference_output="ans",
    )
    ev = EvalReport(
        weak_scores=[SolverScore(solver="weak", attempt=0, raw_response="x", total=0.2)],
        strong_scores=[SolverScore(solver="strong", attempt=0, raw_response="y", total=0.9)],
        weak_avg=0.2, weak_max=0.2, weak_min=0.2,
        strong_avg=0.9, strong_max=0.9, strong_min=0.9,
        gap=0.7, accepted=True, acceptance_rationale="ok",
    )
    return Trajectory(
        trajectory_id="t1", run_id="r1", domain="qa_from_documents", source_id="s1",
        rounds=[Round(refinement_round=1, candidate=cand, quality=QualityCheck(passed=True), evaluation=ev)],
        total_rounds=1, final_accepted_round=1,
    )


def test_writer_round_trip(tmp_path: Path):
    cfg = RunConfig(
        run_id="r1",
        output_dir=str(tmp_path),
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(tmp_path)}),
    )
    writer = RunWriter(cfg, "r1")
    traj = _trajectory()
    writer.write_trajectory(traj)
    loaded = writer.load_trajectory("s1")
    assert loaded is not None
    assert loaded.final_accepted_round == 1


def test_accepted_record_shape(tmp_path: Path):
    cfg = RunConfig(
        run_id="r1",
        output_dir=str(tmp_path),
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(tmp_path)}),
    )
    writer = RunWriter(cfg, "r1")
    from autodata.domain import build_domain
    domain = build_domain("qa_from_documents", None, {"source_dir": str(tmp_path)})
    rec = build_accepted_record(domain=domain, trajectory=_trajectory(), extra={})
    writer.write_accepted(rec)
    lines = (tmp_path / "r1" / "accepted.jsonl").read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["gap"] == 0.7
    assert data["domain"] == "qa_from_documents"
    assert data["acceptance_rationale"]
