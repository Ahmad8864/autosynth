from autosynth.schemas import (
    Candidate,
    EvalReport,
    QualityCheck,
    RubricCriterion,
    Trajectory,
)


def test_rubric_weight_validation():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RubricCriterion(id="c1", description="x", weight=0)
    c = RubricCriterion(id="c1", description="x", weight=5)
    assert c.weight == 5


def test_trajectory_accepted_round_lookup():
    from autosynth.schemas import Round

    cand = Candidate(
        candidate_id="x",
        domain="d",
        source_id="s",
        payload={"q": "?"},
        rubric=[RubricCriterion(id="c1", description="x", weight=1)],
    )
    t = Trajectory(trajectory_id="t", run_id="r", domain="d", source_id="s")
    t.rounds.append(
        Round(refinement_round=1, candidate=cand, quality=QualityCheck(passed=True), evaluation=EvalReport())
    )
    t.rounds.append(
        Round(
            refinement_round=2,
            candidate=cand,
            quality=QualityCheck(passed=True),
            evaluation=EvalReport(accepted=True),
        )
    )
    t.final_accepted_round = 2
    accepted = t.accepted_round()
    assert accepted is not None and accepted.refinement_round == 2
