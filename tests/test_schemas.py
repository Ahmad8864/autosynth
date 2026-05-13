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


def test_candidate_roundtrip():
    c = Candidate(
        candidate_id="x",
        domain="d",
        source_id="s",
        payload={"q": "?"},
        rubric=[RubricCriterion(id="c1", description="x", weight=2)],
        reference_output="r",
    )
    s = c.model_dump_json()
    c2 = Candidate.model_validate_json(s)
    assert c2.candidate_id == "x"
    assert c2.rubric[0].weight == 2


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
    assert t.accepted_round().refinement_round == 2
