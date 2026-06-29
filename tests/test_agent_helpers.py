"""Tests for the additive module-level agent helpers used by the new pipeline."""

from __future__ import annotations

import json

import pytest

from autosynth.agents import challenger, reflector, solver, verifier
from autosynth.domains.qa_from_documents import QAFromDocuments
from autosynth.harness import make_harness
from autosynth.schemas import (
    Candidate,
    EvalReport,
    QualityCheck,
    Round,
    RubricCriterion,
)
from autosynth.utils import extract_json


@pytest.fixture
def domain(tmp_path):
    (tmp_path / "x.md").write_text("body")
    return QAFromDocuments(source_dir=str(tmp_path))


@pytest.fixture
def grounding(domain):
    return next(iter(domain.load_grounding()))


@pytest.fixture
def candidate():
    return Candidate(
        candidate_id="c1",
        domain="qa",
        source_id="s1",
        payload={"question": "Q?", "context": "ctx"},
        rubric=[
            RubricCriterion(id="c1", description="x", weight=5),
            RubricCriterion(id="c2", description="y", weight=3),
        ],
        reference_output="ref",
    )


# ---------------------------------------------------------------------------
# challenger
# ---------------------------------------------------------------------------


def test_challenger_build_request_is_deterministic(domain, grounding):
    req1 = challenger.build_request(
        item_id="i1",
        round_n=1,
        model_key="mock/foo",
        grounding=grounding,
        feedback=[],
        prior_payloads=[],
        domain=domain,
    )
    req2 = challenger.build_request(
        item_id="i1",
        round_n=1,
        model_key="mock/foo",
        grounding=grounding,
        feedback=[],
        prior_payloads=[],
        domain=domain,
    )
    assert req1.request_id == req2.request_id
    assert req1.role == "challenger"
    assert req1.json_mode is True


def test_challenger_build_request_injects_harness_rules(domain, grounding):
    h = make_harness(challenger_rules=["UNIQUE_MARKER_XYZ"])
    req = challenger.build_request(
        item_id="i1",
        round_n=1,
        model_key="mock/foo",
        grounding=grounding,
        feedback=[],
        prior_payloads=[],
        domain=domain,
        harness=h,
    )
    all_text = " ".join(m["content"] for m in req.messages)
    assert "UNIQUE_MARKER_XYZ" in all_text


def test_challenger_parse_response_caps_rubric_weight():
    text = json.dumps(
        {
            "payload": {"question": "Q?"},
            "reference_output": "ref",
            "rubric": [{"id": "c1", "description": "x", "weight": 99}],  # > cap
        }
    )
    cand = challenger.parse_response(text, source_id="s1", round_n=1, domain_name="qa", rubric_max_weight=7)
    assert cand.rubric[0].weight == 7


def test_challenger_parse_response_handles_garbage_rubric_weights():
    text = json.dumps(
        {
            "payload": {"question": "Q?"},
            "reference_output": "r",
            "rubric": [{"id": "c1", "description": "x", "weight": "huh?"}],
        }
    )
    cand = challenger.parse_response(text, source_id="s1", round_n=1, domain_name="qa")
    assert cand.rubric[0].weight == 1  # defaulted


def test_challenger_parse_response_tolerates_string_rubric_items():
    # Wrong inner shape (rubric of bare strings) must coerce, not raise (C1 vector b).
    text = json.dumps(
        {"payload": {"question": "Q?"}, "reference_output": "r", "rubric": ["correctness", "clarity"]}
    )
    cand = challenger.parse_response(text, source_id="s1", round_n=1, domain_name="qa")
    assert [c.description for c in cand.rubric] == ["correctness", "clarity"]
    assert all(c.weight == 1 for c in cand.rubric)


def test_extract_json_rejects_non_object():
    # Non-object JSON breaks the dict contract callers rely on; reject it (C1 vector a).
    for bad in ('["correctness", "clarity"]', "42", '"accept"', "true", "null"):
        with pytest.raises(ValueError):
            extract_json(bad)


def test_extract_json_salvages_embedded_object():
    assert extract_json('prose [1, 2] then {"a": 1} tail') == {"a": 1}
    assert extract_json('Here you go: {"x": {"y": 2}}') == {"x": {"y": 2}}


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


def test_solver_build_request_role_validation(domain, candidate):
    with pytest.raises(ValueError):
        solver.build_request(
            item_id="i1",
            round_n=1,
            attempt=0,
            model_key="mock/x",
            candidate=candidate,
            role="judge",
            domain=domain,
        )


def test_solver_build_request_distinct_ids_per_attempt(domain, candidate):
    a0 = solver.build_request(
        item_id="i1", round_n=1, attempt=0, model_key="m", candidate=candidate, role="weak", domain=domain
    )
    a1 = solver.build_request(
        item_id="i1", round_n=1, attempt=1, model_key="m", candidate=candidate, role="weak", domain=domain
    )
    sweak = solver.build_request(
        item_id="i1", round_n=1, attempt=0, model_key="m", candidate=candidate, role="strong", domain=domain
    )
    ids = {a0.request_id, a1.request_id, sweak.request_id}
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# verifier (quality + judge)
# ---------------------------------------------------------------------------


def test_quality_build_request_is_json_mode(domain, candidate):
    req = verifier.build_quality_request(
        item_id="i1", round_n=1, model_key="m", candidate=candidate, domain=domain
    )
    assert req.json_mode is True
    assert req.role == "quality"


def test_quality_parse_passes_through():
    qc = verifier.parse_quality(json.dumps({"passed": True, "failures": [], "notes": "ok"}))
    assert qc.passed is True
    assert qc.notes == "ok"


def test_quality_parse_handles_bad_json():
    qc = verifier.parse_quality("not json")
    assert qc.passed is False
    assert any("parse_error" in f for f in qc.failures)


def test_judge_build_request_keyed_on_parent_id(domain, candidate):
    j1 = verifier.build_judge_request(
        item_id="i1",
        round_n=1,
        attempt=0,
        model_key="m",
        candidate=candidate,
        solver_response="resp-a",
        solver_role="weak",
        domain=domain,
        parent_response_id="parent-A",
    )
    j2 = verifier.build_judge_request(
        item_id="i1",
        round_n=1,
        attempt=0,
        model_key="m",
        candidate=candidate,
        solver_response="resp-b",
        solver_role="weak",
        domain=domain,
        parent_response_id="parent-B",
    )
    assert j1.request_id != j2.request_id
    assert j1.parent_response_id == "parent-A"
    assert j2.parent_response_id == "parent-B"


def test_judge_parse_uses_weighted_average_when_total_missing(candidate):
    text = json.dumps({"per_criterion": {"c1": 1.0, "c2": 0.0}})
    s = verifier.parse_judge(
        text, candidate=candidate, solver_role="strong", attempt=0, solver_response_text="x"
    )
    # weighted: (5*1.0 + 3*0.0) / 8 = 0.625
    assert s.total == pytest.approx(0.625)


def test_judge_parse_clamps_and_drops_bad_per_criterion(candidate):
    text = json.dumps({"per_criterion": {"c1": 1.5, "c2": "garbage"}, "total": 0.4})
    s = verifier.parse_judge(
        text, candidate=candidate, solver_role="weak", attempt=0, solver_response_text="x"
    )
    assert s.per_criterion["c1"] == 1.0  # clamped
    assert "c2" not in s.per_criterion  # dropped
    assert s.total == 0.4  # explicit total wins


# ---------------------------------------------------------------------------
# reflector
# ---------------------------------------------------------------------------


def test_reflector_build_request_summarizes_priors(candidate):
    rounds = [
        Round(
            refinement_round=1,
            candidate=candidate,
            quality=QualityCheck(passed=True),
            evaluation=EvalReport(
                weak_avg=0.8, strong_avg=0.9, gap=0.1, accepted=False, rejection_reasons=["weak_avg too high"]
            ),
        ),
    ]
    req = reflector.build_request(
        item_id="i1",
        round_n=2,
        model_key="m",
        prior_rounds=rounds,
        domain_name="qa",
        leakage_rules=["x"],
        weak_ceiling=0.65,
        strong_floor=0.60,
    )
    all_text = " ".join(m["content"] for m in req.messages)
    assert "TOO_EASY" in all_text
    assert "weak_avg" in all_text
    assert req.role == "reflector"


def test_reflector_parse_response_ok():
    text = json.dumps({"feedback": ["a", "b"], "new_angle": "X"})
    res = reflector.parse_response(text)
    assert res.feedback == ["a", "b"]
    assert res.new_angle == "X"


def test_reflector_parse_response_fallback_on_garbage():
    res = reflector.parse_response("not json")
    assert res.feedback
    assert res.new_angle == ""
