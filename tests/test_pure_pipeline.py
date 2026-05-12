"""Exhaustive state-transition tests for the pure pipeline.

Every state has at least one happy-path test and at least one rejection
path. The partial-completion invariant (MIGRATION_PLAN.md §2.3) is the
non-negotiable test: violating it would cause double-emission of requests
on the next dispatcher loop iteration.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodata.config import (
    AcceptanceConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autodata.domain import GroundingItem
from autodata.domains.qa_from_documents import QAFromDocuments
from autodata.harness import DEFAULT_HARNESS
from autodata.pipeline import (
    ItemState,
    State,
    StepResponse,
    step,
)
from autodata.safety import SafetyVerdict
from autodata.utils import stable_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def domain(tmp_path: Path):
    (tmp_path / "doc.md").write_text("body of doc")
    return QAFromDocuments(source_dir=str(tmp_path))


@pytest.fixture
def grounding(domain) -> GroundingItem:
    return next(iter(domain.load_grounding()))


def _cfg(*, max_rounds: int = 3, weak: int = 2, strong: int = 2,
        safety_enabled: bool = False) -> RunConfig:
    return RunConfig(
        run_id="r1",
        output_dir="/tmp/out",
        max_examples=1,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": "/tmp"}),
        loop=LoopConfig(max_rounds=max_rounds, weak_samples=weak, strong_samples=strong),
        acceptance=AcceptanceConfig(forbid_weak_zero=False),
        orchestrator=ModelConfig(provider_model="mock/x"),
        challenger=ModelConfig(provider_model="mock/x"),
        weak_solver=ModelConfig(provider_model="mock/x"),
        strong_solver=ModelConfig(provider_model="mock/x"),
        judge=ModelConfig(provider_model="mock/x"),
    )


def _seed_item(grounding: GroundingItem, *, state: State = State.PENDING,
               current_round: int = 1, **overrides) -> ItemState:
    base = ItemState(
        item_id="i1", run_id="r1", source_id=grounding.source_id,
        domain="qa_from_documents", state=state, current_round=current_round,
        source_metadata=grounding.metadata,
    )
    return base if not overrides else type(base)(**{**base.__dict__, **overrides})


def _challenger_text(**overrides) -> str:
    payload = {
        "question": "What does the source say?",
        "context": "synthetic ctx",
    }
    rubric = [
        {"id": "c1", "description": "names contribution", "weight": 5},
        {"id": "c2", "description": "cites detail", "weight": 3},
    ]
    body = {"payload": payload, "reference_output": "reference answer", "rubric": rubric}
    body.update(overrides)
    return json.dumps(body)


def _solver_resp(role: str, attempt: int, *, round_n: int = 1) -> StepResponse:
    return StepResponse(
        request_id=stable_id("i1", round_n, role, attempt),
        role=role, round_n=round_n, attempt=attempt,
        text=f"{role}-attempt-{attempt}",
    )


def _judge_resp(*, solver_role: str, attempt: int, total: float,
                round_n: int = 1, solver_text: str = "x") -> StepResponse:
    parent_id = stable_id("i1", round_n, solver_role, attempt)
    return StepResponse(
        request_id=stable_id("i1", round_n, "judge", solver_role, parent_id),
        role="judge", round_n=round_n, attempt=attempt,
        text=json.dumps({"per_criterion": {"c1": total, "c2": total}, "total": total}),
        parent_response_id=parent_id,
        solver_response_text=solver_text,
        solver_role=solver_role,
    )


# ---------------------------------------------------------------------------
# PENDING → NEED_CANDIDATE
# ---------------------------------------------------------------------------

def test_pending_emits_challenger_request(domain, grounding):
    item = _seed_item(grounding)
    res = step(item, [], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_CANDIDATE
    assert len(res.new_requests) == 1
    assert res.new_requests[0].role == "challenger"


# ---------------------------------------------------------------------------
# NEED_CANDIDATE
# ---------------------------------------------------------------------------

def test_need_candidate_advances_to_need_quality_on_valid_response(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    resp = StepResponse(
        request_id=stable_id("i1", 1, "challenger", 0),
        role="challenger", round_n=1, attempt=0, text=_challenger_text(),
    )
    res = step(item, [resp], cfg=_cfg(), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_QUALITY
    assert res.state.candidate is not None
    assert len(res.new_requests) == 1
    assert res.new_requests[0].role == "quality"


def test_need_candidate_partial_responses_noop(domain, grounding):
    """The challenger hasn't responded yet — step() must be a noop."""
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    res = step(item, [], cfg=_cfg(), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state == item                # exact same state object
    assert res.new_requests == ()
    assert res.completed_round is None


def test_need_candidate_parse_failure_goes_to_reflection(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    bad = StepResponse(
        request_id=stable_id("i1", 1, "challenger", 0),
        role="challenger", round_n=1, attempt=0, text="not json",
    )
    res = step(item, [bad], cfg=_cfg(max_rounds=3), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_REFLECTION
    assert res.new_requests[0].role == "reflector"


def test_need_candidate_parse_failure_at_max_rounds_goes_to_rejected(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE, current_round=3)
    bad = StepResponse(
        request_id=stable_id("i1", 3, "challenger", 0),
        role="challenger", round_n=3, attempt=0, text="bad",
    )
    res = step(item, [bad], cfg=_cfg(max_rounds=3), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state.state == State.REJECTED
    assert res.new_requests == ()
    assert res.state.rejection_reasons


def test_need_candidate_safety_block_goes_to_reflection(domain, grounding):
    cfg = _cfg(safety_enabled=False)
    cfg.safety.enabled = True
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    resp = StepResponse(
        request_id=stable_id("i1", 1, "challenger", 0),
        role="challenger", round_n=1, attempt=0, text=_challenger_text(),
    )
    def blocking_filter(txt):
        return SafetyVerdict(allowed=False, reasons=["pii:email"])

    res = step(item, [resp], cfg=cfg, harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding, safety_filter=blocking_filter)
    assert res.state.state == State.NEED_REFLECTION
    assert res.new_requests[0].role == "reflector"


# ---------------------------------------------------------------------------
# NEED_QUALITY
# ---------------------------------------------------------------------------

def test_need_quality_pass_emits_2n_solver_requests(domain, grounding):
    cand_text = _challenger_text()
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    # Advance to NEED_QUALITY with candidate.
    step1 = step(item, [StepResponse(stable_id("i1", 1, "challenger", 0),
                                      "challenger", 1, 0, cand_text)],
                 cfg=_cfg(weak=3, strong=3), harness=DEFAULT_HARNESS,
                 domain=domain, grounding=grounding)
    quality_resp = StepResponse(
        request_id=stable_id("i1", 1, "quality", 0),
        role="quality", round_n=1, attempt=0,
        text=json.dumps({"passed": True, "failures": [], "notes": "ok"}),
    )
    res = step(step1.state, [quality_resp], cfg=_cfg(weak=3, strong=3),
               harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_SCORES
    roles = [r.role for r in res.new_requests]
    assert roles.count("weak") == 3
    assert roles.count("strong") == 3


def test_need_quality_failure_with_rounds_left_goes_to_reflection(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_QUALITY,
                      candidate=_make_candidate(grounding.source_id))
    qresp = StepResponse(
        request_id=stable_id("i1", 1, "quality", 0),
        role="quality", round_n=1, attempt=0,
        text=json.dumps({"passed": False, "failures": ["leakage"], "notes": "bad"}),
    )
    res = step(item, [qresp], cfg=_cfg(max_rounds=3), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_REFLECTION
    assert res.completed_round is not None
    assert res.completed_round.quality.passed is False


def test_need_quality_failure_at_max_rounds_goes_to_rejected(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_QUALITY, current_round=2,
                      candidate=_make_candidate(grounding.source_id))
    qresp = StepResponse(
        request_id=stable_id("i1", 2, "quality", 0),
        role="quality", round_n=2, attempt=0,
        text=json.dumps({"passed": False, "failures": ["x"], "notes": "n"}),
    )
    res = step(item, [qresp], cfg=_cfg(max_rounds=2), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state.state == State.REJECTED
    assert res.state.rejection_reasons


def test_need_quality_partial_response_noop(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_QUALITY,
                      candidate=_make_candidate(grounding.source_id))
    res = step(item, [], cfg=_cfg(), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state == item
    assert res.new_requests == ()


# ---------------------------------------------------------------------------
# NEED_SCORES — the throughput-unlock state and the partial-completion test
# ---------------------------------------------------------------------------

def test_need_scores_emits_judge_for_each_solver_response(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_SCORES,
                      candidate=_make_candidate(grounding.source_id),
                      quality=_pass_quality())
    cfg = _cfg(weak=2, strong=2)
    res = step(item, [_solver_resp("weak", 0), _solver_resp("weak", 1)],
               cfg=cfg, harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_SCORES        # not advanced
    assert {r.role for r in res.new_requests} == {"judge"}
    assert len(res.new_requests) == 2
    assert res.completed_round is None


def test_partial_responses_noop_at_need_scores(domain, grounding):
    """MIGRATION_PLAN §2.3: with 3/4 judge responses, step() emits no new
    requests for the missing one and does NOT advance state."""
    item = _seed_item(grounding, state=State.NEED_SCORES,
                      candidate=_make_candidate(grounding.source_id),
                      quality=_pass_quality())
    cfg = _cfg(weak=2, strong=2)   # expects 4 judge responses total
    res = step(item,
               [_judge_resp(solver_role="weak", attempt=0, total=0.2),
                _judge_resp(solver_role="weak", attempt=1, total=0.2),
                _judge_resp(solver_role="strong", attempt=0, total=0.9)],
               cfg=cfg, harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_SCORES        # NOT advanced
    assert res.new_requests == ()                      # no double-emission
    assert res.completed_round is None
    assert len(res.state.weak_scores) == 2
    assert len(res.state.strong_scores) == 1           # only 1 strong so far


def test_need_scores_completes_and_accepts(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_SCORES,
                      candidate=_make_candidate(grounding.source_id),
                      quality=_pass_quality())
    cfg = _cfg(weak=2, strong=2)
    res = step(item,
               [_judge_resp(solver_role="weak", attempt=0, total=0.1),
                _judge_resp(solver_role="weak", attempt=1, total=0.2),
                _judge_resp(solver_role="strong", attempt=0, total=0.9),
                _judge_resp(solver_role="strong", attempt=1, total=0.85)],
               cfg=cfg, harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.ACCEPTED
    assert res.completed_round is not None
    assert res.completed_round.evaluation.accepted is True
    assert len(res.scores_to_persist) == 4


def test_need_scores_rejects_and_reflects(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_SCORES,
                      candidate=_make_candidate(grounding.source_id),
                      quality=_pass_quality())
    cfg = _cfg(weak=2, strong=2, max_rounds=3)
    res = step(item,
               [_judge_resp(solver_role="weak", attempt=0, total=0.7),  # too high
                _judge_resp(solver_role="weak", attempt=1, total=0.7),
                _judge_resp(solver_role="strong", attempt=0, total=0.9),
                _judge_resp(solver_role="strong", attempt=1, total=0.9)],
               cfg=cfg, harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_REFLECTION
    assert res.completed_round is not None
    assert res.completed_round.evaluation.accepted is False
    assert any(r.role == "reflector" for r in res.new_requests)


def test_need_scores_rejects_terminally_at_max_rounds(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_SCORES, current_round=2,
                      candidate=_make_candidate(grounding.source_id),
                      quality=_pass_quality())
    cfg = _cfg(weak=2, strong=2, max_rounds=2)
    res = step(item,
               [_judge_resp(solver_role="weak", attempt=0, total=0.7, round_n=2),
                _judge_resp(solver_role="weak", attempt=1, total=0.7, round_n=2),
                _judge_resp(solver_role="strong", attempt=0, total=0.9, round_n=2),
                _judge_resp(solver_role="strong", attempt=1, total=0.9, round_n=2)],
               cfg=cfg, harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.REJECTED
    assert res.state.rejection_reasons


# ---------------------------------------------------------------------------
# NEED_REFLECTION
# ---------------------------------------------------------------------------

def test_reflection_advances_to_need_candidate_with_bumped_round(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_REFLECTION, current_round=1,
                      rounds_history=())
    rresp = StepResponse(
        request_id=stable_id("i1", 1, "reflector", 0),
        role="reflector", round_n=1, attempt=0,
        text=json.dumps({"feedback": ["try harder"], "new_angle": "X"}),
    )
    res = step(item, [rresp], cfg=_cfg(), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_CANDIDATE
    assert res.state.current_round == 2
    assert "try harder" in res.state.last_feedback
    assert any("NEW_ANGLE" in f for f in res.state.last_feedback)
    assert res.new_requests[0].role == "challenger"


def test_reflection_partial_response_noop(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_REFLECTION)
    res = step(item, [], cfg=_cfg(), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state == item


# ---------------------------------------------------------------------------
# Terminal + determinism
# ---------------------------------------------------------------------------

def test_terminal_states_are_noops(domain, grounding):
    for state in (State.ACCEPTED, State.REJECTED):
        item = _seed_item(grounding, state=state)
        res = step(item, [_solver_resp("weak", 0)], cfg=_cfg(),
                   harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
        assert res.state == item
        assert res.new_requests == ()


def test_step_is_deterministic(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    resp = StepResponse(stable_id("i1", 1, "challenger", 0),
                        "challenger", 1, 0, _challenger_text())
    r1 = step(item, [resp], cfg=_cfg(), harness=DEFAULT_HARNESS,
              domain=domain, grounding=grounding)
    r2 = step(item, [resp], cfg=_cfg(), harness=DEFAULT_HARNESS,
              domain=domain, grounding=grounding)
    assert r1.state.state == r2.state.state
    assert [q.request_id for q in r1.new_requests] == [q.request_id for q in r2.new_requests]


def test_responses_for_other_rounds_are_ignored(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE, current_round=2)
    stale = StepResponse(stable_id("i1", 1, "challenger", 0),
                         "challenger", 1, 0, _challenger_text())   # round 1
    res = step(item, [stale], cfg=_cfg(), harness=DEFAULT_HARNESS,
               domain=domain, grounding=grounding)
    assert res.state == item
    assert res.new_requests == ()


# ---------------------------------------------------------------------------
# Helpers used by tests above
# ---------------------------------------------------------------------------

def _make_candidate(source_id: str):
    from autodata.schemas import Candidate, RubricCriterion
    return Candidate(
        candidate_id="c1", domain="qa_from_documents", source_id=source_id,
        payload={"question": "Q?", "context": "ctx"},
        rubric=[
            RubricCriterion(id="c1", description="x", weight=5),
            RubricCriterion(id="c2", description="y", weight=3),
        ],
        reference_output="reference answer",
    )


def _pass_quality():
    from autodata.schemas import QualityCheck
    return QualityCheck(passed=True, failures=[], notes="ok")
