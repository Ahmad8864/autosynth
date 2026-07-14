"""State transitions and partial-response handling in the pure pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosynth.acceptance import JudgePolicy, VerifiablePolicy
from autosynth.config import (
    AcceptanceConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autosynth.domain import DomainAdapter, GroundingItem
from autosynth.domains.qa_from_documents import QAFromDocuments
from autosynth.harness import DEFAULT_HARNESS
from autosynth.pipeline import (
    ItemState,
    State,
    StepResponse,
    step,
)
from autosynth.safety import SafetyVerdict
from autosynth.schemas import SolverScore
from autosynth.utils import stable_id

# Fixtures


@pytest.fixture
def domain(tmp_path: Path):
    (tmp_path / "doc.md").write_text("body of doc")
    return QAFromDocuments(source_dir=str(tmp_path))


@pytest.fixture
def grounding(domain) -> GroundingItem:
    return next(iter(domain.load_grounding()))


def _cfg(*, max_rounds: int = 3, weak: int = 2, strong: int = 2, safety_enabled: bool = False) -> RunConfig:
    return RunConfig(
        run_id="r1",
        output_dir="/tmp/out",
        max_examples=1,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": "/tmp"}),
        loop=LoopConfig(
            max_rounds=max_rounds, weak_samples=weak, strong_samples=strong, short_circuit_strong=False
        ),
        acceptance=AcceptanceConfig(forbid_weak_zero=False),
        orchestrator=ModelConfig(provider_model="mock/x"),
        challenger=ModelConfig(provider_model="mock/x"),
        weak_solver=ModelConfig(provider_model="mock/x"),
        strong_solver=ModelConfig(provider_model="mock/x"),
        judge=ModelConfig(provider_model="mock/x"),
    )


def _seed_item(
    grounding: GroundingItem, *, state: State = State.PENDING, current_round: int = 1, **overrides
) -> ItemState:
    base = ItemState(
        item_id="i1",
        run_id="r1",
        source_id=grounding.source_id,
        domain="qa_from_documents",
        state=state,
        current_round=current_round,
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
        role=role,
        round_n=round_n,
        attempt=attempt,
        text=f"{role}-attempt-{attempt}",
    )


def _judge_resp(
    *, solver_role: str, attempt: int, total: float, round_n: int = 1, solver_text: str = "x"
) -> StepResponse:
    parent_id = stable_id("i1", round_n, solver_role, attempt)
    return StepResponse(
        request_id=stable_id("i1", round_n, "judge", solver_role, parent_id),
        role="judge",
        round_n=round_n,
        attempt=attempt,
        text=json.dumps({"per_criterion": {"c1": total, "c2": total}, "total": total}),
        parent_response_id=parent_id,
        solver_response_text=solver_text,
        solver_role=solver_role,
    )


# PENDING → NEED_CANDIDATE


def test_pending_emits_challenger_request(domain, grounding):
    item = _seed_item(grounding)
    res = step(item, [], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_CANDIDATE
    assert len(res.new_requests) == 1
    assert res.new_requests[0].role == "challenger"


# NEED_CANDIDATE


def test_need_candidate_advances_to_need_quality_on_valid_response(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    resp = StepResponse(
        request_id=stable_id("i1", 1, "challenger", 0),
        role="challenger",
        round_n=1,
        attempt=0,
        text=_challenger_text(),
    )
    res = step(item, [resp], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_QUALITY
    assert res.state.candidate is not None
    assert len(res.new_requests) == 1
    assert res.new_requests[0].role == "quality"


def test_need_candidate_partial_responses_noop(domain, grounding):
    """Wait for the challenger without changing state."""
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    res = step(item, [], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state == item
    assert res.new_requests == ()
    assert res.completed_round is None


def test_need_candidate_parse_failure_goes_to_reflection(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    bad = StepResponse(
        request_id=stable_id("i1", 1, "challenger", 0),
        role="challenger",
        round_n=1,
        attempt=0,
        text="not json",
    )
    res = step(
        item, [bad], cfg=_cfg(max_rounds=3), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding
    )
    assert res.state.state == State.NEED_REFLECTION
    assert res.new_requests[0].role == "reflector"


def test_need_candidate_parse_failure_at_max_rounds_goes_to_rejected(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE, current_round=3)
    bad = StepResponse(
        request_id=stable_id("i1", 3, "challenger", 0),
        role="challenger",
        round_n=3,
        attempt=0,
        text="bad",
    )
    res = step(
        item, [bad], cfg=_cfg(max_rounds=3), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding
    )
    assert res.state.state == State.REJECTED
    assert res.new_requests == ()
    assert res.state.rejection_reasons


def test_need_candidate_non_object_json_goes_to_reflection(domain, grounding):
    # Invalid top-level shape should enter the normal reflection path.
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    bad = StepResponse(
        request_id=stable_id("i1", 1, "challenger", 0),
        role="challenger",
        round_n=1,
        attempt=0,
        text='["correctness", "clarity"]',
    )
    res = step(
        item, [bad], cfg=_cfg(max_rounds=3), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding
    )
    assert res.state.state == State.NEED_REFLECTION


def test_safety_filter_exception_fails_closed_to_reflection(domain, grounding):
    # User filter errors fail closed without escaping ``step``.
    cfg = _cfg(max_rounds=3)
    cfg.safety.enabled = True
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    resp = StepResponse(
        request_id=stable_id("i1", 1, "challenger", 0),
        role="challenger",
        round_n=1,
        attempt=0,
        text=_challenger_text(),
    )

    def _raising_filter(_text: str):
        raise RuntimeError("DLP backend down")

    res = step(
        item,
        [resp],
        cfg=cfg,
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
        safety_filter=_raising_filter,
    )
    assert res.state.state == State.NEED_REFLECTION
    cr = res.completed_round
    assert cr is not None
    quality = cr.quality
    assert quality is not None
    assert any("filter_error" in f for f in quality.failures)


def test_need_candidate_safety_block_goes_to_reflection(domain, grounding):
    cfg = _cfg(safety_enabled=False)
    cfg.safety.enabled = True
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    resp = StepResponse(
        request_id=stable_id("i1", 1, "challenger", 0),
        role="challenger",
        round_n=1,
        attempt=0,
        text=_challenger_text(),
    )

    def blocking_filter(txt):
        return SafetyVerdict(allowed=False, reasons=["pii:email"])

    res = step(
        item,
        [resp],
        cfg=cfg,
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
        safety_filter=blocking_filter,
    )
    assert res.state.state == State.NEED_REFLECTION
    assert res.new_requests[0].role == "reflector"


# NEED_QUALITY


def test_need_quality_pass_emits_2n_solver_requests(domain, grounding):
    cand_text = _challenger_text()
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    step1 = step(
        item,
        [StepResponse(stable_id("i1", 1, "challenger", 0), "challenger", 1, 0, cand_text)],
        cfg=_cfg(weak=3, strong=3),
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    quality_resp = StepResponse(
        request_id=stable_id("i1", 1, "quality", 0),
        role="quality",
        round_n=1,
        attempt=0,
        text=json.dumps({"passed": True, "failures": [], "notes": "ok"}),
    )
    res = step(
        step1.state,
        [quality_resp],
        cfg=_cfg(weak=3, strong=3),
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.NEED_SCORES
    roles = [r.role for r in res.new_requests]
    assert roles.count("weak") == 3
    assert roles.count("strong") == 3


def test_need_quality_failure_with_rounds_left_goes_to_reflection(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_QUALITY, candidate=_make_candidate(grounding.source_id))
    qresp = StepResponse(
        request_id=stable_id("i1", 1, "quality", 0),
        role="quality",
        round_n=1,
        attempt=0,
        text=json.dumps({"passed": False, "failures": ["leakage"], "notes": "bad"}),
    )
    res = step(
        item, [qresp], cfg=_cfg(max_rounds=3), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding
    )
    assert res.state.state == State.NEED_REFLECTION
    assert res.completed_round is not None
    assert res.completed_round.quality.passed is False


def test_need_quality_failure_at_max_rounds_goes_to_rejected(domain, grounding):
    item = _seed_item(
        grounding, state=State.NEED_QUALITY, current_round=2, candidate=_make_candidate(grounding.source_id)
    )
    qresp = StepResponse(
        request_id=stable_id("i1", 2, "quality", 0),
        role="quality",
        round_n=2,
        attempt=0,
        text=json.dumps({"passed": False, "failures": ["x"], "notes": "n"}),
    )
    res = step(
        item, [qresp], cfg=_cfg(max_rounds=2), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding
    )
    assert res.state.state == State.REJECTED
    assert res.state.rejection_reasons


def test_need_quality_partial_response_noop(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_QUALITY, candidate=_make_candidate(grounding.source_id))
    res = step(item, [], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state == item
    assert res.new_requests == ()


# NEED_SCORES and partial completion


def test_need_scores_emits_judge_for_each_solver_response(domain, grounding):
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
    )
    cfg = _cfg(weak=2, strong=2)
    res = step(
        item,
        [_solver_resp("weak", 0), _solver_resp("weak", 1)],
        cfg=cfg,
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.NEED_SCORES
    assert {r.role for r in res.new_requests} == {"judge"}
    assert len(res.new_requests) == 2
    assert res.completed_round is None


def test_partial_responses_noop_at_need_scores(domain, grounding):
    """Partial scores neither advance state nor duplicate requests."""
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
    )
    cfg = _cfg(weak=2, strong=2)
    res = step(
        item,
        [
            _judge_resp(solver_role="weak", attempt=0, total=0.2),
            _judge_resp(solver_role="weak", attempt=1, total=0.2),
            _judge_resp(solver_role="strong", attempt=0, total=0.9),
        ],
        cfg=cfg,
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.NEED_SCORES
    assert res.new_requests == ()
    assert res.completed_round is None
    assert len(res.state.weak_scores) == 2
    assert len(res.state.strong_scores) == 1


def test_need_scores_completes_and_accepts(domain, grounding):
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
    )
    cfg = _cfg(weak=2, strong=2)
    res = step(
        item,
        [
            _judge_resp(solver_role="weak", attempt=0, total=0.1),
            _judge_resp(solver_role="weak", attempt=1, total=0.2),
            _judge_resp(solver_role="strong", attempt=0, total=0.9),
            _judge_resp(solver_role="strong", attempt=1, total=0.85),
        ],
        cfg=cfg,
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.ACCEPTED
    assert res.completed_round is not None and res.completed_round.evaluation is not None
    assert res.completed_round.evaluation.accepted is True
    assert len(res.scores_to_persist) == 4


def test_need_scores_rejects_and_reflects(domain, grounding):
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
    )
    cfg = _cfg(weak=2, strong=2, max_rounds=3)
    res = step(
        item,
        [
            _judge_resp(solver_role="weak", attempt=0, total=0.7),
            _judge_resp(solver_role="weak", attempt=1, total=0.7),
            _judge_resp(solver_role="strong", attempt=0, total=0.9),
            _judge_resp(solver_role="strong", attempt=1, total=0.9),
        ],
        cfg=cfg,
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.NEED_REFLECTION
    assert res.completed_round is not None and res.completed_round.evaluation is not None
    assert res.completed_round.evaluation.accepted is False
    assert any(r.role == "reflector" for r in res.new_requests)


def test_need_scores_rejects_terminally_at_max_rounds(domain, grounding):
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        current_round=2,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
    )
    cfg = _cfg(weak=2, strong=2, max_rounds=2)
    res = step(
        item,
        [
            _judge_resp(solver_role="weak", attempt=0, total=0.7, round_n=2),
            _judge_resp(solver_role="weak", attempt=1, total=0.7, round_n=2),
            _judge_resp(solver_role="strong", attempt=0, total=0.9, round_n=2),
            _judge_resp(solver_role="strong", attempt=1, total=0.9, round_n=2),
        ],
        cfg=cfg,
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.REJECTED
    assert res.state.rejection_reasons


# NEED_REFLECTION


def test_reflection_advances_to_need_candidate_with_bumped_round(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_REFLECTION, current_round=1, rounds_history=())
    rresp = StepResponse(
        request_id=stable_id("i1", 1, "reflector", 0),
        role="reflector",
        round_n=1,
        attempt=0,
        text=json.dumps({"feedback": ["try harder"], "new_angle": "X"}),
    )
    res = step(item, [rresp], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_CANDIDATE
    assert res.state.current_round == 2
    assert "try harder" in res.state.last_feedback
    assert any("NEW_ANGLE" in f for f in res.state.last_feedback)
    assert res.new_requests[0].role == "challenger"


def test_reflection_partial_response_noop(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_REFLECTION)
    res = step(item, [], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state == item


# Terminal + determinism


def test_terminal_states_are_noops(domain, grounding):
    for state in (State.ACCEPTED, State.REJECTED):
        item = _seed_item(grounding, state=state)
        res = step(
            item,
            [_solver_resp("weak", 0)],
            cfg=_cfg(),
            harness=DEFAULT_HARNESS,
            domain=domain,
            grounding=grounding,
        )
        assert res.state == item
        assert res.new_requests == ()


def test_step_is_deterministic(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE)
    resp = StepResponse(stable_id("i1", 1, "challenger", 0), "challenger", 1, 0, _challenger_text())
    r1 = step(item, [resp], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    r2 = step(item, [resp], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert r1 == r2


def test_responses_for_other_rounds_are_ignored(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_CANDIDATE, current_round=2)
    stale = StepResponse(stable_id("i1", 1, "challenger", 0), "challenger", 1, 0, _challenger_text())
    res = step(item, [stale], cfg=_cfg(), harness=DEFAULT_HARNESS, domain=domain, grounding=grounding)
    assert res.state == item
    assert res.new_requests == ()


# Helpers used by tests above


def _make_candidate(source_id: str):
    from autosynth.schemas import Candidate, RubricCriterion

    return Candidate(
        candidate_id="c1",
        domain="qa_from_documents",
        source_id=source_id,
        payload={"question": "Q?", "context": "ctx"},
        rubric=[
            RubricCriterion(id="c1", description="x", weight=5),
            RubricCriterion(id="c2", description="y", weight=3),
        ],
        reference_output="reference answer",
    )


def _pass_quality():
    from autosynth.schemas import QualityCheck

    return QualityCheck(passed=True, failures=[], notes="ok")


# Verifiable mode — in-process scoring via domain.verify() (no judge)


class _VerifyDomain(DomainAdapter):
    """Minimal verifiable domain: an attempt is correct iff it contains 'CORRECT'."""

    name = "verify_test"
    default_acceptance_mode = "verifiable"

    def load_grounding(self):
        raise NotImplementedError

    def generation_prompt(self, item, feedback, round_n, prior_payloads):
        raise NotImplementedError

    def validate_candidate(self, candidate):
        return []

    def solver_prompt(self, candidate):
        raise NotImplementedError

    def quality_prompt(self, candidate):
        raise NotImplementedError

    def judge_prompt(self, candidate, solver_response):
        raise NotImplementedError

    def verify(self, candidate, solver_response):
        if "RAISE" in solver_response:
            raise RuntimeError("boom")
        if "SKIP" in solver_response:
            return None
        return "CORRECT" in solver_response


def _verify_policy(*, weak_max=0, strong_min=2):
    return VerifiablePolicy(
        AcceptanceConfig(verifiable_weak_max_correct=weak_max, verifiable_strong_min_correct=strong_min),
        weak_samples=2,
        strong_samples=2,
    )


def _solver_text(role: str, attempt: int, text: str, *, round_n: int = 1) -> StepResponse:
    return StepResponse(
        request_id=stable_id("i1", round_n, role, attempt),
        role=role,
        round_n=round_n,
        attempt=attempt,
        text=text,
    )


def _verify_step(item, responses, **kw):
    return step(
        item,
        responses,
        cfg=_cfg(weak=2, strong=2, **kw),
        harness=DEFAULT_HARNESS,
        domain=_VerifyDomain(),
        grounding=None,
        policy=_verify_policy(),
    )


def _scored_item(grounding, **overrides):
    return _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
        **overrides,
    )


def test_verified_scores_solvers_without_judge(grounding):
    res = _verify_step(
        _scored_item(grounding), [_solver_text("weak", 0, "wrong"), _solver_text("strong", 0, "CORRECT")]
    )
    assert res.state.state == State.NEED_SCORES
    assert res.new_requests == ()
    assert len(res.state.weak_scores) == 1 and len(res.state.strong_scores) == 1
    assert res.state.weak_scores[0].correct is False and res.state.weak_scores[0].total == 0.0
    assert res.state.strong_scores[0].correct is True and res.state.strong_scores[0].total == 1.0
    assert len(res.scores_to_persist) == 2


def test_verified_completes_and_accepts(grounding):
    res = _verify_step(
        _scored_item(grounding),
        [
            _solver_text("weak", 0, "wrong"),
            _solver_text("weak", 1, "wrong"),
            _solver_text("strong", 0, "CORRECT"),
            _solver_text("strong", 1, "CORRECT"),
        ],
    )
    assert res.state.state == State.ACCEPTED
    assert res.completed_round is not None and res.completed_round.evaluation is not None
    assert res.completed_round.evaluation.accepted is True
    assert len(res.scores_to_persist) == 4


def test_verified_rejects_and_reflects(grounding):
    res = _verify_step(
        _scored_item(grounding),
        [
            _solver_text("weak", 0, "wrong"),
            _solver_text("weak", 1, "wrong"),
            _solver_text("strong", 0, "CORRECT"),
            _solver_text("strong", 1, "wrong"),
        ],
        max_rounds=3,
    )
    assert res.state.state == State.NEED_REFLECTION
    assert res.completed_round is not None and res.completed_round.evaluation is not None
    assert res.completed_round.evaluation.accepted is False
    assert any(r.role == "reflector" for r in res.new_requests)


def test_verified_verify_exception_counts_incorrect(grounding):
    """Verifier errors count as incorrect without escaping ``step``."""
    res = _verify_step(
        _scored_item(grounding),
        [_solver_text(role, a, "RAISE") for role in ("weak", "strong") for a in (0, 1)],
        max_rounds=1,
    )
    assert res.state.state == State.REJECTED
    scores = res.state.weak_scores + res.state.strong_scores
    assert all(s.total == 0.0 and s.correct is None for s in scores)
    assert all(any("verify_error" in f for f in s.failure_modes) for s in scores)


def test_verified_dedups_redelivered_solver(grounding):
    prior = (SolverScore(solver="weak", attempt=0, raw_response="x", total=0.0, correct=False),)
    res = _verify_step(
        _scored_item(grounding, weak_scores=prior),
        [_solver_text("weak", 0, "wrong"), _solver_text("weak", 1, "wrong")],
    )
    assert len(res.state.weak_scores) == 2
    assert len(res.scores_to_persist) == 1  # only the genuinely new attempt


# Judge-decided policy — NEED_SCORES → NEED_DECISION → accept / improve


def _loop_judge_resp(*, verdict="improve", suggestion="harder", reason="too easy", round_n=1) -> StepResponse:
    body = {"verdict": verdict, "grpo_suitability": "low", "reason": reason, "suggestion": suggestion}
    return StepResponse(
        request_id=stable_id("i1", round_n, "loop_judge", 0),
        role="loop_judge",
        round_n=round_n,
        attempt=0,
        text=json.dumps(body),
    )


def _decided_item(grounding, *, current_round=1, **overrides):
    return _seed_item(
        grounding,
        state=State.NEED_DECISION,
        current_round=current_round,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
        weak_scores=(SolverScore(solver="weak", attempt=0, raw_response="x", total=0.2),),
        strong_scores=(SolverScore(solver="strong", attempt=0, raw_response="x", total=0.8),),
        **overrides,
    )


def _judge_step(item, responses, grounding, **cfg_kw):
    return step(
        item,
        responses,
        cfg=_cfg(weak=2, strong=2, **cfg_kw),
        harness=DEFAULT_HARNESS,
        domain=QAFromDocuments(source_dir="/tmp"),
        grounding=grounding,
        policy=JudgePolicy(AcceptanceConfig()),
    )


def test_judge_scores_complete_emits_loop_judge(domain, grounding):
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
    )
    res = step(
        item,
        [
            _judge_resp(solver_role="weak", attempt=0, total=0.2),
            _judge_resp(solver_role="weak", attempt=1, total=0.2),
            _judge_resp(solver_role="strong", attempt=0, total=0.9),
            _judge_resp(solver_role="strong", attempt=1, total=0.85),
        ],
        cfg=_cfg(weak=2, strong=2),
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
        policy=JudgePolicy(AcceptanceConfig()),
    )
    assert res.state.state == State.NEED_DECISION
    assert [r.role for r in res.new_requests] == ["loop_judge"]
    assert res.completed_round is None
    assert len(res.scores_to_persist) == 4


def test_decision_accept(grounding):
    res = _judge_step(_decided_item(grounding), [_loop_judge_resp(verdict="accept")], grounding)
    assert res.state.state == State.ACCEPTED
    assert res.completed_round is not None and res.completed_round.evaluation is not None
    assert res.completed_round.evaluation.accepted is True


def test_decision_improve_bumps_round_with_suggestion(grounding):
    res = _judge_step(
        _decided_item(grounding),
        [_loop_judge_resp(verdict="improve", suggestion="add a numeric trap")],
        grounding,
        max_rounds=3,
    )
    assert res.state.state == State.NEED_CANDIDATE
    assert res.state.current_round == 2
    assert "add a numeric trap" in res.state.last_feedback
    assert [r.role for r in res.new_requests] == ["challenger"]
    assert res.completed_round is not None and res.completed_round.evaluation is not None
    assert res.completed_round.evaluation.accepted is False


def test_decision_improve_at_max_rounds_rejects(grounding):
    res = _judge_step(
        _decided_item(grounding, current_round=2),
        [_loop_judge_resp(verdict="improve", round_n=2)],
        grounding,
        max_rounds=2,
    )
    assert res.state.state == State.REJECTED
    assert res.completed_round is not None


def test_decision_noop_without_response(grounding):
    item = _decided_item(grounding)
    res = _judge_step(item, [], grounding)
    assert res.state == item
    assert res.new_requests == ()


# Conditional strong-solver evaluation (loop.short_circuit_strong)


def _sc_cfg(**kw):
    cfg = _cfg(**kw)
    cfg.loop.short_circuit_strong = True
    return cfg


def _weak_score(total: float, attempt: int = 0):
    return SolverScore(solver="weak", attempt=attempt, raw_response="x", total=total)


def test_short_circuit_emits_weak_only(domain, grounding):
    item = _seed_item(grounding, state=State.NEED_QUALITY, candidate=_make_candidate(grounding.source_id))
    quality_resp = StepResponse(
        request_id=stable_id("i1", 1, "quality", 0),
        role="quality",
        round_n=1,
        attempt=0,
        text=json.dumps({"passed": True, "failures": [], "notes": "ok"}),
    )
    res = step(
        item,
        [quality_resp],
        cfg=_sc_cfg(weak=2, strong=2),
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.NEED_SCORES
    assert {r.role for r in res.new_requests} == {"weak"}
    assert len(res.new_requests) == 2


def test_short_circuit_runs_strong_when_weak_hard(domain, grounding):
    # Passing the weak gate releases strong requests.
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
        weak_scores=(_weak_score(0.2),),
    )
    res = step(
        item,
        [_judge_resp(solver_role="weak", attempt=1, total=0.2)],
        cfg=_sc_cfg(weak=2, strong=2),
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.NEED_SCORES
    assert {r.role for r in res.new_requests} == {"strong"}
    assert len(res.new_requests) == 2


def test_short_circuit_skips_strong_when_weak_too_capable(domain, grounding):
    # A capable weak solver skips strong evaluation and triggers reflection.
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
        weak_scores=(_weak_score(0.7),),
    )
    res = step(
        item,
        [_judge_resp(solver_role="weak", attempt=1, total=0.7)],
        cfg=_sc_cfg(weak=2, strong=2, max_rounds=3),
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.NEED_REFLECTION
    assert not any(r.role == "strong" for r in res.new_requests)
    assert res.completed_round is not None
    ev = res.completed_round.evaluation
    assert ev is not None and ev.accepted is False
    assert any("short_circuit" in r for r in ev.rejection_reasons)


def test_short_circuit_disarms_once_strong_scoring(domain, grounding):
    # Do not re-emit strong requests already in progress.
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
        weak_scores=(_weak_score(0.2), _weak_score(0.2, attempt=1)),
        strong_scores=(SolverScore(solver="strong", attempt=0, raw_response="x", total=0.9),),
    )
    res = step(
        item,
        [_judge_resp(solver_role="strong", attempt=1, total=0.9)],
        cfg=_sc_cfg(weak=2, strong=2),
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state == State.ACCEPTED
    assert not any(r.role == "strong" for r in res.new_requests)


def test_short_circuit_strong_samples_zero_does_not_stall(domain, grounding):
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_make_candidate(grounding.source_id),
        quality=_pass_quality(),
        weak_scores=(_weak_score(0.2),),
    )
    res = step(
        item,
        [_judge_resp(solver_role="weak", attempt=1, total=0.2)],
        cfg=_sc_cfg(weak=2, strong=0, max_rounds=1),
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
    )
    assert res.state.state != State.NEED_SCORES
