"""Pure pipeline: state machine that turns responses into requests.

This is the keystone of the event-sourced architecture. ``step()`` is a pure
function: ``(state, fresh_responses) -> (new_state, new_requests,
completed_round?, scores_to_persist?)``. No I/O, no time, no randomness.

The dispatcher orders responses by ``request_id`` before calling ``step()``
so concurrent fulfillment doesn't perturb state evolution.

States (5 + 2 terminal):
    PENDING → NEED_CANDIDATE → NEED_QUALITY → NEED_SCORES
              ↘ NEED_REFLECTION ↗
              ↘ ACCEPTED | REJECTED

Sync work (structural validation, safety filter, acceptance evaluation)
happens *inside* the response handler for the preceding async state.

See MIGRATION_PLAN.md §2.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from autodata.agents import challenger as challenger_agent
from autodata.agents import reflector as reflector_agent
from autodata.agents import solver as solver_agent
from autodata.agents import verifier as verifier_agent
from autodata.config import AcceptanceConfig, RunConfig
from autodata.domain import DomainAdapter, GroundingItem
from autodata.evaluator import evaluate
from autodata.harness import DEFAULT_HARNESS, HarnessSpec
from autodata.llm import LLMRequest
from autodata.safety import SafetyFilter, SafetyVerdict
from autodata.schemas import (
    Candidate,
    EvalReport,
    QualityCheck,
    Round,
    SolverScore,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class State(str, Enum):
    PENDING = "PENDING"
    NEED_CANDIDATE = "NEED_CANDIDATE"
    NEED_QUALITY = "NEED_QUALITY"
    NEED_SCORES = "NEED_SCORES"
    NEED_REFLECTION = "NEED_REFLECTION"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


TERMINAL_STATES: frozenset[State] = frozenset({State.ACCEPTED, State.REJECTED})


@dataclass(frozen=True)
class StepResponse:
    """A response surfaced to step(). The dispatcher hydrates these from the
    store before each invocation. For judge responses, the dispatcher also
    resolves and attaches ``solver_response_text`` and ``solver_role`` from
    the parent solver request."""

    request_id: str
    role: str
    round_n: int
    attempt: int
    text: str
    parent_response_id: Optional[str] = None
    solver_response_text: Optional[str] = None  # judge only
    solver_role: Optional[str] = None           # judge only


@dataclass(frozen=True)
class ScoreRecord:
    """A score plus the two response_ids it references; persisted to solver_scores."""

    score: SolverScore
    solver_response_id: str
    judge_response_id: str


@dataclass(frozen=True)
class ItemState:
    item_id: str
    run_id: str
    source_id: str
    domain: str
    state: State
    current_round: int                              # 1-indexed
    rounds_history: tuple[Round, ...] = ()
    candidate: Optional[Candidate] = None
    quality: Optional[QualityCheck] = None
    weak_scores: tuple[SolverScore, ...] = ()
    strong_scores: tuple[SolverScore, ...] = ()
    last_feedback: tuple[str, ...] = ()
    source_metadata: dict[str, Any] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()
    # Map of solver request_id → judge response IDs we've already paired.
    # Used by NEED_SCORES to find score_records when emitting score persistence.
    judge_response_for_solver: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    state: ItemState
    new_requests: tuple[LLMRequest, ...] = ()
    completed_round: Optional[Round] = None
    scores_to_persist: tuple[ScoreRecord, ...] = ()


# ---------------------------------------------------------------------------
# Model-key resolution
# ---------------------------------------------------------------------------

_ROLE_TO_CFG_ATTR: dict[str, str] = {
    "challenger": "challenger",
    "quality": "judge",
    "judge": "judge",
    "weak": "weak_solver",
    "strong": "strong_solver",
    "reflector": "orchestrator",
}


def model_key_for(cfg: RunConfig, role: str) -> str:
    attr = _ROLE_TO_CFG_ATTR[role]
    return getattr(cfg, attr).provider_model


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def step(
    item: ItemState,
    new_responses: list[StepResponse],
    *,
    cfg: RunConfig,
    harness: HarnessSpec,
    domain: DomainAdapter,
    grounding: GroundingItem,
    safety_filter: Optional[SafetyFilter] = None,
) -> StepResult:
    """Pure state transition. See module docstring."""
    if item.state in TERMINAL_STATES:
        return StepResult(state=item)

    # Ignore responses from prior rounds (artifacts of resume or out-of-order arrival).
    relevant = [r for r in new_responses if r.round_n == item.current_round]

    h = harness or DEFAULT_HARNESS

    if item.state == State.PENDING:
        return _emit_challenger(item, cfg, h, domain, grounding)
    if item.state == State.NEED_CANDIDATE:
        return _on_challenger(item, relevant, cfg, h, domain, grounding, safety_filter)
    if item.state == State.NEED_QUALITY:
        return _on_quality(item, relevant, cfg, h, domain, grounding)
    if item.state == State.NEED_SCORES:
        return _on_scores(item, relevant, cfg, h, domain, grounding)
    if item.state == State.NEED_REFLECTION:
        return _on_reflection(item, relevant, cfg, h, domain, grounding)
    raise ValueError(f"unknown state {item.state!r}")


# ---------------------------------------------------------------------------
# Per-state handlers
# ---------------------------------------------------------------------------

def _emit_challenger(item, cfg, harness, domain, grounding) -> StepResult:
    req = challenger_agent.build_request(
        item_id=item.item_id,
        round_n=item.current_round,
        model_key=model_key_for(cfg, "challenger"),
        grounding=grounding,
        feedback=list(item.last_feedback),
        prior_payloads=[r.candidate.payload for r in item.rounds_history],
        domain=domain,
        harness=harness,
    )
    new_state = replace(item, state=State.NEED_CANDIDATE)
    return StepResult(state=new_state, new_requests=(req,))


def _on_challenger(item, relevant, cfg, harness, domain, grounding, safety_filter) -> StepResult:
    resp = _find_role(relevant, "challenger")
    if resp is None:
        return StepResult(state=item)  # partial-completion noop

    # Parse
    try:
        candidate = challenger_agent.parse_response(
            resp.text,
            source_id=item.source_id,
            round_n=item.current_round,
            domain_name=domain.name,
            source_metadata=item.source_metadata,
            rubric_max_weight=harness.rubric_max_weight or cfg.acceptance.rubric_max_weight,
        )
    except ValueError as e:
        return _go_to_reflection_or_reject(
            item, cfg, harness, domain, grounding,
            failure_quality=QualityCheck(passed=False, failures=[f"challenger_parse_error:{e}"]),
        )

    # Structural validation
    struct_errs = domain.validate_candidate(candidate)
    if struct_errs:
        item_with_cand = replace(item, candidate=candidate)
        return _go_to_reflection_or_reject(
            item_with_cand, cfg, harness, domain, grounding,
            failure_quality=QualityCheck(passed=False, failures=struct_errs,
                                         notes="structural validation failed"),
        )

    # Safety
    if cfg.safety.enabled and safety_filter is not None:
        verdict = safety_filter(_concat_payload(candidate.payload))
        if not verdict.allowed:
            item_with_cand = replace(item, candidate=candidate)
            return _go_to_reflection_or_reject(
                item_with_cand, cfg, harness, domain, grounding,
                failure_quality=QualityCheck(passed=False,
                                             failures=[f"safety:{r}" for r in verdict.reasons],
                                             notes="safety filter rejected"),
            )

    # All sync checks pass: emit quality_req → NEED_QUALITY.
    qreq = verifier_agent.build_quality_request(
        item_id=item.item_id, round_n=item.current_round,
        model_key=model_key_for(cfg, "quality"),
        candidate=candidate, domain=domain, harness=harness,
    )
    new_state = replace(item, state=State.NEED_QUALITY, candidate=candidate)
    return StepResult(state=new_state, new_requests=(qreq,))


def _on_quality(item, relevant, cfg, harness, domain, grounding) -> StepResult:
    resp = _find_role(relevant, "quality")
    if resp is None:
        return StepResult(state=item)

    quality = verifier_agent.parse_quality(resp.text)
    if not quality.passed:
        item_with_q = replace(item, quality=quality)
        return _go_to_reflection_or_reject(item_with_q, cfg, harness, domain, grounding,
                                           failure_quality=quality)

    # Emit solver requests: N weak + N strong, all at once.
    reqs: list[LLMRequest] = []
    for k in range(cfg.loop.weak_samples):
        reqs.append(solver_agent.build_request(
            item_id=item.item_id, round_n=item.current_round, attempt=k,
            model_key=model_key_for(cfg, "weak"),
            candidate=item.candidate, role="weak", domain=domain, harness=harness,
        ))
    for k in range(cfg.loop.strong_samples):
        reqs.append(solver_agent.build_request(
            item_id=item.item_id, round_n=item.current_round, attempt=k,
            model_key=model_key_for(cfg, "strong"),
            candidate=item.candidate, role="strong", domain=domain, harness=harness,
        ))
    new_state = replace(item, state=State.NEED_SCORES, quality=quality,
                        weak_scores=(), strong_scores=(), judge_response_for_solver={})
    return StepResult(state=new_state, new_requests=tuple(reqs))


def _on_scores(item, relevant, cfg, harness, domain, grounding) -> StepResult:
    """The throughput-unlock state. Handles two kinds of responses:

      - solver response → emit a judge request (no state change)
      - judge response  → record a SolverScore (no state change *unless*
                          all 2N scores are in, in which case we evaluate)
    """
    new_requests: list[LLMRequest] = []
    new_weak = list(item.weak_scores)
    new_strong = list(item.strong_scores)
    new_scores: list[ScoreRecord] = []
    judge_for_solver = dict(item.judge_response_for_solver)

    # 1. Emit judge requests for any newly-arrived solver responses.
    for r in relevant:
        if r.role in ("weak", "strong"):
            jreq = verifier_agent.build_judge_request(
                item_id=item.item_id, round_n=item.current_round, attempt=r.attempt,
                model_key=model_key_for(cfg, "judge"),
                candidate=item.candidate, solver_response=r.text,
                solver_role=r.role, domain=domain, harness=harness,
                parent_response_id=r.request_id,
            )
            new_requests.append(jreq)

    # 2. Process judge responses → SolverScores + ScoreRecords.
    for r in relevant:
        if r.role != "judge":
            continue
        solver_text = r.solver_response_text or ""
        solver_role = _solver_role_from_judge(r)
        if solver_role is None or item.candidate is None:
            continue
        score = verifier_agent.parse_judge(
            r.text, candidate=item.candidate, solver_role=solver_role,
            attempt=r.attempt, solver_response_text=solver_text,
        )
        if solver_role == "weak":
            new_weak.append(score)
        else:
            new_strong.append(score)
        # parent_response_id is the solver's request_id (== response_id).
        if r.parent_response_id is not None:
            judge_for_solver[r.parent_response_id] = r.request_id
            new_scores.append(ScoreRecord(
                score=score,
                solver_response_id=r.parent_response_id,
                judge_response_id=r.request_id,
            ))

    # 3. Have we collected all 2N scores?
    expected_weak = cfg.loop.weak_samples
    expected_strong = cfg.loop.strong_samples
    if len(new_weak) < expected_weak or len(new_strong) < expected_strong:
        # Partial completion — return updated accumulators, no state change.
        new_state = replace(
            item,
            weak_scores=tuple(new_weak),
            strong_scores=tuple(new_strong),
            judge_response_for_solver=judge_for_solver,
        )
        return StepResult(
            state=new_state,
            new_requests=tuple(new_requests),
            scores_to_persist=tuple(new_scores),
        )

    # 4. All scores in — evaluate acceptance.
    ev: EvalReport = evaluate(new_weak, new_strong, item.quality, cfg.acceptance)
    round_obj = Round(
        refinement_round=item.current_round,
        candidate=item.candidate,
        quality=item.quality,
        evaluation=ev,
        reflection=None,
        ended_at=datetime.now(timezone.utc),
    )

    if ev.accepted:
        new_state = replace(item, state=State.ACCEPTED,
                            weak_scores=tuple(new_weak), strong_scores=tuple(new_strong),
                            rounds_history=item.rounds_history + (round_obj,))
        return StepResult(
            state=new_state, new_requests=tuple(new_requests),
            completed_round=round_obj, scores_to_persist=tuple(new_scores),
        )

    # Rejected this round. Reflect or terminate.
    if item.current_round < cfg.loop.max_rounds:
        rreq = reflector_agent.build_request(
            item_id=item.item_id, round_n=item.current_round,
            model_key=model_key_for(cfg, "reflector"),
            prior_rounds=list(item.rounds_history) + [round_obj],
            domain_name=domain.name, leakage_rules=domain.leakage_rules(),
            acceptance=cfg.acceptance, harness=harness,
        )
        new_state = replace(item, state=State.NEED_REFLECTION,
                            weak_scores=tuple(new_weak), strong_scores=tuple(new_strong),
                            rounds_history=item.rounds_history + (round_obj,))
        return StepResult(
            state=new_state, new_requests=tuple(new_requests) + (rreq,),
            completed_round=round_obj, scores_to_persist=tuple(new_scores),
        )

    new_state = replace(item, state=State.REJECTED,
                        weak_scores=tuple(new_weak), strong_scores=tuple(new_strong),
                        rounds_history=item.rounds_history + (round_obj,),
                        rejection_reasons=tuple(ev.rejection_reasons))
    return StepResult(
        state=new_state, new_requests=tuple(new_requests),
        completed_round=round_obj, scores_to_persist=tuple(new_scores),
    )


def _on_reflection(item, relevant, cfg, harness, domain, grounding) -> StepResult:
    resp = _find_role(relevant, "reflector")
    if resp is None:
        return StepResult(state=item)
    reflection = reflector_agent.parse_response(resp.text)
    feedback = list(reflection.feedback)
    if reflection.new_angle:
        feedback.append(f"NEW_ANGLE: {reflection.new_angle}")
    # Advance round, clear per-round accumulators, emit next challenger.
    next_round = item.current_round + 1
    bumped = replace(
        item,
        current_round=next_round,
        candidate=None,
        quality=None,
        weak_scores=(),
        strong_scores=(),
        last_feedback=tuple(feedback),
        judge_response_for_solver={},
    )
    return _emit_challenger(bumped, cfg, harness, domain, grounding)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_role(responses: list[StepResponse], role: str) -> Optional[StepResponse]:
    for r in responses:
        if r.role == role:
            return r
    return None


def _solver_role_from_judge(judge_resp: StepResponse) -> Optional[str]:
    return judge_resp.solver_role


def _go_to_reflection_or_reject(item, cfg, harness, domain, grounding, *,
                                failure_quality: QualityCheck) -> StepResult:
    """Drive the unified rejection path: emit reflector or terminate REJECTED."""
    cand = item.candidate or Candidate(
        candidate_id="invalid", domain=item.domain, source_id=item.source_id,
        payload={}, rubric=[], reference_output=None,
    )
    round_obj = Round(
        refinement_round=item.current_round,
        candidate=cand,
        quality=failure_quality,
        evaluation=None,
        reflection=None,
        ended_at=datetime.now(timezone.utc),
    )
    if item.current_round < cfg.loop.max_rounds:
        rreq = reflector_agent.build_request(
            item_id=item.item_id, round_n=item.current_round,
            model_key=model_key_for(cfg, "reflector"),
            prior_rounds=list(item.rounds_history) + [round_obj],
            domain_name=domain.name, leakage_rules=domain.leakage_rules(),
            acceptance=cfg.acceptance, harness=harness,
        )
        new_state = replace(item, state=State.NEED_REFLECTION, quality=failure_quality,
                            rounds_history=item.rounds_history + (round_obj,))
        return StepResult(state=new_state, new_requests=(rreq,), completed_round=round_obj)
    new_state = replace(
        item, state=State.REJECTED, quality=failure_quality,
        rounds_history=item.rounds_history + (round_obj,),
        rejection_reasons=tuple(failure_quality.failures or ["unspecified"]),
    )
    return StepResult(state=new_state, new_requests=(), completed_round=round_obj)


def _concat_payload(payload: dict) -> str:
    return " ".join(str(v) for v in payload.values() if isinstance(v, (str, int, float)))
