"""Pure pipeline: state machine that turns responses into requests.

``step()`` is pure: ``(state, fresh_responses) -> (new_state, new_requests,
completed_round?, scores_to_persist?)``. No I/O, no time, no randomness.
The dispatcher orders responses by ``request_id`` before calling ``step()``
so concurrent fulfillment doesn't perturb state evolution.

States (5 + 2 terminal)::

    PENDING → NEED_CANDIDATE → NEED_QUALITY → NEED_SCORES
              ↘ NEED_REFLECTION ↗
              ↘ ACCEPTED | REJECTED

Sync work (structural validation, safety filter, acceptance evaluation)
happens *inside* the response handler for the preceding async state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from autodata.agents import challenger as challenger_agent
from autodata.agents import reflector as reflector_agent
from autodata.agents import solver as solver_agent
from autodata.agents import verifier as verifier_agent
from autodata.config import RunConfig
from autodata.domain import DomainAdapter, GroundingItem
from autodata.evaluator import evaluate
from autodata.harness import DEFAULT_HARNESS, HarnessSpec
from autodata.llm import LLMRequest
from autodata.safety import SafetyFilter
from autodata.schemas import Candidate, QualityCheck, Round, SolverScore


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
    parent_response_id: str | None = None
    solver_response_text: str | None = None  # judge only
    solver_role: str | None = None           # judge only


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
    candidate: Candidate | None = None
    quality: QualityCheck | None = None
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
    completed_round: Round | None = None
    scores_to_persist: tuple[ScoreRecord, ...] = ()


_ROLE_TO_CFG_ATTR: dict[str, str] = {
    "challenger": "challenger",
    "quality": "judge",
    "judge": "judge",
    "weak": "weak_solver",
    "strong": "strong_solver",
    "reflector": "orchestrator",
}


def model_key_for(cfg: RunConfig, role: str) -> str:
    """Look up the configured provider model for a pipeline role."""
    return getattr(cfg, _ROLE_TO_CFG_ATTR[role]).provider_model


def step(
    item: ItemState,
    new_responses: list[StepResponse],
    *,
    cfg: RunConfig,
    harness: HarnessSpec,
    domain: DomainAdapter,
    grounding: GroundingItem,
    safety_filter: SafetyFilter | None = None,
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
        return StepResult(state=item)

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

    struct_errs = domain.validate_candidate(candidate)
    if struct_errs:
        return _go_to_reflection_or_reject(
            replace(item, candidate=candidate), cfg, harness, domain, grounding,
            failure_quality=QualityCheck(
                passed=False, failures=struct_errs, notes="structural validation failed",
            ),
        )

    if cfg.safety.enabled and safety_filter is not None:
        verdict = safety_filter(_payload_text(candidate.payload))
        if not verdict.allowed:
            return _go_to_reflection_or_reject(
                replace(item, candidate=candidate), cfg, harness, domain, grounding,
                failure_quality=QualityCheck(
                    passed=False,
                    failures=[f"safety:{r}" for r in verdict.reasons],
                    notes="safety filter rejected",
                ),
            )

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
        return _go_to_reflection_or_reject(
            replace(item, quality=quality), cfg, harness, domain, grounding,
            failure_quality=quality,
        )

    reqs = tuple(_build_solver_requests(item, cfg, harness, domain))
    new_state = replace(
        item, state=State.NEED_SCORES, quality=quality,
        weak_scores=(), strong_scores=(), judge_response_for_solver={},
    )
    return StepResult(state=new_state, new_requests=reqs)


def _build_solver_requests(item, cfg, harness, domain):
    for role, n in (("weak", cfg.loop.weak_samples), ("strong", cfg.loop.strong_samples)):
        for k in range(n):
            yield solver_agent.build_request(
                item_id=item.item_id, round_n=item.current_round, attempt=k,
                model_key=model_key_for(cfg, role),
                candidate=item.candidate, role=role, domain=domain, harness=harness,
            )


def _on_scores(item, relevant, cfg, harness, domain, grounding) -> StepResult:
    """Two kinds of responses come through here:

      - solver response → emit a judge request (no state change)
      - judge response  → record a SolverScore (no state change *unless*
                          all 2N scores are in, in which case we evaluate)
    """
    judge_requests = _emit_judges_for_new_solvers(item, relevant, cfg, harness, domain)
    new_weak, new_strong, new_scores, judge_for_solver = _ingest_judge_responses(item, relevant)

    have_all_scores = (
        len(new_weak) >= cfg.loop.weak_samples
        and len(new_strong) >= cfg.loop.strong_samples
    )
    if not have_all_scores:
        new_state = replace(
            item,
            weak_scores=tuple(new_weak),
            strong_scores=tuple(new_strong),
            judge_response_for_solver=judge_for_solver,
        )
        return StepResult(
            state=new_state,
            new_requests=tuple(judge_requests),
            scores_to_persist=tuple(new_scores),
        )

    return _finalize_scored_round(
        item, cfg, harness, domain,
        new_weak=new_weak, new_strong=new_strong,
        new_scores=new_scores, judge_requests=judge_requests,
    )


def _emit_judges_for_new_solvers(item, relevant, cfg, harness, domain) -> list[LLMRequest]:
    out: list[LLMRequest] = []
    for r in relevant:
        if r.role not in ("weak", "strong"):
            continue
        out.append(verifier_agent.build_judge_request(
            item_id=item.item_id, round_n=item.current_round, attempt=r.attempt,
            model_key=model_key_for(cfg, "judge"),
            candidate=item.candidate, solver_response=r.text,
            solver_role=r.role, domain=domain, harness=harness,
            parent_response_id=r.request_id,
        ))
    return out


def _ingest_judge_responses(item, relevant):
    new_weak = list(item.weak_scores)
    new_strong = list(item.strong_scores)
    new_scores: list[ScoreRecord] = []
    judge_for_solver = dict(item.judge_response_for_solver)

    for r in relevant:
        if r.role != "judge" or item.candidate is None or r.solver_role is None:
            continue
        score = verifier_agent.parse_judge(
            r.text, candidate=item.candidate, solver_role=r.solver_role,
            attempt=r.attempt, solver_response_text=r.solver_response_text or "",
        )
        (new_weak if r.solver_role == "weak" else new_strong).append(score)
        if r.parent_response_id is not None:
            judge_for_solver[r.parent_response_id] = r.request_id
            new_scores.append(ScoreRecord(
                score=score,
                solver_response_id=r.parent_response_id,
                judge_response_id=r.request_id,
            ))
    return new_weak, new_strong, new_scores, judge_for_solver


def _finalize_scored_round(
    item, cfg, harness, domain,
    *, new_weak, new_strong, new_scores, judge_requests,
) -> StepResult:
    evaluation = evaluate(new_weak, new_strong, item.quality, cfg.acceptance)
    round_obj = Round(
        refinement_round=item.current_round,
        candidate=item.candidate,
        quality=item.quality,
        evaluation=evaluation,
        reflection=None,
        ended_at=datetime.now(timezone.utc),
    )
    rounds_history = item.rounds_history + (round_obj,)
    common = dict(
        weak_scores=tuple(new_weak),
        strong_scores=tuple(new_strong),
        rounds_history=rounds_history,
    )

    if evaluation.accepted:
        new_state = replace(item, state=State.ACCEPTED, **common)
        return StepResult(
            state=new_state, new_requests=tuple(judge_requests),
            completed_round=round_obj, scores_to_persist=tuple(new_scores),
        )

    if item.current_round < cfg.loop.max_rounds:
        rreq = reflector_agent.build_request(
            item_id=item.item_id, round_n=item.current_round,
            model_key=model_key_for(cfg, "reflector"),
            prior_rounds=list(rounds_history),
            domain_name=domain.name, leakage_rules=domain.leakage_rules(),
            acceptance=cfg.acceptance, harness=harness,
        )
        new_state = replace(item, state=State.NEED_REFLECTION, **common)
        return StepResult(
            state=new_state, new_requests=tuple(judge_requests) + (rreq,),
            completed_round=round_obj, scores_to_persist=tuple(new_scores),
        )

    new_state = replace(
        item, state=State.REJECTED,
        rejection_reasons=tuple(evaluation.rejection_reasons),
        **common,
    )
    return StepResult(
        state=new_state, new_requests=tuple(judge_requests),
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
    bumped = replace(
        item,
        current_round=item.current_round + 1,
        candidate=None,
        quality=None,
        weak_scores=(),
        strong_scores=(),
        last_feedback=tuple(feedback),
        judge_response_for_solver={},
    )
    return _emit_challenger(bumped, cfg, harness, domain, grounding)


def _find_role(responses: list[StepResponse], role: str) -> StepResponse | None:
    for r in responses:
        if r.role == role:
            return r
    return None


def _go_to_reflection_or_reject(
    item, cfg, harness, domain, grounding, *, failure_quality: QualityCheck,
) -> StepResult:
    """Unified rejection path: emit reflector or terminate REJECTED."""
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
    rounds_history = item.rounds_history + (round_obj,)
    if item.current_round < cfg.loop.max_rounds:
        rreq = reflector_agent.build_request(
            item_id=item.item_id, round_n=item.current_round,
            model_key=model_key_for(cfg, "reflector"),
            prior_rounds=list(rounds_history),
            domain_name=domain.name, leakage_rules=domain.leakage_rules(),
            acceptance=cfg.acceptance, harness=harness,
        )
        new_state = replace(
            item, state=State.NEED_REFLECTION,
            quality=failure_quality, rounds_history=rounds_history,
        )
        return StepResult(state=new_state, new_requests=(rreq,), completed_round=round_obj)

    new_state = replace(
        item, state=State.REJECTED, quality=failure_quality,
        rounds_history=rounds_history,
        rejection_reasons=tuple(failure_quality.failures or ["unspecified"]),
    )
    return StepResult(state=new_state, new_requests=(), completed_round=round_obj)


def _payload_text(payload: dict) -> str:
    """Concatenate stringy/numeric payload values for the safety filter."""
    return " ".join(str(v) for v in payload.values() if isinstance(v, (str, int, float)))
