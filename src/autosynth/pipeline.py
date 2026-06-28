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

from autosynth.acceptance import AcceptancePolicy, ThresholdPolicy
from autosynth.agents import challenger as challenger_agent
from autosynth.agents import reflector as reflector_agent
from autosynth.agents import solver as solver_agent
from autosynth.agents import verifier as verifier_agent
from autosynth.config import ModelConfig, RunConfig
from autosynth.domain import DomainAdapter, GroundingItem
from autosynth.harness import DEFAULT_HARNESS, HarnessSpec
from autosynth.llm import LLMRequest
from autosynth.safety import SafetyFilter
from autosynth.schemas import Candidate, QualityCheck, Round, SolverScore


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
    solver_role: str | None = None  # judge only


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
    current_round: int  # 1-indexed
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


@dataclass(frozen=True)
class ScoreIngestion:
    weak_scores: tuple[SolverScore, ...]
    strong_scores: tuple[SolverScore, ...]
    records: tuple[ScoreRecord, ...]
    judge_response_for_solver: dict[str, str]


_ROLE_TO_CFG_ATTR: dict[str, str] = {
    "challenger": "challenger",
    "quality": "judge",
    "judge": "judge",
    "weak": "weak_solver",
    "strong": "strong_solver",
    "reflector": "orchestrator",
}


def _model_cfg_for(cfg: RunConfig, role: str) -> ModelConfig:
    return getattr(cfg, _ROLE_TO_CFG_ATTR[role])


def model_key_for(cfg: RunConfig, role: str) -> str:
    """Look up the configured provider model for a pipeline role."""
    return _model_cfg_for(cfg, role).provider_model


def _dispatch_kwargs(cfg: RunConfig, role: str) -> dict:
    """Per-role provider/sampling kwargs, ready to splat into a build_request call."""
    mc = _model_cfg_for(cfg, role)
    return {
        "model_key": mc.provider_model,
        "temperature": mc.temperature,
        "max_tokens": mc.max_tokens,
    }


def step(
    item: ItemState,
    new_responses: list[StepResponse],
    *,
    cfg: RunConfig,
    harness: HarnessSpec,
    domain: DomainAdapter,
    grounding: GroundingItem | None,
    safety_filter: SafetyFilter | None = None,
    policy: AcceptancePolicy | None = None,
) -> StepResult:
    """Pure state transition. See module docstring.

    ``policy`` defaults to a rubric-gap :class:`ThresholdPolicy`; the dispatcher
    passes the policy resolved from ``cfg.acceptance.mode`` + domain.
    """
    if item.state in TERMINAL_STATES:
        return StepResult(state=item)

    # Ignore responses from prior rounds (artifacts of resume or out-of-order arrival).
    relevant = [r for r in new_responses if r.round_n == item.current_round]

    h = harness or DEFAULT_HARNESS
    pol = policy or ThresholdPolicy(cfg.acceptance)

    if item.state == State.PENDING:
        return _emit_challenger(item, cfg, h, domain, grounding)
    if item.state == State.NEED_CANDIDATE:
        return _on_challenger(item, relevant, cfg, h, domain, grounding, safety_filter, pol)
    if item.state == State.NEED_QUALITY:
        return _on_quality(item, relevant, cfg, h, domain, grounding, pol)
    if item.state == State.NEED_SCORES:
        return _on_scores(item, relevant, cfg, h, domain, grounding, pol)
    if item.state == State.NEED_REFLECTION:
        return _on_reflection(item, relevant, cfg, h, domain, grounding)
    raise ValueError(f"unknown state {item.state!r}")


def _emit_challenger(item, cfg, harness, domain, grounding) -> StepResult:
    req = challenger_agent.build_request(
        item_id=item.item_id,
        round_n=item.current_round,
        **_dispatch_kwargs(cfg, "challenger"),
        grounding=grounding,
        feedback=list(item.last_feedback),
        prior_payloads=[r.candidate.payload for r in item.rounds_history],
        domain=domain,
        harness=harness,
    )
    new_state = replace(item, state=State.NEED_CANDIDATE)
    return StepResult(state=new_state, new_requests=(req,))


def _on_challenger(item, relevant, cfg, harness, domain, grounding, safety_filter, policy) -> StepResult:
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
            item,
            cfg,
            harness,
            domain,
            grounding,
            policy,
            failure_quality=QualityCheck(passed=False, failures=[f"challenger_parse_error:{e}"]),
        )

    struct_errs = domain.validate_candidate(candidate)
    if struct_errs:
        return _go_to_reflection_or_reject(
            replace(item, candidate=candidate),
            cfg,
            harness,
            domain,
            grounding,
            policy,
            failure_quality=QualityCheck(
                passed=False,
                failures=struct_errs,
                notes="structural validation failed",
            ),
        )

    if cfg.safety.enabled and safety_filter is not None:
        verdict = safety_filter(_payload_text(candidate.payload))
        if not verdict.allowed:
            return _go_to_reflection_or_reject(
                replace(item, candidate=candidate),
                cfg,
                harness,
                domain,
                grounding,
                policy,
                failure_quality=QualityCheck(
                    passed=False,
                    failures=[f"safety:{r}" for r in verdict.reasons],
                    notes="safety filter rejected",
                ),
            )

    qreq = verifier_agent.build_quality_request(
        item_id=item.item_id,
        round_n=item.current_round,
        **_dispatch_kwargs(cfg, "quality"),
        candidate=candidate,
        domain=domain,
        harness=harness,
    )
    new_state = replace(item, state=State.NEED_QUALITY, candidate=candidate)
    return StepResult(state=new_state, new_requests=(qreq,))


def _on_quality(item, relevant, cfg, harness, domain, grounding, policy) -> StepResult:
    resp = _find_role(relevant, "quality")
    if resp is None:
        return StepResult(state=item)

    quality = verifier_agent.parse_quality(resp.text)
    if not quality.passed:
        return _go_to_reflection_or_reject(
            replace(item, quality=quality),
            cfg,
            harness,
            domain,
            grounding,
            policy,
            failure_quality=quality,
        )

    reqs = tuple(_build_solver_requests(item, cfg, harness, domain))
    new_state = replace(
        item,
        state=State.NEED_SCORES,
        quality=quality,
        weak_scores=(),
        strong_scores=(),
        judge_response_for_solver={},
    )
    return StepResult(state=new_state, new_requests=reqs)


def _build_solver_requests(item, cfg, harness, domain):
    for role, n in (("weak", cfg.loop.weak_samples), ("strong", cfg.loop.strong_samples)):
        for k in range(n):
            yield solver_agent.build_request(
                item_id=item.item_id,
                round_n=item.current_round,
                attempt=k,
                **_dispatch_kwargs(cfg, role),
                candidate=item.candidate,
                role=role,
                domain=domain,
                harness=harness,
            )


def _on_scores(item, relevant, cfg, harness, domain, grounding, policy) -> StepResult:
    """Score new solver attempts; finalize once all 2N are in.

    Rubric mode (``policy.requires_judge``) emits a judge request per solver
    response and turns judge responses into scores. Verifiable mode scores each
    solver response in-process via ``domain.verify()``. State only advances when
    every weak and strong attempt has a score.
    """
    if policy.requires_judge:
        new_requests, scores = _score_via_judge(item, relevant, cfg, harness, domain)
    else:
        new_requests, scores = _score_via_verify(item, relevant, domain)

    have_all_scores = (
        len(scores.weak_scores) >= cfg.loop.weak_samples
        and len(scores.strong_scores) >= cfg.loop.strong_samples
    )
    if not have_all_scores:
        new_state = replace(
            item,
            weak_scores=scores.weak_scores,
            strong_scores=scores.strong_scores,
            judge_response_for_solver=scores.judge_response_for_solver,
        )
        return StepResult(
            state=new_state,
            new_requests=tuple(new_requests),
            scores_to_persist=scores.records,
        )

    return _finalize_scored_round(
        item,
        cfg,
        harness,
        domain,
        scores=scores,
        judge_requests=new_requests,
        policy=policy,
    )


def _score_via_judge(item, relevant, cfg, harness, domain):
    """Rubric mode: emit a judge per new solver response and ingest judge scores."""
    judge_requests = _emit_judges_for_new_solvers(item, relevant, cfg, harness, domain)
    scores = _ingest_judge_responses(item, relevant)
    return judge_requests, scores


def _score_via_verify(item, relevant, domain):
    """Verifiable mode: score each new solver response via ``domain.verify()``.

    No judge is dispatched; the SolverScore is binary and the ScoreRecord
    self-references the solver response (there is no judge row). Re-delivered
    attempts are deduped by (solver, attempt) so resume can't double-count.
    """
    new_weak = list(item.weak_scores)
    new_strong = list(item.strong_scores)
    seen = {(s.solver, s.attempt) for s in (*new_weak, *new_strong)}
    records: list[ScoreRecord] = []
    for r in relevant:
        if r.role not in ("weak", "strong") or item.candidate is None:
            continue
        if (r.role, r.attempt) in seen:
            continue
        seen.add((r.role, r.attempt))
        score = _verify_score(domain, item.candidate, r)
        (new_weak if r.role == "weak" else new_strong).append(score)
        records.append(
            ScoreRecord(score=score, solver_response_id=r.request_id, judge_response_id=r.request_id)
        )
    scores = ScoreIngestion(
        weak_scores=tuple(new_weak),
        strong_scores=tuple(new_strong),
        records=tuple(records),
        judge_response_for_solver=item.judge_response_for_solver,
    )
    return [], scores


def _verify_score(domain, candidate, resp) -> SolverScore:
    """Binary SolverScore from a domain verdict. ``total`` is 1.0/0.0; ``correct``
    is the verdict (None when unverifiable). Exceptions are swallowed here — never
    raised into the dispatcher — so a faulty verify() can't livelock an item."""
    common = dict(solver=resp.role, attempt=resp.attempt, raw_response=resp.text)
    try:
        correct = domain.verify(candidate, resp.text)
    except Exception as e:
        return SolverScore(**common, total=0.0, correct=None, failure_modes=[f"verify_error:{e}"])
    if correct is None:
        return SolverScore(**common, total=0.0, correct=None, failure_modes=["unverifiable"])
    return SolverScore(
        **common,
        total=1.0 if correct else 0.0,
        correct=correct,
        failure_modes=[] if correct else ["incorrect"],
    )


def _emit_judges_for_new_solvers(item, relevant, cfg, harness, domain) -> list[LLMRequest]:
    out: list[LLMRequest] = []
    for r in relevant:
        if r.role not in ("weak", "strong"):
            continue
        out.append(
            verifier_agent.build_judge_request(
                item_id=item.item_id,
                round_n=item.current_round,
                attempt=r.attempt,
                **_dispatch_kwargs(cfg, "judge"),
                candidate=item.candidate,
                solver_response=r.text,
                solver_role=r.role,
                domain=domain,
                harness=harness,
                parent_response_id=r.request_id,
            )
        )
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
            r.text,
            candidate=item.candidate,
            solver_role=r.solver_role,
            attempt=r.attempt,
            solver_response_text=r.solver_response_text or "",
        )
        (new_weak if r.solver_role == "weak" else new_strong).append(score)
        if r.parent_response_id is not None:
            judge_for_solver[r.parent_response_id] = r.request_id
            new_scores.append(
                ScoreRecord(
                    score=score,
                    solver_response_id=r.parent_response_id,
                    judge_response_id=r.request_id,
                )
            )
    return ScoreIngestion(
        weak_scores=tuple(new_weak),
        strong_scores=tuple(new_strong),
        records=tuple(new_scores),
        judge_response_for_solver=judge_for_solver,
    )


def _finalize_scored_round(
    item,
    cfg,
    harness,
    domain,
    *,
    scores: ScoreIngestion,
    judge_requests,
    policy,
) -> StepResult:
    evaluation = policy.evaluate(scores.weak_scores, scores.strong_scores, item.quality)
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
        weak_scores=scores.weak_scores,
        strong_scores=scores.strong_scores,
        rounds_history=rounds_history,
    )

    if evaluation.accepted:
        new_state = replace(item, state=State.ACCEPTED, **common)
        return StepResult(
            state=new_state,
            new_requests=tuple(judge_requests),
            completed_round=round_obj,
            scores_to_persist=scores.records,
        )

    if item.current_round < cfg.loop.max_rounds:
        rreq = reflector_agent.build_request(
            item_id=item.item_id,
            round_n=item.current_round,
            **_dispatch_kwargs(cfg, "reflector"),
            prior_rounds=list(rounds_history),
            domain_name=domain.name,
            leakage_rules=domain.leakage_rules(),
            weak_ceiling=policy.weak_ceiling,
            strong_floor=policy.strong_floor,
            harness=harness,
        )
        new_state = replace(item, state=State.NEED_REFLECTION, **common)
        return StepResult(
            state=new_state,
            new_requests=tuple(judge_requests) + (rreq,),
            completed_round=round_obj,
            scores_to_persist=scores.records,
        )

    new_state = replace(
        item,
        state=State.REJECTED,
        rejection_reasons=tuple(evaluation.rejection_reasons),
        **common,
    )
    return StepResult(
        state=new_state,
        new_requests=tuple(judge_requests),
        completed_round=round_obj,
        scores_to_persist=scores.records,
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
    item,
    cfg,
    harness,
    domain,
    grounding,
    policy,
    *,
    failure_quality: QualityCheck,
) -> StepResult:
    """Unified rejection path: emit reflector or terminate REJECTED."""
    cand = item.candidate or Candidate(
        candidate_id="invalid",
        domain=item.domain,
        source_id=item.source_id,
        payload={},
        rubric=[],
        reference_output=None,
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
            item_id=item.item_id,
            round_n=item.current_round,
            **_dispatch_kwargs(cfg, "reflector"),
            prior_rounds=list(rounds_history),
            domain_name=domain.name,
            leakage_rules=domain.leakage_rules(),
            weak_ceiling=policy.weak_ceiling,
            strong_floor=policy.strong_floor,
            harness=harness,
        )
        new_state = replace(
            item,
            state=State.NEED_REFLECTION,
            quality=failure_quality,
            rounds_history=rounds_history,
        )
        return StepResult(state=new_state, new_requests=(rreq,), completed_round=round_obj)

    new_state = replace(
        item,
        state=State.REJECTED,
        quality=failure_quality,
        rounds_history=rounds_history,
        rejection_reasons=tuple(failure_quality.failures or ["unspecified"]),
    )
    return StepResult(state=new_state, new_requests=(), completed_round=round_obj)


def _payload_text(payload: dict) -> str:
    """Concatenate stringy/numeric payload values for the safety filter."""
    return " ".join(str(v) for v in payload.values() if isinstance(v, (str, int, float)))
