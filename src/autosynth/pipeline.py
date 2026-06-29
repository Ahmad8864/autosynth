"""Pure pipeline: state machine that turns responses into requests.

``step()`` is pure: ``(state, fresh_responses) -> (new_state, new_requests,
completed_round?, scores_to_persist?)``. No I/O, no time, no randomness.
The dispatcher orders responses by ``request_id`` before calling ``step()``
so concurrent fulfillment doesn't perturb state evolution.

States (6 + 2 terminal)::

    PENDING → NEED_CANDIDATE → NEED_QUALITY → NEED_SCORES → [NEED_DECISION]
              ↘ NEED_REFLECTION ↗
              ↘ ACCEPTED | REJECTED

NEED_DECISION only occurs under a judge-decided policy: once all rollouts are
scored, a loop-judge LLM decides accept/improve (paper §3.2). Sync work
(structural validation, safety filter, threshold/count evaluation) happens
*inside* the response handler for the preceding async state.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from autosynth.acceptance import AcceptancePolicy, Decision, ThresholdPolicy, weak_gate_report
from autosynth.agents import challenger as challenger_agent
from autosynth.agents import loop_judge as loop_judge_agent
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
    NEED_DECISION = "NEED_DECISION"
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


@dataclass(frozen=True)
class StepContext:
    """Invariant dependencies for one ``step()`` call.

    Built once in :func:`step` (from the resolved harness/policy) and threaded
    unchanged through every state handler. Holds only what stays constant across
    a transition; the varying ``item`` (reduction state) and ``new_responses``
    (fold input) remain explicit parameters.
    """

    cfg: RunConfig
    harness: HarnessSpec
    domain: DomainAdapter
    grounding: GroundingItem | None
    policy: AcceptancePolicy
    safety_filter: SafetyFilter | None = None


_ROLE_TO_CFG_ATTR: dict[str, str] = {
    "challenger": "challenger",
    "quality": "judge",
    "judge": "judge",
    "loop_judge": "judge",
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

    ctx = StepContext(
        cfg=cfg,
        harness=harness or DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
        policy=policy or ThresholdPolicy(cfg.acceptance),
        safety_filter=safety_filter,
    )

    if item.state == State.PENDING:
        return _emit_challenger(item, ctx)
    if item.state == State.NEED_CANDIDATE:
        return _on_challenger(item, relevant, ctx)
    if item.state == State.NEED_QUALITY:
        return _on_quality(item, relevant, ctx)
    if item.state == State.NEED_SCORES:
        return _on_scores(item, relevant, ctx)
    if item.state == State.NEED_DECISION:
        return _on_decision(item, relevant, ctx)
    if item.state == State.NEED_REFLECTION:
        return _on_reflection(item, relevant, ctx)
    raise ValueError(f"unknown state {item.state!r}")


def _emit_challenger(item: ItemState, ctx: StepContext) -> StepResult:
    grounding = ctx.grounding
    assert grounding is not None, "dispatcher rejects items with missing grounding before step()"
    req = challenger_agent.build_request(
        item_id=item.item_id,
        round_n=item.current_round,
        **_dispatch_kwargs(ctx.cfg, "challenger"),
        grounding=grounding,
        feedback=list(item.last_feedback),
        prior_payloads=[r.candidate.payload for r in item.rounds_history],
        domain=ctx.domain,
        harness=ctx.harness,
    )
    new_state = replace(item, state=State.NEED_CANDIDATE)
    return StepResult(state=new_state, new_requests=(req,))


def _on_challenger(item: ItemState, relevant: list[StepResponse], ctx: StepContext) -> StepResult:
    resp = _find_role(relevant, "challenger")
    if resp is None:
        return StepResult(state=item)

    try:
        candidate = challenger_agent.parse_response(
            resp.text,
            source_id=item.source_id,
            round_n=item.current_round,
            domain_name=ctx.domain.name,
            source_metadata=item.source_metadata,
            rubric_max_weight=ctx.harness.rubric_max_weight or ctx.cfg.acceptance.rubric_max_weight,
        )
    except ValueError as e:
        return _go_to_reflection_or_reject(
            item,
            ctx,
            failure_quality=QualityCheck(passed=False, failures=[f"challenger_parse_error:{e}"]),
        )

    struct_errs = ctx.domain.validate_candidate(candidate)
    if struct_errs:
        return _go_to_reflection_or_reject(
            replace(item, candidate=candidate),
            ctx,
            failure_quality=QualityCheck(
                passed=False,
                failures=struct_errs,
                notes="structural validation failed",
            ),
        )

    if ctx.cfg.safety.enabled and ctx.safety_filter is not None:
        try:
            verdict = ctx.safety_filter(_safety_text(candidate))
        except Exception as e:
            # User-supplied filter: fail closed rather than let it escape step().
            return _go_to_reflection_or_reject(
                replace(item, candidate=candidate),
                ctx,
                failure_quality=QualityCheck(
                    passed=False,
                    failures=[f"safety:filter_error:{e}"],
                    notes="safety filter raised",
                ),
            )
        if not verdict.allowed:
            return _go_to_reflection_or_reject(
                replace(item, candidate=candidate),
                ctx,
                failure_quality=QualityCheck(
                    passed=False,
                    failures=[f"safety:{r}" for r in verdict.reasons],
                    notes="safety filter rejected",
                ),
            )

    qreq = verifier_agent.build_quality_request(
        item_id=item.item_id,
        round_n=item.current_round,
        **_dispatch_kwargs(ctx.cfg, "quality"),
        candidate=candidate,
        domain=ctx.domain,
        harness=ctx.harness,
    )
    new_state = replace(item, state=State.NEED_QUALITY, candidate=candidate)
    return StepResult(state=new_state, new_requests=(qreq,))


def _on_quality(item: ItemState, relevant: list[StepResponse], ctx: StepContext) -> StepResult:
    resp = _find_role(relevant, "quality")
    if resp is None:
        return StepResult(state=item)

    quality = verifier_agent.parse_quality(resp.text)
    if not quality.passed:
        return _go_to_reflection_or_reject(
            replace(item, quality=quality),
            ctx,
            failure_quality=quality,
        )

    # Short-circuit mode fans out only weak solvers first; strong is emitted
    # later (in _on_scores) iff the weak gate passes.
    roles = ("weak",) if ctx.cfg.loop.short_circuit_strong else ("weak", "strong")
    reqs = tuple(_build_solver_requests(item, ctx, roles))
    new_state = replace(
        item,
        state=State.NEED_SCORES,
        quality=quality,
        weak_scores=(),
        strong_scores=(),
    )
    return StepResult(state=new_state, new_requests=reqs)


def _build_solver_requests(
    item: ItemState, ctx: StepContext, roles: tuple[str, ...] = ("weak", "strong")
) -> Iterator[LLMRequest]:
    candidate = item.candidate
    assert candidate is not None  # set on entry to NEED_QUALITY, never cleared mid-round
    counts = {"weak": ctx.cfg.loop.weak_samples, "strong": ctx.cfg.loop.strong_samples}
    for role in roles:
        for k in range(counts[role]):
            yield solver_agent.build_request(
                item_id=item.item_id,
                round_n=item.current_round,
                attempt=k,
                **_dispatch_kwargs(ctx.cfg, role),
                candidate=candidate,
                role=role,
                domain=ctx.domain,
                harness=ctx.harness,
            )


def _on_scores(item: ItemState, relevant: list[StepResponse], ctx: StepContext) -> StepResult:
    """Score new solver attempts; finalize once every rollout is scored.

    Scoring: rubric mode (``policy.requires_judge``) emits a judge per solver
    response; verifiable mode scores in-process via ``domain.verify()``. With
    ``short_circuit_strong``, weak runs first and strong is emitted only if the
    weak gate passes. Deciding: a judge-decided policy defers to a loop-judge
    (NEED_DECISION); threshold/verifiable policies decide synchronously here.
    """
    candidate, quality = item.candidate, item.quality
    assert candidate is not None and quality is not None  # guaranteed in NEED_SCORES
    if ctx.policy.requires_judge:
        new_requests, scores = _score_via_judge(item, relevant, ctx)
    else:
        new_requests, scores = _score_via_verify(item, relevant, ctx)

    weak_done = len(scores.weak_scores) >= ctx.cfg.loop.weak_samples

    # Conditional strong eval (opt-in): once weak is done, run strong only if the
    # weak gate passes. Re-emit is idempotent (stable ids); strong_samples>0 avoids
    # stranding a degenerate config here.
    if (
        ctx.cfg.loop.short_circuit_strong
        and ctx.cfg.loop.strong_samples > 0
        and weak_done
        and len(scores.strong_scores) == 0
    ):
        if ctx.policy.weak_gate_passes(scores.weak_scores):
            strong_reqs = tuple(_build_solver_requests(item, ctx, ("strong",)))
            new_state = replace(
                item,
                weak_scores=scores.weak_scores,
                strong_scores=scores.strong_scores,
            )
            return StepResult(
                state=new_state,
                new_requests=tuple(new_requests) + strong_reqs,
                scores_to_persist=scores.records,
            )
        # Weak gate failed (too easy): skip strong, finalize the round now.
        return _finalize_decision(
            item,
            ctx,
            decision=Decision(report=weak_gate_report(scores.weak_scores)),
            scores_to_persist=scores.records,
            extra_requests=tuple(new_requests),
            use_reflector=True,
        )

    have_all_scores = weak_done and len(scores.strong_scores) >= ctx.cfg.loop.strong_samples
    if not have_all_scores:
        new_state = replace(
            item,
            weak_scores=scores.weak_scores,
            strong_scores=scores.strong_scores,
        )
        return StepResult(
            state=new_state,
            new_requests=tuple(new_requests),
            scores_to_persist=scores.records,
        )

    # All rollouts scored. A judge-decided policy runs the loop-judge (async, via
    # NEED_DECISION); threshold/verifiable policies decide synchronously here.
    if ctx.policy.decides_async:
        req = loop_judge_agent.build_request(
            item_id=item.item_id,
            round_n=item.current_round,
            **_dispatch_kwargs(ctx.cfg, "loop_judge"),
            candidate=candidate,
            weak_scores=scores.weak_scores,
            strong_scores=scores.strong_scores,
            quality=quality,
            harness=ctx.harness,
        )
        new_state = replace(
            item,
            state=State.NEED_DECISION,
            weak_scores=scores.weak_scores,
            strong_scores=scores.strong_scores,
        )
        return StepResult(
            state=new_state,
            new_requests=tuple(new_requests) + (req,),
            scores_to_persist=scores.records,
        )

    decision = Decision(report=ctx.policy.evaluate(scores.weak_scores, scores.strong_scores, quality))
    return _finalize_decision(
        item,
        ctx,
        decision=decision,
        scores_to_persist=scores.records,
        extra_requests=tuple(new_requests),
        use_reflector=True,
    )


def _score_via_judge(
    item: ItemState, relevant: list[StepResponse], ctx: StepContext
) -> tuple[list[LLMRequest], ScoreIngestion]:
    """Rubric mode: emit a judge per new solver response and ingest judge scores."""
    judge_requests = _emit_judges_for_new_solvers(item, relevant, ctx)
    scores = _ingest_judge_responses(item, relevant)
    return judge_requests, scores


def _score_via_verify(
    item: ItemState, relevant: list[StepResponse], ctx: StepContext
) -> tuple[list[LLMRequest], ScoreIngestion]:
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
        score = _verify_score(ctx.domain, item.candidate, r)
        (new_weak if r.role == "weak" else new_strong).append(score)
        records.append(
            ScoreRecord(score=score, solver_response_id=r.request_id, judge_response_id=r.request_id)
        )
    scores = ScoreIngestion(
        weak_scores=tuple(new_weak),
        strong_scores=tuple(new_strong),
        records=tuple(records),
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


def _emit_judges_for_new_solvers(
    item: ItemState, relevant: list[StepResponse], ctx: StepContext
) -> list[LLMRequest]:
    candidate = item.candidate
    assert candidate is not None  # NEED_SCORES guarantees a candidate
    out: list[LLMRequest] = []
    for r in relevant:
        if r.role not in ("weak", "strong"):
            continue
        out.append(
            verifier_agent.build_judge_request(
                item_id=item.item_id,
                round_n=item.current_round,
                attempt=r.attempt,
                **_dispatch_kwargs(ctx.cfg, "judge"),
                candidate=candidate,
                solver_response=r.text,
                solver_role=r.role,
                domain=ctx.domain,
                harness=ctx.harness,
                parent_response_id=r.request_id,
            )
        )
    return out


def _ingest_judge_responses(item, relevant):
    new_weak = list(item.weak_scores)
    new_strong = list(item.strong_scores)
    seen = {(s.solver, s.attempt) for s in (*new_weak, *new_strong)}
    new_scores: list[ScoreRecord] = []

    for r in relevant:
        if r.role != "judge" or item.candidate is None or r.solver_role is None:
            continue
        if (r.solver_role, r.attempt) in seen:
            continue  # re-delivered on resume; store already deduped, don't double-count in memory
        seen.add((r.solver_role, r.attempt))
        score = verifier_agent.parse_judge(
            r.text,
            candidate=item.candidate,
            solver_role=r.solver_role,
            attempt=r.attempt,
            solver_response_text=r.solver_response_text or "",
        )
        (new_weak if r.solver_role == "weak" else new_strong).append(score)
        if r.parent_response_id is not None:
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
    )


def _on_decision(item: ItemState, relevant: list[StepResponse], ctx: StepContext) -> StepResult:
    """A judge-decided policy: turn the loop-judge response into accept/improve."""
    resp = _find_role(relevant, "loop_judge")
    if resp is None:
        return StepResult(state=item)
    verdict = loop_judge_agent.parse_verdict(resp.text)
    quality = item.quality
    assert quality is not None  # NEED_DECISION guarantees quality
    decision = ctx.policy.decide(verdict, item.weak_scores, item.strong_scores, quality)
    return _finalize_decision(
        item,
        ctx,
        decision=decision,
        scores_to_persist=(),  # scores were persisted at the NEED_SCORES→NEED_DECISION transition
        extra_requests=(),
        use_reflector=False,
    )


def _finalize_decision(
    item: ItemState,
    ctx: StepContext,
    *,
    decision: Decision,
    scores_to_persist: tuple[ScoreRecord, ...] = (),
    extra_requests: tuple[LLMRequest, ...] = (),
    use_reflector: bool,
) -> StepResult:
    """Turn a Decision into the next state: accept, reflect/improve, or reject.

    ``use_reflector`` selects the improve path: sync policies run the reflector
    (NEED_REFLECTION); the judge policy supplies ``decision.feedback`` and goes
    straight to the next challenger round.
    """
    candidate, quality = item.candidate, item.quality
    assert candidate is not None and quality is not None  # set since NEED_SCORES
    report = decision.report
    round_obj = Round(
        refinement_round=item.current_round,
        candidate=candidate,
        quality=quality,
        evaluation=report,
        reflection=None,
        ended_at=datetime.now(timezone.utc),
    )
    rounds_history = item.rounds_history + (round_obj,)
    common = dict(
        weak_scores=item.weak_scores,
        strong_scores=item.strong_scores,
        rounds_history=rounds_history,
    )

    if report.accepted:
        new_state = replace(item, state=State.ACCEPTED, **common)
        return StepResult(
            state=new_state,
            new_requests=extra_requests,
            completed_round=round_obj,
            scores_to_persist=scores_to_persist,
        )

    if item.current_round < ctx.cfg.loop.max_rounds:
        if use_reflector:
            rreq = reflector_agent.build_request(
                item_id=item.item_id,
                round_n=item.current_round,
                **_dispatch_kwargs(ctx.cfg, "reflector"),
                prior_rounds=list(rounds_history),
                domain_name=ctx.domain.name,
                leakage_rules=ctx.domain.leakage_rules(),
                weak_ceiling=ctx.policy.weak_ceiling,
                strong_floor=ctx.policy.strong_floor,
                harness=ctx.harness,
            )
            new_state = replace(item, state=State.NEED_REFLECTION, **common)
            return StepResult(
                state=new_state,
                new_requests=extra_requests + (rreq,),
                completed_round=round_obj,
                scores_to_persist=scores_to_persist,
            )
        # Judge-improve: the loop-judge already gave the suggestion; bump the round
        # with it as feedback (skip the reflector) and emit the next challenger.
        # Scores were persisted at NEED_DECISION entry; insert_score files under the
        # (now-bumped) current_round, so none may ride along here.
        assert not scores_to_persist, (
            "judge-improve must not carry scores (they misfile under the bumped round)"
        )
        bumped = replace(
            item,
            current_round=item.current_round + 1,
            candidate=None,
            quality=None,
            weak_scores=(),
            strong_scores=(),
            last_feedback=decision.feedback,
            rounds_history=rounds_history,
        )
        result = _emit_challenger(bumped, ctx)
        return replace(result, new_requests=result.new_requests + extra_requests, completed_round=round_obj)

    new_state = replace(
        item,
        state=State.REJECTED,
        rejection_reasons=tuple(report.rejection_reasons),
        **common,
    )
    return StepResult(
        state=new_state,
        new_requests=extra_requests,
        completed_round=round_obj,
        scores_to_persist=scores_to_persist,
    )


def _on_reflection(item: ItemState, relevant: list[StepResponse], ctx: StepContext) -> StepResult:
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
    )
    return _emit_challenger(bumped, ctx)


def _find_role(responses: list[StepResponse], role: str) -> StepResponse | None:
    for r in responses:
        if r.role == role:
            return r
    return None


def _go_to_reflection_or_reject(
    item: ItemState,
    ctx: StepContext,
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
    if item.current_round < ctx.cfg.loop.max_rounds:
        rreq = reflector_agent.build_request(
            item_id=item.item_id,
            round_n=item.current_round,
            **_dispatch_kwargs(ctx.cfg, "reflector"),
            prior_rounds=list(rounds_history),
            domain_name=ctx.domain.name,
            leakage_rules=ctx.domain.leakage_rules(),
            weak_ceiling=ctx.policy.weak_ceiling,
            strong_floor=ctx.policy.strong_floor,
            harness=ctx.harness,
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


def _safety_text(candidate: Candidate) -> str:
    """Text scanned by the safety filter: every nested string/number value in the
    payload plus the reference answer (where a leaked PII value most often lands)."""
    parts: list[str] = []
    _collect_text(candidate.payload, parts)
    if candidate.reference_output:
        parts.append(candidate.reference_output)
    return " ".join(parts)


def _collect_text(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, (int, float)):  # bool is an int subclass; harmless
        out.append(str(value))
    elif isinstance(value, dict):
        for v in value.values():
            _collect_text(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_text(v, out)
