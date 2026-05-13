"""Verifier / Judge: quality-check and per-solver rubric scoring."""

from __future__ import annotations

from autodata.domain import DomainAdapter
from autodata.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness
from autodata.llm import LLMRequest
from autodata.schemas import Candidate, QualityCheck, SolverScore
from autodata.utils import clamp, extract_json, stable_id


def build_quality_request(
    *,
    item_id: str,
    round_n: int,
    model_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    candidate: Candidate,
    domain: DomainAdapter,
    harness: HarnessSpec | None = None,
) -> LLMRequest:
    h = harness or DEFAULT_HARNESS
    messages = domain.quality_prompt(candidate)
    messages = apply_harness(messages, h.rules_for("quality"))
    return LLMRequest(
        request_id=stable_id(item_id, round_n, "quality", 0),
        item_id=item_id,
        round_n=round_n,
        role="quality",
        model_key=model_key,
        messages=messages,
        json_mode=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def parse_quality(text: str) -> QualityCheck:
    """Parse a quality-verifier response. Returns a failing check on parse error."""
    try:
        data = extract_json(text)
    except ValueError as e:
        return QualityCheck(passed=False, failures=[f"quality_judge_parse_error:{e}"])
    return QualityCheck(
        passed=bool(data.get("passed", False)),
        failures=[str(x) for x in (data.get("failures") or [])],
        notes=data.get("notes"),
    )


def build_judge_request(
    *,
    item_id: str,
    round_n: int,
    attempt: int,
    model_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    candidate: Candidate,
    solver_response: str,
    solver_role: str,
    domain: DomainAdapter,
    harness: HarnessSpec | None = None,
    parent_response_id: str,
) -> LLMRequest:
    """Build a judge request that scores one solver attempt.

    The request_id is keyed on parent_response_id so distinct solver attempts
    produce distinct judge requests even if attempt indices collide.
    """
    h = harness or DEFAULT_HARNESS
    messages = domain.judge_prompt(candidate, solver_response, solver_role)
    messages = apply_harness(messages, h.rules_for("judge"))
    return LLMRequest(
        request_id=stable_id(item_id, round_n, "judge", solver_role, parent_response_id),
        item_id=item_id,
        round_n=round_n,
        role="judge",
        model_key=model_key,
        messages=messages,
        attempt=attempt,
        json_mode=True,
        parent_response_id=parent_response_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def parse_judge(
    text: str,
    *,
    candidate: Candidate,
    solver_role: str,
    attempt: int,
    solver_response_text: str,
) -> SolverScore:
    """Parse a judge response into a SolverScore against the candidate's rubric."""
    try:
        data = extract_json(text)
    except ValueError as e:
        return SolverScore(
            solver=solver_role,
            attempt=attempt,
            raw_response=solver_response_text,
            total=0.0,
            failure_modes=[f"judge_parse_error:{e}"],
        )

    per: dict[str, float] = {}
    for k, v in (data.get("per_criterion") or {}).items():
        try:
            per[str(k)] = clamp(float(v))
        except (TypeError, ValueError):
            continue

    if "total" in data:
        try:
            total = clamp(float(data["total"]))
        except (TypeError, ValueError):
            total = _weighted_average(per, candidate)
    else:
        total = _weighted_average(per, candidate)

    return SolverScore(
        solver=solver_role,
        attempt=attempt,
        raw_response=solver_response_text,
        total=total,
        per_criterion=per,
        failure_modes=[str(x) for x in (data.get("failure_modes") or [])],
    )


def _weighted_average(per: dict[str, float], candidate: Candidate) -> float:
    weights = {c.id: c.weight for c in candidate.rubric}
    num = sum(per.get(cid, 0.0) * w for cid, w in weights.items())
    den = sum(weights.values()) or 1
    return clamp(num / den)
