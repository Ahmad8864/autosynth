"""Verifier/Judge — runs both the quality audit and the rubric scoring."""
from __future__ import annotations

from typing import Any

from loguru import logger

from autodata.domain import DomainAdapter
from autodata.models import LLMClient
from autodata.schemas import Candidate, QualityCheck, SolverScore
from autodata.utils import clamp


class VerifierJudge:
    def __init__(self, client: LLMClient, domain: DomainAdapter):
        self.client = client
        self.domain = domain

    def quality_check(self, candidate: Candidate) -> QualityCheck:
        messages = self.domain.quality_prompt(candidate)
        try:
            data = self.client.complete_json(messages)
        except Exception as e:
            logger.warning("quality check parse failure: {}", e)
            return QualityCheck(passed=False, failures=[f"quality_judge_parse_error:{e}"])
        return QualityCheck(
            passed=bool(data.get("passed", False)),
            failures=[str(x) for x in (data.get("failures") or [])],
            notes=data.get("notes"),
        )

    def score(self, candidate: Candidate, solver_response: str, solver_role: str, attempt: int) -> SolverScore:
        messages = self.domain.judge_prompt(candidate, solver_response, solver_role)
        try:
            data = self.client.complete_json(messages)
        except Exception as e:
            logger.warning("judge parse failure: {}", e)
            return SolverScore(
                solver=solver_role,
                attempt=attempt,
                raw_response=solver_response,
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
            raw_response=solver_response,
            total=total,
            per_criterion=per,
            failure_modes=[str(x) for x in (data.get("failure_modes") or [])],
        )


def _weighted_average(per: dict[str, float], candidate: Candidate) -> float:
    weights = {c.id: c.weight for c in candidate.rubric}
    num = sum(per.get(cid, 0.0) * w for cid, w in weights.items())
    den = sum(weights.values()) or 1
    return clamp(num / den)
