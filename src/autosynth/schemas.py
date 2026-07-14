"""Domain-independent data models for the generation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class RubricCriterion(BaseModel):
    """A positive, weighted scoring criterion."""

    id: str
    description: str
    weight: int = Field(ge=1)


class Candidate(BaseModel):
    """A candidate with a domain-defined payload."""

    candidate_id: str
    domain: str
    source_id: str
    payload: dict[str, Any]
    rubric: list[RubricCriterion] = Field(default_factory=list)
    reference_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityCheck(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    notes: str | None = None


class SolverScore(BaseModel):
    """Rubric score or binary verification result for one solver attempt."""

    solver: str
    attempt: int
    raw_response: str
    total: float = Field(ge=0.0, le=1.0)
    per_criterion: dict[str, float] = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)
    correct: bool | None = None


class EvalReport(BaseModel):
    weak_scores: list[SolverScore] = Field(default_factory=list)
    strong_scores: list[SolverScore] = Field(default_factory=list)
    weak_avg: float = 0.0
    weak_max: float = 0.0
    weak_min: float = 0.0
    weak_std: float = 0.0
    strong_avg: float = 0.0
    strong_max: float = 0.0
    strong_min: float = 0.0
    strong_std: float = 0.0
    gap: float = 0.0
    accepted: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    acceptance_rationale: str | None = None


class Round(BaseModel):
    refinement_round: int
    candidate: Candidate
    quality: QualityCheck
    evaluation: EvalReport | None = None
    reflection: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None


class Trajectory(BaseModel):
    trajectory_id: str
    run_id: str
    domain: str
    source_id: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    rounds: list[Round] = Field(default_factory=list)
    final_accepted_round: int | None = None
    total_rounds: int = 0

    def latest_round(self) -> Round | None:
        return self.rounds[-1] if self.rounds else None

    def accepted_round(self) -> Round | None:
        if self.final_accepted_round is None:
            return None
        for r in self.rounds:
            if r.refinement_round == self.final_accepted_round:
                return r
        return None
