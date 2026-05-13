"""Core data shapes for candidates, rounds, trajectories, and evaluations.

These shapes are domain-agnostic. A domain plugin extends `Candidate.payload`
and `Candidate.rubric` to carry its own structured content.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class RubricCriterion(BaseModel):
    """A single, positive, weighted scoring criterion.

    Following the paper's meta-optimized recipe: positive-only criteria with
    integer weights capped at 7 by default. The cap is enforced at validation
    time by the verifier, not here, so domains can opt out.
    """

    id: str
    description: str
    weight: int = Field(ge=1)


class Candidate(BaseModel):
    """A candidate datapoint emitted by the challenger.

    `payload` is the domain-specific structured content (question, answer,
    context, problem statement, ticket body, etc.). Domains define its shape;
    here it is opaque JSON.
    """

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
    """Score for a single solver invocation, judged against the rubric."""

    solver: str  # "weak" | "strong"
    attempt: int
    raw_response: str
    total: float = Field(ge=0.0, le=1.0)
    per_criterion: dict[str, float] = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    weak_scores: list[SolverScore] = Field(default_factory=list)
    strong_scores: list[SolverScore] = Field(default_factory=list)
    weak_avg: float = 0.0
    weak_max: float = 0.0
    weak_min: float = 0.0
    strong_avg: float = 0.0
    strong_max: float = 0.0
    strong_min: float = 0.0
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
