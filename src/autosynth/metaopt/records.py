"""Record types persisted across meta-optimization iterations."""

from __future__ import annotations

from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from autosynth.harness import HarnessSpec


class HarnessRecord(BaseModel):
    spec: HarnessSpec
    train_score: float = 0.0
    # All validation scores for this harness. The paper accumulates re-evals
    # whenever an accepted candidate is sampled as a parent and compares the
    # *mean* — so the accept gate isn't decided by a single noisy draw.
    val_scores: list[float] = Field(default_factory=list)
    val_score: float | None = None  # most recent val score (reporting only)
    accepted: bool = True  # seed is always accepted
    parent_accepted_id: str | None = None

    @property
    def val_mean(self) -> float | None:
        return mean(self.val_scores) if self.val_scores else self.val_score


class MetaIteration(BaseModel):
    iteration: int
    parent_id: str
    child_id: str
    mutation: dict[str, Any] = Field(default_factory=dict)
    train_score: float
    val_score: float | None = None
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
