"""Record types persisted across meta-optimization iterations."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from autodata.harness import HarnessSpec


class HarnessRecord(BaseModel):
    spec: HarnessSpec
    train_score: float = 0.0
    val_score: float | None = None
    accepted: bool = True  # seed is always accepted
    parent_accepted_id: str | None = None


class MetaIteration(BaseModel):
    iteration: int
    parent_id: str
    child_id: str
    mutation: dict[str, Any] = Field(default_factory=dict)
    train_score: float
    val_score: float | None = None
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
