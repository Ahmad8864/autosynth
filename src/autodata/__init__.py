"""Autodata: agentic synthetic-data generation framework."""
from autodata.schemas import (
    Candidate,
    EvalReport,
    Round,
    SolverScore,
    Trajectory,
)
from autodata.domain import DomainAdapter, GroundingItem, register_domain
from autodata.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness, make_harness

__all__ = [
    "Candidate",
    "EvalReport",
    "Round",
    "SolverScore",
    "Trajectory",
    "DomainAdapter",
    "GroundingItem",
    "register_domain",
    "HarnessSpec",
    "DEFAULT_HARNESS",
    "make_harness",
    "apply_harness",
]

__version__ = "0.1.0"
