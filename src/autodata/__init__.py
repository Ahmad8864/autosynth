"""Autodata: agentic synthetic-data generation framework."""
from autodata.schemas import (
    Candidate,
    EvalReport,
    Round,
    SolverScore,
    Trajectory,
)
from autodata.domain import DomainAdapter, GroundingItem, register_domain

__all__ = [
    "Candidate",
    "EvalReport",
    "Round",
    "SolverScore",
    "Trajectory",
    "DomainAdapter",
    "GroundingItem",
    "register_domain",
]

__version__ = "0.1.0"
