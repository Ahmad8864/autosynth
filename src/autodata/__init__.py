"""Autodata: agentic synthetic-data generation framework."""

__version__ = "0.1.0"

from autodata.config import (
    AcceptanceConfig,
    DomainConfig,
    LoopConfig,
    MetaOptConfig,
    ModelConfig,
    RunConfig,
    SafetyConfig,
    load_config,
)
from autodata.domain import (
    DomainAdapter,
    GroundingItem,
    bullet_list,
    register_domain,
    rubric,
)
from autodata.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness, make_harness
from autodata.models import LLMClient, register_mock
from autodata.schemas import (
    Candidate,
    EvalReport,
    QualityCheck,
    Round,
    RubricCriterion,
    SolverScore,
    Trajectory,
)

__all__ = [
    "__version__",
    # Schemas
    "Candidate",
    "EvalReport",
    "QualityCheck",
    "Round",
    "RubricCriterion",
    "SolverScore",
    "Trajectory",
    # Domain plugin API
    "DomainAdapter",
    "GroundingItem",
    "register_domain",
    "rubric",
    "bullet_list",
    # Config
    "AcceptanceConfig",
    "DomainConfig",
    "LoopConfig",
    "MetaOptConfig",
    "ModelConfig",
    "RunConfig",
    "SafetyConfig",
    "load_config",
    # Harness / meta-opt
    "HarnessSpec",
    "DEFAULT_HARNESS",
    "make_harness",
    "apply_harness",
    # Provider hooks
    "LLMClient",
    "register_mock",
]
