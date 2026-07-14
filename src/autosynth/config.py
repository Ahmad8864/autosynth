"""Pydantic models for YAML/JSON run configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


class _StrictModel(BaseModel):
    """Reject unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else m.group(0))

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


class ModelConfig(_StrictModel):
    """Settings for one role's LiteLLM model.

    A ``None`` temperature leaves the parameter up to the provider.
    """

    provider_model: str = "mock/scripted"
    temperature: float | None = None
    max_tokens: int = 8192
    top_p: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)


class AcceptanceConfig(_StrictModel):
    """Thresholds for rubric, verifiable, and judge-driven acceptance.

    A ``None`` mode uses the domain default.
    """

    mode: Literal["rubric", "verifiable", "judge"] | None = None

    weak_avg_max: float = 0.65
    weak_max_max: float = 0.75
    forbid_weak_zero: bool = True
    strong_avg_min: float = 0.60
    strong_avg_max: float = 0.95
    min_gap: float = 0.20
    require_quality_passed: bool = True
    rubric_max_weight: int = 7

    verifiable_weak_max_correct: int = 1
    verifiable_strong_min_correct: int = 3


class AuditConfig(_StrictModel):
    """Optional final check before a round is accepted."""

    enabled: bool = False
    include_evidence: bool = True
    grounding_chars: int = 4000


class LoopConfig(_StrictModel):
    max_rounds: int = 20
    weak_samples: int = 4
    strong_samples: int = 4
    short_circuit_strong: bool = True


class DomainConfig(_StrictModel):
    """Select a registered domain or a ``module.py:Class`` implementation."""

    name: str | None = None
    path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_of(self) -> DomainConfig:
        if not self.name and not self.path:
            raise ValueError("domain.name or domain.path required")
        return self


class SafetyConfig(_StrictModel):
    enabled: bool = False
    filter: str | None = None  # "module:attr"


class DispatcherConfig(_StrictModel):
    """Settings for the request-fulfillment dispatcher."""

    concurrency: int = 4
    poll_interval_s: float = 5.0
    shutdown_grace_s: float = 5.0
    max_request_failures: int = 3
    items_per_advance: int = 50

    mode: Literal["local", "batch"] = "local"
    batch_provider: str = "openai"
    batch_completion_window: str = "24h"


class MetaOptConfig(_StrictModel):
    """Settings for harness mutation and validation."""

    enabled: bool = False
    max_iterations: int = 20
    boltzmann_temp: float = 0.1
    train_size: int = 6
    val_size: int = 4
    val_seed: int = 1
    max_val_reevals: int = 5
    mutator: ModelConfig | None = None
    seed_harness_path: str | None = None
    inner_max_examples_per_run: int = 8
    inner_max_rounds: int = 3
    output_subdir: str = "metaopt"


class RunConfig(_StrictModel):
    run_id: str | None = None
    output_dir: str = "outputs"

    domain: DomainConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)

    orchestrator: ModelConfig = Field(default_factory=ModelConfig)
    challenger: ModelConfig = Field(default_factory=ModelConfig)
    weak_solver: ModelConfig = Field(default_factory=ModelConfig)
    strong_solver: ModelConfig = Field(default_factory=ModelConfig)
    judge: ModelConfig = Field(default_factory=ModelConfig)
    auditor: ModelConfig | None = None

    max_examples: int = 10
    request_timeout_s: int = 60
    max_retries: int = 3

    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    resume: bool = True

    dispatcher: DispatcherConfig = Field(default_factory=DispatcherConfig)
    budget_usd: float | None = None

    metaopt: MetaOptConfig = Field(default_factory=MetaOptConfig)

    @model_validator(mode="after")
    def _auditor_matches_batch_provider(self) -> RunConfig:
        """Require all batch requests to use the configured provider."""
        if self.audit.enabled and self.auditor is not None and self.dispatcher.mode == "batch":
            provider = self.auditor.provider_model.split("/", 1)[0]
            if provider != self.dispatcher.batch_provider:
                raise ValueError(
                    f"auditor provider {provider!r} does not match dispatcher.batch_provider "
                    f"{self.dispatcher.batch_provider!r}; use dispatcher.mode=local for a "
                    "cross-provider auditor"
                )
        return self


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw = _interpolate(raw)
    return RunConfig.model_validate(raw)


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    for c in (annotation, *get_args(annotation)):
        if isinstance(c, type) and issubclass(c, BaseModel):
            return c
    return None


def _strip_unknown(raw: Any, model: type[BaseModel]) -> Any:
    """Recursively drop keys not declared on ``model`` (and its nested models)."""
    if not isinstance(raw, dict):
        return raw
    out: dict[str, Any] = {}
    for key, value in raw.items():
        field = model.model_fields.get(key)
        if field is None:
            continue
        nested = _nested_model(field.annotation)
        out[key] = _strip_unknown(value, nested) if nested else value
    return out


def load_snapshot(path: str | Path) -> RunConfig:
    """Load an older snapshot after dropping fields no longer in the schema."""
    raw = _interpolate(yaml.safe_load(Path(path).read_text()))
    return RunConfig.model_validate(_strip_unknown(raw, RunConfig))
