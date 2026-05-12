"""Run configuration: model providers, agent settings, acceptance thresholds.

Loaded from YAML (or JSON), validated with Pydantic. Env-var interpolation
is supported via `${VAR}` and `${VAR:default}` in any string field.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


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


class ModelConfig(BaseModel):
    """Per-role model settings. `provider_model` is a LiteLLM model string,
    e.g. `openai/gpt-4o-mini`, `anthropic/claude-haiku-4-5`,
    `together_ai/meta-llama/...`, `ollama/llama3`, or `mock/scripted`.
    """

    provider_model: str = "mock/scripted"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)


class AcceptanceConfig(BaseModel):
    """Defaults mirror Agentic Self-Instruct (paper §3): weak ≤ 0.65,
    weak_max ≤ 0.75, no weak-zero, strong ≥ 0.60, strong < 0.95, gap ≥ 0.20.
    """

    weak_avg_max: float = 0.65
    weak_max_max: float = 0.75
    forbid_weak_zero: bool = True
    strong_avg_min: float = 0.60
    strong_avg_max: float = 0.95  # "too easy" ceiling
    min_gap: float = 0.20
    require_quality_passed: bool = True
    rubric_max_weight: int = 7  # paper-recommended cap


class LoopConfig(BaseModel):
    max_rounds: int = 5
    weak_samples: int = 3
    strong_samples: int = 3
    stop_on_first_accept: bool = True


class DomainConfig(BaseModel):
    """Either `name` (registered) or `path` (`module.py:Class` or
    `pkg.module:Class`) selects the domain. `params` is passed to its ctor.
    """

    name: str | None = None
    path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_of(self) -> DomainConfig:
        if not self.name and not self.path:
            raise ValueError("domain.name or domain.path required")
        return self


class SafetyConfig(BaseModel):
    enabled: bool = False
    filter: str | None = None  # "module:attr"


class DispatcherConfig(BaseModel):
    """Settings for the request-fulfillment dispatcher."""

    concurrency: int = 4              # outbound LLM calls in flight per loop tick
    poll_interval_s: float = 5.0      # batch dispatcher polling cadence
    shutdown_grace_s: float = 5.0     # SIGINT/SIGTERM grace before hard exit
    max_request_failures: int = 3     # request retries before terminal REJECTED
    items_per_advance: int = 50       # max items to advance per loop iteration


class MetaOptConfig(BaseModel):
    """Settings for the meta-optimization loop (paper §meta-opt).

    The loop selects parents by Boltzmann sampling on training score, asks
    the mutator LLM to propose an edit, evaluates the child on train items,
    and accepts only if validation also improves on the parent.
    """

    enabled: bool = False
    max_iterations: int = 20
    boltzmann_temp: float = 0.1
    train_size: int = 6
    val_size: int = 4
    val_seed: int = 1  # for deterministic train/val split
    # Mutator model — usually a strong reasoning model. Falls back to orchestrator.
    mutator: ModelConfig | None = None
    # If set, seed the population with the harness JSON at this path; otherwise DEFAULT_HARNESS.
    seed_harness_path: str | None = None
    # Cap inner runs so meta-opt doesn't sprawl.
    inner_max_examples_per_run: int = 8
    inner_max_rounds: int = 3
    # Output sub-directory under the parent run's output_dir.
    output_subdir: str = "metaopt"


class RunConfig(BaseModel):
    run_id: str | None = None  # auto-generated if unset
    output_dir: str = "outputs"
    seed: int = 0

    domain: DomainConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)

    orchestrator: ModelConfig = Field(default_factory=ModelConfig)
    challenger: ModelConfig = Field(default_factory=ModelConfig)
    weak_solver: ModelConfig = Field(default_factory=ModelConfig)
    strong_solver: ModelConfig = Field(default_factory=ModelConfig)
    judge: ModelConfig = Field(default_factory=ModelConfig)

    max_examples: int = 10
    max_concurrency: int = 1  # set >1 for concurrent source-item processing
    request_timeout_s: int = 60
    max_retries: int = 3
    request_budget_usd: float | None = None  # soft budget, advisory only

    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    resume: bool = True
    hf_export: bool = False

    dispatcher: DispatcherConfig = Field(default_factory=DispatcherConfig)
    budget_usd: float | None = None  # null = unlimited

    metaopt: MetaOptConfig = Field(default_factory=MetaOptConfig)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw = _interpolate(raw)
    return RunConfig.model_validate(raw)
