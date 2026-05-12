"""Run configuration: model providers, agent settings, acceptance thresholds.

Loaded from YAML (or JSON), validated with Pydantic. Env-var interpolation
is supported via `${VAR}` and `${VAR:default}` in any string field.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

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

    name: Optional[str] = None
    path: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_of(self) -> "DomainConfig":
        if not self.name and not self.path:
            raise ValueError("domain.name or domain.path required")
        return self


class SafetyConfig(BaseModel):
    enabled: bool = False
    filter: Optional[str] = None  # "module:attr"


class RunConfig(BaseModel):
    run_id: Optional[str] = None  # auto-generated if unset
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
    request_budget_usd: Optional[float] = None  # soft budget, advisory only

    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    resume: bool = True
    hf_export: bool = False


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw = _interpolate(raw)
    return RunConfig.model_validate(raw)
