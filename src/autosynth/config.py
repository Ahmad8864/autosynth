"""Run configuration: model providers, agent settings, acceptance thresholds.

Loaded from YAML (or JSON), validated with Pydantic. Env-var interpolation
is supported via `${VAR}` and `${VAR:default}` in any string field.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


class _StrictModel(BaseModel):
    """Base for config models: reject unknown keys so typos (e.g. ``weak_avg_maxx``)
    fail loudly instead of silently falling back to defaults."""

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
    """Per-role model settings. `provider_model` is a LiteLLM model string,
    e.g. `openai/gpt-4o-mini`, `anthropic/claude-haiku-4-5`,
    `together_ai/meta-llama/...`, `ollama/llama3`, or `mock/scripted`.

    `temperature=None` means "don't send a temperature param" — the provider
    uses its own default. This keeps the framework opinion-free across models
    with different value constraints (e.g. reasoning models that pin temp=1).
    """

    provider_model: str = "mock/scripted"
    temperature: float | None = None
    max_tokens: int = 8192
    top_p: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)


class AcceptanceConfig(_StrictModel):
    """Acceptance thresholds for both regimes the paper uses.

    ``mode`` selects the regime:

    - ``"rubric"`` (paper §3.1/3.2): an LLM judge scores each rollout against
      a weighted rubric; the ``weak_*``/``strong_*``/``min_gap`` thresholds
      below apply. Defaults mirror the CS main-agent criteria (Fig 7): weak
      ≤ 0.65, weak_max ≤ 0.75, no weak-zero, strong ∈ [0.60, 0.95), gap ≥ 0.20.
    - ``"verifiable"`` (paper §3.3): each rollout is scored by the domain's
      programmatic ``verify()``; acceptance is a count gate ("weak must fail,
      strong must succeed") via the ``verifiable_*`` knobs, which default to the
      paper's 4-rollout setting (weak ≤ 1 correct, strong ≥ 3).
    - ``"judge"`` (paper §3.2 legal loop-judge): after the rubric judge scores
      every rollout, a loop-judge LLM decides accept/improve per round and
      supplies the next-round suggestion — no fixed thresholds.

    ``None`` defers to the domain's ``default_acceptance_mode``.
    """

    mode: Literal["rubric", "verifiable", "judge"] | None = None

    weak_avg_max: float = 0.65
    weak_max_max: float = 0.75
    forbid_weak_zero: bool = True
    strong_avg_min: float = 0.60
    strong_avg_max: float = 0.95  # "too easy" ceiling
    min_gap: float = 0.20
    require_quality_passed: bool = True
    rubric_max_weight: int = 7  # paper-recommended cap

    verifiable_weak_max_correct: int = 1
    verifiable_strong_min_correct: int = 3


class LoopConfig(_StrictModel):
    max_rounds: int = 20
    weak_samples: int = 4
    strong_samples: int = 4
    short_circuit_strong: bool = True  # score the strong solver only when the weak gate passes (cost saver)


class DomainConfig(_StrictModel):
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


class SafetyConfig(_StrictModel):
    enabled: bool = False
    filter: str | None = None  # "module:attr"


class DispatcherConfig(_StrictModel):
    """Settings for the request-fulfillment dispatcher."""

    concurrency: int = 4  # outbound LLM calls in flight per loop tick
    poll_interval_s: float = 5.0  # batch dispatcher polling cadence
    shutdown_grace_s: float = 5.0  # SIGINT/SIGTERM grace before hard exit
    max_request_failures: int = 3  # request retries before terminal REJECTED
    items_per_advance: int = 50  # max items to advance per loop iteration

    mode: Literal["local", "batch"] = "local"  # "batch" = provider batch API: ~50% cheaper, waits the SLA
    batch_provider: str = "openai"  # batch backend; "mock" = in-process provider (offline)
    batch_completion_window: str = "24h"  # provider batch SLA window


class MetaOptConfig(_StrictModel):
    """Settings for the meta-optimization loop (paper §meta-opt).

    The loop selects parents by Boltzmann sampling on their *mean* validation
    score, asks the mutator LLM to propose an edit, and accepts only if the
    child's validation score exceeds the parent's running mean. Accepted
    parents are re-evaluated when re-sampled (up to ``max_val_reevals``) so the
    accept gate isn't decided by a single noisy draw.
    """

    enabled: bool = False
    max_iterations: int = 20
    boltzmann_temp: float = 0.1
    train_size: int = 6
    val_size: int = 4
    val_seed: int = 1  # for deterministic train/val split
    max_val_reevals: int = 5  # cap on val re-evaluations accumulated per harness
    # Mutator model — usually a strong reasoning model. Falls back to orchestrator.
    mutator: ModelConfig | None = None
    # If set, seed the population with the harness JSON at this path; otherwise DEFAULT_HARNESS.
    seed_harness_path: str | None = None
    # Cap inner runs so meta-opt doesn't sprawl.
    inner_max_examples_per_run: int = 8
    inner_max_rounds: int = 3
    # Output sub-directory under the parent run's output_dir.
    output_subdir: str = "metaopt"


class RunConfig(_StrictModel):
    run_id: str | None = None  # auto-generated if unset
    output_dir: str = "outputs"

    domain: DomainConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)

    orchestrator: ModelConfig = Field(default_factory=ModelConfig)
    challenger: ModelConfig = Field(default_factory=ModelConfig)
    weak_solver: ModelConfig = Field(default_factory=ModelConfig)
    strong_solver: ModelConfig = Field(default_factory=ModelConfig)
    judge: ModelConfig = Field(default_factory=ModelConfig)

    max_examples: int = 10
    request_timeout_s: int = 60
    max_retries: int = 3

    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    resume: bool = True

    dispatcher: DispatcherConfig = Field(default_factory=DispatcherConfig)
    budget_usd: float | None = None  # null = unlimited; batch mode counts list price (real spend lower)

    metaopt: MetaOptConfig = Field(default_factory=MetaOptConfig)


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
    """Load a run's config snapshot, tolerating keys removed since it was written.

    Unlike :func:`load_config` (strict), unknown keys are dropped so an in-flight
    run still resumes after a field is removed from the schema."""
    raw = _interpolate(yaml.safe_load(Path(path).read_text()))
    return RunConfig.model_validate(_strip_unknown(raw, RunConfig))
