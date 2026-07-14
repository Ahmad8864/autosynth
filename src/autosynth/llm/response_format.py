"""Response-format selection: strict schemas for fixed-shape roles on providers
that support structured outputs, plain JSON mode otherwise."""

from __future__ import annotations

from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from autosynth.llm.types import LLMRequest


class QualityOutput(BaseModel):
    passed: bool
    failures: list[str]
    notes: str | None = None


class LoopJudgeOutput(BaseModel):
    verdict: Literal["accept", "improve"]
    grpo_suitability: Literal["high", "medium", "low"]
    reason: str
    suggestion: str


class ReflectorOutput(BaseModel):
    feedback: list[str]
    new_angle: str


class RubricCriterionOutput(BaseModel):
    id: str
    description: str
    weight: int = Field(ge=1)


class ChallengerEnvelope(BaseModel):
    """Generic challenger candidate envelope — the fallback when a domain
    doesn't declare a stricter payload schema."""

    payload: dict[str, Any]
    reference_output: str
    rubric: list[RubricCriterionOutput]


class JudgeOutput(BaseModel):
    # `per_criterion` is an open map (runtime rubric ids), outside the provider
    # strict-schema subset, so the guard below degrades judge to plain JSON mode.
    per_criterion: dict[str, float]
    total: float
    failure_modes: list[str] = Field(default_factory=list)


class MutatorOutput(BaseModel):
    rationale: str
    challenger_rules_add: list[str] = Field(default_factory=list)
    challenger_rules_remove_indices: list[int] = Field(default_factory=list)
    quality_rules_add: list[str] = Field(default_factory=list)
    quality_rules_remove_indices: list[int] = Field(default_factory=list)
    judge_rules_add: list[str] = Field(default_factory=list)
    judge_rules_remove_indices: list[int] = Field(default_factory=list)
    solver_rules_add: list[str] = Field(default_factory=list)
    solver_rules_remove_indices: list[int] = Field(default_factory=list)
    reflector_rules_add: list[str] = Field(default_factory=list)
    reflector_rules_remove_indices: list[int] = Field(default_factory=list)
    rubric_max_weight: int | None = None
    require_self_test: bool | None = None


# Fixed-shape roles: schema derivable from role alone. Challenger is absent —
# its schema is domain-specific and attached to the request as `response_schema`.
_MODELS: dict[str, type[BaseModel]] = {
    "quality": QualityOutput,
    "loop_judge": LoopJudgeOutput,
    "reflector": ReflectorOutput,
    "judge": JudgeOutput,
    "meta_mutator": MutatorOutput,
}


@cache
def challenger_schema_for(payload_model: type[BaseModel] | None) -> type[BaseModel]:
    """Build the strict challenger envelope, tightening `payload` to `payload_model`
    when the domain declares one. Cached so equal payload models share one class."""
    if payload_model is None:
        return ChallengerEnvelope
    return create_model(
        f"ChallengerOutput_{payload_model.__name__}",
        payload=(payload_model, ...),
        reference_output=(str, ...),
        rubric=(list[RubricCriterionOutput], ...),
        __base__=BaseModel,
    )


def _has_open_map(node: Any) -> bool:
    """True if the JSON schema has an open/dynamic-key object anywhere.

    An ``additionalProperties`` that is ``True`` or a subschema (i.e. a dict-typed
    field like challenger ``payload`` without a domain model, or judge
    ``per_criterion``) is outside the OpenAI/Azure strict structured-output subset,
    where every object must pin ``additionalProperties: false``.
    """
    if isinstance(node, dict):
        ap = node.get("additionalProperties", False)
        if ap is True or isinstance(ap, dict):
            return True
        return any(_has_open_map(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_open_map(v) for v in node)
    return False


@cache
def _strict_compatible(model: type[BaseModel]) -> bool:
    """Whether ``model`` can be sent as a provider ``strict`` schema at all.

    Schemas with an open map (see :func:`_has_open_map`) 400 on OpenAI/Azure, so
    they must fall back to plain JSON mode regardless of provider schema support.
    """
    return not _has_open_map(model.model_json_schema())


def response_format_for(litellm: Any, req: LLMRequest) -> Any:
    schema = req.response_schema or _MODELS.get(req.role)
    if schema is not None and _strict_compatible(schema):
        try:
            if litellm.supports_response_schema(model=req.model_key):
                return schema
        except Exception:
            pass
    return {"type": "json_object"}
