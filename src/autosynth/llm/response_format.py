"""Response-format selection: strict schemas for fixed-shape roles on providers
that support structured outputs, plain JSON mode otherwise."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

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


# Challenger payload and judge per_criterion are free-form-keyed (no strict schema).
_MODELS: dict[str, type[BaseModel]] = {
    "quality": QualityOutput,
    "loop_judge": LoopJudgeOutput,
    "reflector": ReflectorOutput,
}


def response_format_for(litellm: Any, req: LLMRequest) -> Any:
    schema = _MODELS.get(req.role)
    if schema is not None:
        try:
            if litellm.supports_response_schema(model=req.model_key):
                return schema
        except Exception:
            pass
    return {"type": "json_object"}
