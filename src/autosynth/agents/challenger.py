"""Challenger: turns grounding + feedback into a structured :class:`Candidate`."""

from __future__ import annotations

from typing import Any

from autosynth.domain import DomainAdapter, GroundingItem
from autosynth.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness
from autosynth.llm import LLMRequest
from autosynth.llm.response_format import challenger_schema_for
from autosynth.schemas import Candidate, RubricCriterion
from autosynth.utils import extract_json, stable_id


def build_request(
    *,
    item_id: str,
    round_n: int,
    model_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    grounding: GroundingItem,
    feedback: list[str],
    prior_payloads: list[dict[str, Any]],
    domain: DomainAdapter,
    harness: HarnessSpec | None = None,
) -> LLMRequest:
    """Build the challenger LLMRequest for a single round."""
    h = harness or DEFAULT_HARNESS
    messages = domain.generation_prompt(grounding, feedback, round_n, prior_payloads)
    messages = apply_harness(messages, h.rules_for("challenger"))
    return LLMRequest(
        request_id=stable_id(item_id, round_n, "challenger", 0),
        item_id=item_id,
        round_n=round_n,
        role="challenger",
        model_key=model_key,
        messages=messages,
        json_mode=True,
        response_schema=challenger_schema_for(domain.payload_model()),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def parse_response(
    text: str,
    *,
    source_id: str,
    round_n: int,
    domain_name: str,
    source_metadata: dict[str, Any] | None = None,
    rubric_max_weight: int = 7,
) -> Candidate:
    """Parse a challenger response, raising ``ValueError`` for invalid JSON."""
    data = extract_json(text)
    payload = data.get("payload") or {}
    reference_output = data.get("reference_output")
    rubric_raw = data.get("rubric") or []
    if not isinstance(rubric_raw, list):
        rubric_raw = []

    rubric: list[RubricCriterion] = []
    for i, r in enumerate(rubric_raw):
        if not isinstance(r, dict):
            r = {"description": str(r)}  # tolerate a bare-string/scalar criterion
        try:
            weight = int(r.get("weight", 1))
        except (TypeError, ValueError):
            weight = 1
        weight = max(1, min(rubric_max_weight, weight))
        rubric.append(
            RubricCriterion(
                id=str(r.get("id") or f"c{i + 1}"),
                description=str(r.get("description", "")).strip() or f"criterion {i + 1}",
                weight=weight,
            )
        )

    cid = stable_id(source_id, round_n, str(payload)[:200])
    return Candidate(
        candidate_id=cid,
        domain=domain_name,
        source_id=source_id,
        payload=payload,
        rubric=rubric,
        reference_output=reference_output,
        metadata={"round": round_n, "source": source_metadata or {}},
    )
