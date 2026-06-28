"""Reflector / RecipeUpdater.

Reads the prior rounds for an item and emits a small list of *targeted*
feedback bullets for the next challenger attempt, plus a hint toward a
different angle. This is the "updated recipe" from the paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from autosynth.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness
from autosynth.llm import LLMRequest
from autosynth.schemas import Round
from autosynth.utils import extract_json, stable_id


@dataclass(frozen=True)
class ReflectionResult:
    feedback: list[str]
    new_angle: str


def _summarize_prior_rounds(
    rounds: list[Round], weak_ceiling: float, strong_floor: float
) -> dict[str, list[dict]]:
    too_easy: list[dict] = []
    failed_strong: list[dict] = []
    failed_quality: list[dict] = []
    for r in rounds:
        short = {
            "round": r.refinement_round,
            "payload_summary": _summarize_payload(r.candidate.payload),
        }
        if not r.quality.passed:
            failed_quality.append({**short, "failures": r.quality.failures})
            continue
        ev = r.evaluation
        if ev is None:
            continue
        if ev.weak_avg > weak_ceiling:
            too_easy.append({**short, "weak_avg": ev.weak_avg})
        if ev.strong_avg < strong_floor:
            failed_strong.append({**short, "strong_avg": ev.strong_avg, "gap": ev.gap})
    return {"too_easy": too_easy, "failed_strong": failed_strong, "failed_quality": failed_quality}


def build_request(
    *,
    item_id: str,
    round_n: int,
    model_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    prior_rounds: list[Round],
    domain_name: str,
    leakage_rules: list[str],
    weak_ceiling: float,
    strong_floor: float,
    harness: HarnessSpec | None = None,
) -> LLMRequest:
    """Build the reflector request for an item entering round_n (round_n >= 2).

    ``weak_ceiling`` / ``strong_floor`` come from the active acceptance policy
    and bucket prior rounds into too-easy / failed-strong feedback.
    """
    h = harness or DEFAULT_HARNESS
    summary = _summarize_prior_rounds(prior_rounds, weak_ceiling, strong_floor)
    messages = [
        {
            "role": "system",
            "content": (
                "ROLE:REFLECTION. Analyze why prior candidates failed and produce TARGETED feedback for the next "
                "challenger round. Be specific. Encourage a DIFFERENT reasoning angle than what has been tried. "
                "Return JSON: {feedback: [strings], new_angle: string}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"DOMAIN: {domain_name}\n"
                f"LEAKAGE_RULES: {json.dumps(leakage_rules)}\n"
                f"TOO_EASY (weak scored too high): {json.dumps(summary['too_easy'])}\n"
                f"FAILED_STRONG (strong scored too low): {json.dumps(summary['failed_strong'])}\n"
                f"FAILED_QUALITY: {json.dumps(summary['failed_quality'])}\n"
            ),
        },
    ]
    messages = apply_harness(messages, h.rules_for("reflector"))
    return LLMRequest(
        request_id=stable_id(item_id, round_n, "reflector", 0),
        item_id=item_id,
        round_n=round_n,
        role="reflector",
        model_key=model_key,
        messages=messages,
        json_mode=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def parse_response(text: str) -> ReflectionResult:
    """Parse a reflector response. Returns a generic fallback on parse error."""
    try:
        data = extract_json(text)
    except ValueError as e:
        logger.warning("reflector parse failure: {}", e)
        return ReflectionResult(feedback=["previous attempts failed; try a different angle"], new_angle="")
    return ReflectionResult(
        feedback=[str(x) for x in (data.get("feedback") or [])],
        new_angle=str(data.get("new_angle") or ""),
    )


def _summarize_payload(payload: dict[str, Any], limit: int = 140) -> str:
    """Pick the first stringy field for a compact reference, else dump."""
    for v in payload.values():
        if isinstance(v, str):
            return v[:limit]
    return json.dumps(payload)[:limit]
