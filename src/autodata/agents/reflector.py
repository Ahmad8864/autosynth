"""Reflector / RecipeUpdater.

Reads the prior rounds for a source item and emits a small list of
*targeted* feedback bullets for the challenger's next attempt, plus a hint
toward a different angle. This is the 'updated recipe' from the paper.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from autodata.models import LLMClient
from autodata.schemas import Round


class Reflector:
    def __init__(self, client: LLMClient):
        self.client = client

    def reflect(self, rounds: list[Round], domain_name: str, leakage_rules: list[str]) -> dict[str, Any]:
        too_easy = []
        failed_strong = []
        failed_quality = []
        for r in rounds:
            ev = r.evaluation
            q = r.quality
            cand = r.candidate
            short = {
                "round": r.refinement_round,
                "payload_summary": _summarize_payload(cand.payload),
            }
            if not q.passed:
                failed_quality.append({**short, "failures": q.failures})
                continue
            if not ev:
                continue
            if ev.weak_avg > 0.65:
                too_easy.append({**short, "weak_avg": ev.weak_avg})
            if ev.strong_avg < 0.6:
                failed_strong.append({**short, "strong_avg": ev.strong_avg, "gap": ev.gap})

        messages = [
            {"role": "system", "content": (
                "ROLE:REFLECTION. Analyze why prior candidates failed and produce TARGETED feedback for the next "
                "challenger round. Be specific. Encourage a DIFFERENT reasoning angle than what has been tried. "
                "Return JSON: {feedback: [strings], new_angle: string}."
            )},
            {"role": "user", "content": (
                f"DOMAIN: {domain_name}\n"
                f"LEAKAGE_RULES: {json.dumps(leakage_rules)}\n"
                f"TOO_EASY (weak scored too high): {json.dumps(too_easy)}\n"
                f"FAILED_STRONG (strong scored too low): {json.dumps(failed_strong)}\n"
                f"FAILED_QUALITY: {json.dumps(failed_quality)}\n"
            )},
        ]
        try:
            data = self.client.complete_json(messages)
        except Exception as e:
            logger.warning("reflector parse failure: {}", e)
            return {"feedback": ["previous attempts failed; try a different angle"], "new_angle": ""}
        return {
            "feedback": [str(x) for x in (data.get("feedback") or [])],
            "new_angle": str(data.get("new_angle") or ""),
        }


def _summarize_payload(payload: dict[str, Any], limit: int = 140) -> str:
    # Pick the first stringy field for a compact reference.
    for v in payload.values():
        if isinstance(v, str):
            return v[:limit]
    return json.dumps(payload)[:limit]
