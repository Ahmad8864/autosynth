"""Loop-judge: decides accept/improve for a scored round (paper §3.2).

Unlike the fixed-threshold policies, the loop-judge reads the per-rollout
weak/strong patterns and decides whether the round is good GRPO training data,
emitting a concrete suggestion for the next challenger round when it isn't.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, pstdev

from loguru import logger

from autosynth.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness
from autosynth.llm import LLMRequest
from autosynth.schemas import Candidate, QualityCheck, SolverScore
from autosynth.utils import extract_json, stable_id


@dataclass(frozen=True)
class LoopJudgeVerdict:
    accept: bool
    grpo_suitability: str  # "high" | "medium" | "low"
    reason: str
    suggestion: str


def _scores_summary(weak: Sequence[SolverScore], strong: Sequence[SolverScore]) -> dict[str, object]:
    weak_vals = [s.total for s in weak] or [0.0]
    strong_vals = [s.total for s in strong] or [0.0]
    weak_avg, strong_avg = mean(weak_vals), mean(strong_vals)
    return {
        "weak": {
            "per_rollout": [round(v, 3) for v in weak_vals],
            "avg": round(weak_avg, 3),
            "std": round(pstdev(weak_vals), 3),
        },
        "strong": {
            "per_rollout": [round(v, 3) for v in strong_vals],
            "avg": round(strong_avg, 3),
            "std": round(pstdev(strong_vals), 3),
        },
        "gap": round(strong_avg - weak_avg, 3),
    }


def build_request(
    *,
    item_id: str,
    round_n: int,
    model_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    candidate: Candidate,
    weak_scores: Sequence[SolverScore],
    strong_scores: Sequence[SolverScore],
    quality: QualityCheck | None,
    harness: HarnessSpec | None = None,
) -> LLMRequest:
    h = harness or DEFAULT_HARNESS
    sys = (
        "ROLE:LOOP_JUDGE. Decide whether a scored candidate is good RL (GRPO) training data. "
        "Good data SEPARATES a weak from a strong solver: the weak solver mostly struggles, the strong "
        "solver mostly succeeds, and the per-rollout reward has usable spread (not all-equal — near-zero "
        "variance gives no gradient). If it is good, ACCEPT. Otherwise choose IMPROVE and give ONE concrete "
        "suggestion that makes the next question MORE DISCRIMINATING (harder for the weak solver, not easier "
        "for the strong one) — not a rephrase. "
        "Return STRICT JSON: {verdict: 'accept'|'improve', grpo_suitability: 'high'|'medium'|'low', "
        "reason: string, suggestion: string}."
    )
    usr = json.dumps(
        {
            "payload": candidate.payload,
            "reference_output": candidate.reference_output,
            "scores": _scores_summary(weak_scores, strong_scores),
            "quality_notes": quality.notes if quality else None,
        },
        default=str,
    )
    messages = apply_harness(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        h.rules_for("loop_judge"),
    )
    return LLMRequest(
        request_id=stable_id(item_id, round_n, "loop_judge", 0),
        item_id=item_id,
        round_n=round_n,
        role="loop_judge",
        model_key=model_key,
        messages=messages,
        json_mode=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def parse_verdict(text: str) -> LoopJudgeVerdict:
    """Parse a loop-judge response. Defaults to 'improve' on a malformed response."""
    try:
        data = extract_json(text)
    except ValueError as e:
        logger.warning("loop_judge parse failure: {}", e)
        return LoopJudgeVerdict(
            accept=False, grpo_suitability="low", reason=f"parse_error:{e}", suggestion=""
        )
    return LoopJudgeVerdict(
        accept=str(data.get("verdict", "improve")).strip().lower() == "accept",
        grpo_suitability=str(data.get("grpo_suitability", "low")),
        reason=str(data.get("reason", "")),
        suggestion=str(data.get("suggestion", "")),
    )
