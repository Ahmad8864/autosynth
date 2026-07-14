"""Independent final check for an otherwise accepted round."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from autosynth.agents.loop_judge import scores_summary
from autosynth.config import AuditConfig
from autosynth.domain import DomainAdapter, GroundingItem
from autosynth.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness
from autosynth.llm import LLMRequest
from autosynth.schemas import Candidate, QualityCheck, SolverScore
from autosynth.utils import extract_json, stable_id

_SYSTEM = (
    "ROLE:FINAL_AUDITOR. You are the independent final audit for a synthetic training example that "
    "already passed generation-time checks. Do NOT trust those checks; re-verify from scratch. "
    "FAIL the example if ANY of these hold: "
    "(1) LEAKAGE — the context or question lets a reader construct the reference answer without reasoning; "
    "(2) UNSUPPORTED — the reference answer contradicts or is not supported by the source excerpt; "
    "(3) BRITTLE_TRIVIA — the difficulty comes mainly from recalling a source-specific number or name "
    "rather than reasoning that generalizes; "
    "(4) RUBRIC — criteria are redundant, reward verbosity or style over substance, or can be satisfied "
    "without solving the task; "
    "(5) MEANINGLESS — the task does not test a capability worth training. "
    "Return STRICT JSON: {passed: bool, failures: [one string per violated code, each with a short "
    "specific reason], notes: string}."
)


def build_request(
    *,
    item_id: str,
    round_n: int,
    model_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    candidate: Candidate,
    grounding: GroundingItem | None,
    weak_scores: Sequence[SolverScore],
    strong_scores: Sequence[SolverScore],
    domain: DomainAdapter,
    audit_cfg: AuditConfig,
    harness: HarnessSpec | None = None,
) -> LLMRequest:
    h = harness or DEFAULT_HARNESS
    evidence = scores_summary(weak_scores, strong_scores) if audit_cfg.include_evidence else None
    messages = domain.audit_prompt(candidate, grounding, evidence)
    if messages is None:
        messages = _default_messages(candidate, grounding, evidence, domain, audit_cfg)
    messages = apply_harness(messages, h.rules_for("audit"))
    return LLMRequest(
        request_id=stable_id(item_id, round_n, "audit", 0),
        item_id=item_id,
        round_n=round_n,
        role="audit",
        model_key=model_key,
        messages=messages,
        json_mode=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _default_messages(
    candidate: Candidate,
    grounding: GroundingItem | None,
    evidence: dict[str, Any] | None,
    domain: DomainAdapter,
    audit_cfg: AuditConfig,
) -> list[dict[str, str]]:
    source = None
    if grounding is not None and audit_cfg.grounding_chars > 0:
        source = grounding.body[: audit_cfg.grounding_chars]
    usr = json.dumps(
        {
            "payload": candidate.payload,
            "reference_output": candidate.reference_output,
            "rubric": [c.model_dump() for c in candidate.rubric],
            "source_excerpt": source,
            "rollout_scores": evidence,
            "leakage_rules": domain.leakage_rules() or None,
        },
        default=str,
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": usr}]


def parse_audit(text: str) -> QualityCheck:
    """Parse an auditor response. Returns a failing check on parse error."""
    try:
        data = extract_json(text)
    except ValueError as e:
        return QualityCheck(passed=False, failures=[f"audit_parse_error:{e}"])
    return QualityCheck(
        passed=bool(data.get("passed", False)),
        failures=[str(x) for x in (data.get("failures") or [])],
        notes=data.get("notes"),
    )
