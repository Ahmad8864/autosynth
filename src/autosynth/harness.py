"""Editable agent rules used by the meta-optimization loop."""

from __future__ import annotations

from pydantic import BaseModel, Field

from autosynth.utils import stable_id


def _id(*parts) -> str:
    return stable_id(*parts, length=10)


class HarnessSpec(BaseModel):
    """One candidate set of role-specific instructions."""

    harness_id: str
    parent_id: str | None = None
    iteration: int = 0
    rationale: str = ""

    challenger_rules: list[str] = Field(default_factory=list)
    quality_rules: list[str] = Field(default_factory=list)
    judge_rules: list[str] = Field(default_factory=list)
    solver_rules: list[str] = Field(default_factory=list)
    reflector_rules: list[str] = Field(default_factory=list)
    audit_rules: list[str] = Field(default_factory=list)

    rubric_max_weight: int = 7
    require_self_test: bool = False

    train_score: float | None = None
    val_score: float | None = None

    def rules_for(self, role: str) -> list[str]:
        rules = list(getattr(self, f"{role}_rules", []) or [])
        if role == "challenger" and self.require_self_test:
            rules.append(
                "Include a `self_test` string in payload: a short statement of why "
                "this question cannot be answered without reading the source. The "
                "quality verifier will check it."
            )
        if role == "quality" and self.require_self_test:
            rules.append("Verify that payload.self_test names a source-specific dependency.")
        return rules

    def fingerprint(self) -> str:
        """Stable hash of the harness content (ignoring iteration / scores)."""
        content = self.model_dump(
            exclude={"harness_id", "parent_id", "iteration", "train_score", "val_score", "rationale"}
        )
        return _id("harness", repr(sorted(content.items())))


def make_harness(
    *, parent_id: str | None = None, iteration: int = 0, rationale: str = "", **fields
) -> HarnessSpec:
    """Construct a HarnessSpec with an auto-generated id."""
    hid = _id("h", iteration, parent_id or "", repr(sorted(fields.items())))
    return HarnessSpec(
        harness_id=hid,
        parent_id=parent_id,
        iteration=iteration,
        rationale=rationale,
        **fields,
    )


DEFAULT_HARNESS: HarnessSpec = make_harness(
    iteration=0,
    rationale="initial seed rules",
    challenger_rules=[
        "The candidate must require reading the SPECIFIC source — it should not be answerable from generic knowledge.",
        "Do NOT embed in the context any phrasing that would directly leak the reference answer.",
        "Cover at least one rubric criterion that targets source-specific detail.",
    ],
    quality_rules=[
        "Reject if the context directly leaks the reference_output.",
        "Reject if the question is generic enough to answer without the source.",
        "Reject rubric criteria whose weights are not integers in [1, 7].",
        "Reject negative or penalty-style rubric criteria; rubric must be positive-only.",
    ],
    judge_rules=[
        "Output STRICT JSON only — no prose before or after the object.",
        "per_criterion keys must match the rubric ids exactly.",
        "If the response is generic or boilerplate, lower scores accordingly even if it sounds plausible.",
    ],
    solver_rules=[],
    reflector_rules=[
        "Be specific. Cite the failure mode from the previous round (e.g., 'context_leak', 'too_easy').",
    ],
    rubric_max_weight=7,
    require_self_test=False,
)


# Prompt injection


def apply_harness(messages: list[dict[str, str]], rules: list[str]) -> list[dict[str, str]]:
    """Add rules to the system message, creating one when needed."""
    if not rules:
        return messages
    block = "\n\nADDITIONAL RULES (from harness):\n" + "\n".join(f"- {r}" for r in rules)
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = (msgs[0].get("content") or "") + block
    else:
        msgs.insert(0, {"role": "system", "content": block.lstrip()})
    return msgs
