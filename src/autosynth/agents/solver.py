"""Build weak and strong solver requests."""

from __future__ import annotations

from autosynth.domain import DomainAdapter
from autosynth.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness
from autosynth.llm import LLMRequest
from autosynth.schemas import Candidate
from autosynth.utils import stable_id


def solver_messages(
    candidate: Candidate, domain: DomainAdapter, harness: HarnessSpec | None = None
) -> list[dict[str, str]]:
    """Build the messages used for solver requests and training exports."""
    h = harness or DEFAULT_HARNESS
    return apply_harness(domain.solver_prompt(candidate), h.rules_for("solver"))


def build_request(
    *,
    item_id: str,
    round_n: int,
    attempt: int,
    model_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    candidate: Candidate,
    role: str,
    domain: DomainAdapter,
    harness: HarnessSpec | None = None,
) -> LLMRequest:
    """Build one solver attempt's LLMRequest."""
    if role not in {"weak", "strong"}:
        raise ValueError(f"role must be 'weak' or 'strong', got {role!r}")
    messages = solver_messages(candidate, domain, harness)
    return LLMRequest(
        request_id=stable_id(item_id, round_n, role, attempt),
        item_id=item_id,
        round_n=round_n,
        role=role,
        model_key=model_key,
        messages=messages,
        attempt=attempt,
        json_mode=False,
        temperature=temperature,
        max_tokens=max_tokens,
    )
