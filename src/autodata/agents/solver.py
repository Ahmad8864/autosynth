"""Weak / strong solver wrapper.

There is one class; the role label is `weak` or `strong`. The actual
difficulty difference comes from the configured model + temperature, not from
adversarial prompting (the paper warns against telling a weak model to be
weak — that triggers gaming).

This module also exposes the new module-level ``build_request`` helper used
by the event-sourced pipeline.
"""

from __future__ import annotations

from autodata.domain import DomainAdapter
from autodata.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness
from autodata.llm import LLMRequest
from autodata.models import LLMClient
from autodata.schemas import Candidate
from autodata.utils import stable_id


def build_request(
    *,
    item_id: str,
    round_n: int,
    attempt: int,
    model_key: str,
    candidate: Candidate,
    role: str,
    domain: DomainAdapter,
    harness: HarnessSpec | None = None,
) -> LLMRequest:
    """Build one solver attempt's LLMRequest."""
    if role not in {"weak", "strong"}:
        raise ValueError(f"role must be 'weak' or 'strong', got {role!r}")
    h = harness or DEFAULT_HARNESS
    messages = domain.solver_prompt(candidate, role)
    messages = apply_harness(messages, h.rules_for(f"{role}_solver"))
    return LLMRequest(
        request_id=stable_id(item_id, round_n, role, attempt),
        item_id=item_id,
        round_n=round_n,
        role=role,
        model_key=model_key,
        messages=messages,
        attempt=attempt,
        json_mode=False,
    )


class SolverAgent:
    def __init__(
        self,
        client: LLMClient,
        domain: DomainAdapter,
        role: str,
        harness: HarnessSpec | None = None,
    ):
        if role not in {"weak", "strong"}:
            raise ValueError(f"role must be 'weak' or 'strong', got {role!r}")
        self.client = client
        self.domain = domain
        self.role = role
        self.harness = harness or DEFAULT_HARNESS

    def attempt(self, candidate: Candidate) -> str:
        messages = self.domain.solver_prompt(candidate, self.role)
        rules_key = f"{self.role}_solver"
        messages = apply_harness(messages, self.harness.rules_for(rules_key))
        resp = self.client.complete(messages)
        return resp.text
