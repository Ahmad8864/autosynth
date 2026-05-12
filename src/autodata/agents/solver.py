"""Weak / strong solver wrapper.

There is one class; the role label is `weak` or `strong`. The actual
difficulty difference comes from the configured model + temperature, not from
adversarial prompting (the paper warns against telling a weak model to be
weak — that triggers gaming).
"""
from __future__ import annotations

from typing import Optional

from autodata.domain import DomainAdapter
from autodata.harness import DEFAULT_HARNESS, HarnessSpec, apply_harness
from autodata.models import LLMClient
from autodata.schemas import Candidate


class SolverAgent:
    def __init__(
        self,
        client: LLMClient,
        domain: DomainAdapter,
        role: str,
        harness: Optional[HarnessSpec] = None,
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
