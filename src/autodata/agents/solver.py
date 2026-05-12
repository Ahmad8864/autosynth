"""Weak / strong solver wrapper.

There is one class; the role label is `weak` or `strong`. The actual
difficulty difference comes from the configured model + temperature, not from
adversarial prompting (the paper warns against telling a weak model to be
weak — that triggers gaming).
"""
from __future__ import annotations

from autodata.domain import DomainAdapter
from autodata.models import LLMClient
from autodata.schemas import Candidate


class SolverAgent:
    def __init__(self, client: LLMClient, domain: DomainAdapter, role: str):
        if role not in {"weak", "strong"}:
            raise ValueError(f"role must be 'weak' or 'strong', got {role!r}")
        self.client = client
        self.domain = domain
        self.role = role

    def attempt(self, candidate: Candidate) -> str:
        messages = self.domain.solver_prompt(candidate, self.role)
        resp = self.client.complete(messages)
        return resp.text
