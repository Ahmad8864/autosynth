"""Challenger: turns grounding + feedback into a structured Candidate."""
from __future__ import annotations

from typing import Any

from loguru import logger

from autodata.domain import DomainAdapter, GroundingItem
from autodata.models import LLMClient
from autodata.schemas import Candidate, RubricCriterion
from autodata.utils import stable_id


class ChallengerAgent:
    def __init__(self, client: LLMClient, domain: DomainAdapter, rubric_max_weight: int = 7):
        self.client = client
        self.domain = domain
        self.rubric_max_weight = rubric_max_weight

    def generate(
        self,
        item: GroundingItem,
        round_n: int,
        feedback: list[str],
        prior_payloads: list[dict[str, Any]],
    ) -> Candidate:
        messages = self.domain.generation_prompt(item, feedback, round_n, prior_payloads)
        data = self.client.complete_json(messages)
        return self._parse(data, item, round_n)

    def _parse(self, data: dict[str, Any], item: GroundingItem, round_n: int) -> Candidate:
        payload = data.get("payload") or {}
        reference_output = data.get("reference_output")
        rubric_raw = data.get("rubric") or []

        rubric: list[RubricCriterion] = []
        for i, r in enumerate(rubric_raw):
            try:
                weight = int(r.get("weight", 1))
            except (TypeError, ValueError):
                weight = 1
            weight = max(1, min(self.rubric_max_weight, weight))
            rubric.append(
                RubricCriterion(
                    id=str(r.get("id") or f"c{i + 1}"),
                    description=str(r.get("description", "")).strip() or f"criterion {i + 1}",
                    weight=weight,
                )
            )

        cid = stable_id(item.source_id, round_n, str(payload)[:200])
        cand = Candidate(
            candidate_id=cid,
            domain=self.domain.name,
            source_id=item.source_id,
            payload=payload,
            rubric=rubric,
            reference_output=reference_output,
            metadata={"round": round_n, "source": item.metadata},
        )
        logger.debug("challenger emitted candidate {} (round {})", cid, round_n)
        return cand
