"""Math word problems — a non-document domain.

Grounds the challenger on lightweight 'topic seeds' rather than a corpus,
demonstrating that the framework is not document-bound.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from fractions import Fraction
from typing import Any

from autosynth.domain import DomainAdapter, GroundingItem, bullet_list, register_domain
from autosynth.schemas import Candidate
from autosynth.utils import stable_id

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+/\d+|-?\d*\.\d+|-?\d+\.?\d*")


def _parse_number(text: str) -> Fraction | None:
    """First number in ``text`` as an exact Fraction, or None. '0.5' == '1/2'."""
    m = _NUMBER_RE.search(text.replace(",", ""))
    if m is None:
        return None
    try:
        return Fraction(m.group().rstrip("."))
    except (ValueError, ZeroDivisionError):
        return None


_DEFAULT_TOPICS = [
    {"topic": "linear systems", "difficulty": "high-school"},
    {"topic": "probability with conditional events", "difficulty": "high-school"},
    {"topic": "geometric optimization", "difficulty": "olympiad-lite"},
    {"topic": "modular arithmetic", "difficulty": "competition"},
    {"topic": "rates and mixtures", "difficulty": "high-school"},
]


@register_domain("math_word_problems")
class MathWordProblems(DomainAdapter):
    description = "Generate math word problems with verifiable numeric answers."
    default_acceptance_mode = "verifiable"

    def __init__(self, topics: list[dict[str, Any]] | None = None, **kw: Any):
        super().__init__(topics=topics, **kw)
        self.topics = topics or _DEFAULT_TOPICS

    def load_grounding(self) -> Iterable[GroundingItem]:
        for t in self.topics:
            sid = stable_id("math", t["topic"], t.get("difficulty", "?"))
            yield GroundingItem(
                source_id=sid,
                body=json.dumps(t),
                metadata=t,
            )

    def generation_prompt(self, item, feedback, round_n, prior_payloads):
        feedback_block = bullet_list(feedback)
        prior_block = bullet_list(prior_payloads, key="problem", limit=140)
        sys = (
            "ROLE:CHALLENGER. Construct ONE math word problem with a single, verifiable numeric answer. "
            "Difficulty should yield a clear weak/strong gap: a careful reasoner solves it, a careless one does not. "
            "Return STRICT JSON: payload {problem: string, topic: string, difficulty: string}, "
            "reference_output (canonical numeric answer as a string, e.g. '42' or '3/7'), "
            "rubric (list of {id, description, weight 1..7}). Cover: correctness of final answer (highest weight), "
            "explicit solution steps, and unit/format correctness."
        )
        usr = (
            f"ROUND={round_n}\n"
            f"TOPIC: {item.metadata.get('topic')}\n"
            f"DIFFICULTY: {item.metadata.get('difficulty')}\n\n"
            f"Feedback:\n{feedback_block}\n\n"
            f"Previously attempted problems (different angle):\n{prior_block}\n\n"
            "Emit ONE candidate as a JSON object."
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def validate_candidate(self, candidate: Candidate) -> list[str]:
        # Rubric weight bounds are enforced upstream (challenger clamps,
        # RubricCriterion validates ge=1); no need to re-check here.
        errs: list[str] = []
        p = candidate.payload
        if not isinstance(p.get("problem"), str) or len(p["problem"].strip()) < 10:
            errs.append("payload.problem missing or too short")
        if not candidate.reference_output:
            errs.append("reference_output (numeric answer) missing")
        if not candidate.rubric:
            errs.append("rubric is empty")
        return errs

    def solver_prompt(self, candidate: Candidate, solver_role: str):
        sys = (
            f"ROLE:{'WEAK' if solver_role == 'weak' else 'STRONG'}_SOLVER. "
            "Solve the math word problem. Show your steps briefly, then provide the final numeric answer "
            "on a line starting with 'ANSWER:'."
        )
        usr = f"PROBLEM: {candidate.payload['problem']}\n\nSolve it."
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def quality_prompt(self, candidate: Candidate):
        sys = (
            "ROLE:QUALITY. Audit a math word problem candidate. Check: (a) problem is well-posed and unambiguous; "
            "(b) reference_output is a single canonical numeric answer; (c) rubric is positive-only with integer "
            "weights 1..7 and correctness criterion carries the highest weight. "
            "Return JSON: {passed: bool, failures: [strings], notes: string}."
        )
        usr = json.dumps(
            {
                "problem": candidate.payload.get("problem"),
                "reference_output": candidate.reference_output,
                "rubric": [c.model_dump() for c in candidate.rubric],
            },
            indent=2,
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def judge_prompt(self, candidate: Candidate, solver_response: str, solver_role: str):
        sys = (
            "ROLE:JUDGE. Score the solver against the rubric. The correctness criterion should reflect numeric "
            "equality with reference_output (allow trivial reformatting like '1/2' vs '0.5'). "
            "Return STRICT JSON: {per_criterion: {id: float in [0,1]}, total: float in [0,1], failure_modes: [strings]}."
        )
        usr = (
            f"[solver={solver_role}]\n"
            f"PROBLEM: {candidate.payload.get('problem')}\n"
            f"REFERENCE_ANSWER: {candidate.reference_output}\n"
            f"RUBRIC: {json.dumps([c.model_dump() for c in candidate.rubric])}\n"
            f"SOLVER_RESPONSE: {solver_response}\n"
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def verify(self, candidate: Candidate, solver_response: str) -> bool | None:
        ref = _parse_number(candidate.reference_output or "")
        m = _ANSWER_RE.findall(solver_response)
        if not m:
            return None
        got = _parse_number(m[-1])
        if ref is None or got is None:
            return None
        return got == ref
