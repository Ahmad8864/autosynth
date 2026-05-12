"""Generic 'QA from documents' domain.

Reads .txt/.md/.json files from a directory; instructs the challenger to
produce a question that requires reading the *specific* document (not generic
knowledge), plus a reference answer and a rubric.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from autodata.domain import DomainAdapter, GroundingItem, bullet_list, register_domain
from autodata.schemas import Candidate
from autodata.utils import stable_id


@register_domain("qa_from_documents")
class QAFromDocuments(DomainAdapter):
    description = "Generate document-grounded question/answer/rubric triples."

    def __init__(self, source_dir: str, glob: str = "**/*", max_chars: int = 12000, **kw: Any):
        super().__init__(source_dir=source_dir, glob=glob, max_chars=max_chars, **kw)
        self.source_dir = Path(source_dir)
        self.glob = glob
        self.max_chars = max_chars

    # 1. grounding ------------------------------------------------------------
    def load_grounding(self) -> Iterable[GroundingItem]:
        for path in sorted(self.source_dir.glob(self.glob)):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")[: self.max_chars]
            yield GroundingItem(
                source_id=stable_id(path.resolve()),
                body=text,
                metadata={"path": str(path), "name": path.name},
            )

    # 2. challenger prompt ----------------------------------------------------
    def generation_prompt(self, item, feedback, round_n, prior_payloads):
        feedback_block = bullet_list(feedback)
        prior_block = bullet_list(prior_payloads, key="question")
        sys = (
            "ROLE:CHALLENGER. You are constructing a high-quality QA datapoint grounded in a SOURCE DOCUMENT. "
            "The question MUST be answerable only by someone who read THIS specific document — not from generic knowledge. "
            "Do NOT include passages that, if read aloud, would directly leak the answer. "
            "Return STRICT JSON with keys: payload {question, context, reasoning_skills[]}, reference_output (string), "
            "rubric (list of {id, description, weight: integer 1..7}). Rubric criteria must be POSITIVE statements; do not "
            "include penalties. Cover correctness, source-specificity, and reasoning depth."
        )
        usr = (
            f"ROUND={round_n}\n"
            f"SOURCE_ID: {item.source_id}\n"
            f"SOURCE_NAME: {item.metadata.get('name', 'doc')}\n"
            f"SOURCE_DOCUMENT (truncated):\n---\n{item.body}\n---\n\n"
            f"Feedback from previous rounds:\n{feedback_block}\n\n"
            f"Previously attempted questions (avoid repetition / different angle):\n{prior_block}\n\n"
            "Emit ONE candidate as a JSON object."
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    # 3. validation -----------------------------------------------------------
    def validate_candidate(self, candidate: Candidate) -> list[str]:
        # Rubric weight bounds are already enforced by the challenger
        # (clamped to [1, rubric_max_weight]) and the RubricCriterion schema
        # (ge=1), so we don't re-check them here.
        errs: list[str] = []
        p = candidate.payload
        if not isinstance(p.get("question"), str) or len(p["question"].strip()) < 5:
            errs.append("payload.question missing or too short")
        if not candidate.reference_output or len(candidate.reference_output.strip()) < 5:
            errs.append("reference_output missing or too short")
        if not candidate.rubric:
            errs.append("rubric is empty")
        return errs

    # 4. solver prompt --------------------------------------------------------
    def solver_prompt(self, candidate: Candidate, solver_role: str):
        sys = (
            f"ROLE:{'WEAK' if solver_role == 'weak' else 'STRONG'}_SOLVER. "
            "Answer the question grounded in the provided context. Be concrete; cite specific details from the context."
        )
        usr = (
            f"QUESTION: {candidate.payload['question']}\n\n"
            f"CONTEXT:\n{candidate.payload.get('context', '')}\n\n"
            "Answer:"
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    # 5. quality verifier prompt ---------------------------------------------
    def quality_prompt(self, candidate: Candidate):
        sys = (
            "ROLE:QUALITY. Audit a candidate QA datapoint. Check: (a) the question requires reading the specific source; "
            "(b) the context does not directly leak the answer (no rephrased passages that constitute the answer); "
            "(c) the rubric is positive-only with integer weights 1..7 and covers correctness + source-specificity; "
            "(d) the reference_output is concrete. Return JSON: {passed: bool, failures: [strings], notes: string}."
        )
        usr = json.dumps(
            {
                "question": candidate.payload.get("question"),
                "context": candidate.payload.get("context"),
                "reference_output": candidate.reference_output,
                "rubric": [c.model_dump() for c in candidate.rubric],
            },
            indent=2,
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    # 6. judge prompt ---------------------------------------------------------
    def judge_prompt(self, candidate: Candidate, solver_response: str, solver_role: str):
        sys = (
            "ROLE:JUDGE. Score the solver's response against the rubric. For each criterion, output a per_criterion "
            "score in [0,1]. Compute total as the weighted average normalized to [0,1]. Identify failure modes if any. "
            "Return STRICT JSON: {per_criterion: {id: float}, total: float, failure_modes: [strings]}."
        )
        usr = (
            f"[solver={solver_role}]\n"
            f"QUESTION: {candidate.payload.get('question')}\n"
            f"CONTEXT: {candidate.payload.get('context', '')}\n"
            f"REFERENCE_OUTPUT: {candidate.reference_output}\n"
            f"RUBRIC: {json.dumps([c.model_dump() for c in candidate.rubric])}\n"
            f"SOLVER_RESPONSE: {solver_response}\n"
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def leakage_rules(self) -> list[str]:
        return [
            "context must not contain a rephrasing of the reference_output",
            "question must require source-specific knowledge",
        ]
