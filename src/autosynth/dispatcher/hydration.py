"""DTO mapping between the store (sqlite rows) and the pipeline (frozen dataclasses).

Pure functions. No I/O beyond store reads. Keeping these out of the
:class:`Dispatcher` class so the run loop stays focused on orchestration.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from autosynth.llm import LLMRequest
from autosynth.llm.response_format import challenger_schema_for
from autosynth.pipeline import ItemState, State, StepResponse
from autosynth.schemas import Candidate, EvalReport, QualityCheck, Round
from autosynth.store import RequestRow, Store

if TYPE_CHECKING:
    from autosynth.domain import DomainAdapter

_PLACEHOLDER_CANDIDATE = Candidate(
    candidate_id="missing",
    domain="missing",
    source_id="missing",
    payload={},
    rubric=[],
)


def load_item_state(store: Store, item_row: sqlite3.Row) -> ItemState:
    """Reconstruct an ItemState from the store before each pipeline.step()."""
    item_id = item_row["item_id"]
    rounds_rows = store.rounds_for_item(item_id)
    current_round = int(item_row["current_round"])

    rounds_history: list[Round] = []
    current_round_blob: dict | None = None
    for r in rounds_rows:
        obj = row_to_round(r)
        if r["round_n"] < current_round:
            rounds_history.append(obj)
        elif r["round_n"] == current_round:
            current_round_blob = dict(r)

    candidate = None
    quality = None
    if current_round_blob is not None:
        if current_round_blob["candidate_blob"]:
            candidate = Candidate.model_validate_json(current_round_blob["candidate_blob"])
        if current_round_blob["quality_blob"]:
            quality = QualityCheck.model_validate_json(current_round_blob["quality_blob"])

    scores = store.scores_for_round(item_id, current_round)
    weak_scores = tuple(s for s in scores if s.solver == "weak")
    strong_scores = tuple(s for s in scores if s.solver == "strong")

    # last_feedback from previous round's reflection column.
    last_feedback: tuple[str, ...] = ()
    if current_round > 1:
        prev = store.get_round(item_id, current_round - 1)
        if prev and prev["reflection"]:
            try:
                last_feedback = tuple(json.loads(prev["reflection"]))
            except (json.JSONDecodeError, TypeError):
                last_feedback = ()

    source_metadata = json.loads(item_row["source_metadata"]) if item_row["source_metadata"] else {}
    rejection_reasons = tuple(
        json.loads(item_row["rejection_reasons"]) if item_row["rejection_reasons"] else []
    )

    return ItemState(
        item_id=item_id,
        run_id=item_row["run_id"],
        source_id=item_row["source_id"],
        domain=item_row["domain"],
        state=State(item_row["state"]),
        current_round=current_round,
        rounds_history=tuple(rounds_history),
        candidate=candidate,
        quality=quality,
        weak_scores=weak_scores,
        strong_scores=strong_scores,
        last_feedback=last_feedback,
        source_metadata=source_metadata,
        rejection_reasons=rejection_reasons,
    )


def hydrate_responses(store: Store, item_id: str, since_seq: int) -> tuple[list[StepResponse], int | None]:
    """Pull responses (rowid > since_seq) + matching request/parent fields in one query.

    Returns ``(responses, max_seq)``. The dispatcher writes ``max_seq`` to the
    item's ``consumed_seq`` watermark — a strictly-monotonic integer that bounds
    the just-consumed responses without masking worker rows that committed
    concurrently. ``None`` when no rows matched.
    """
    out: list[StepResponse] = []
    max_seq: int | None = None
    for row in store.hydrate_responses(item_id, since_seq):
        is_judge = row["role"] == "judge" and row["parent_response_id"] is not None
        out.append(
            StepResponse(
                request_id=row["request_id"],
                role=row["role"],
                round_n=row["round_n"],
                attempt=row["attempt"],
                text=row["text"],
                parent_response_id=row["parent_response_id"],
                solver_response_text=(row["parent_text"] or "") if is_judge else None,
                solver_role=row["parent_role"] if is_judge else None,
            )
        )
        seq = row["seq"]
        if max_seq is None or seq > max_seq:
            max_seq = seq
    return out, max_seq


def row_to_round(row) -> Round:
    candidate = Candidate.model_validate_json(row["candidate_blob"]) if row["candidate_blob"] else None
    quality = QualityCheck.model_validate_json(row["quality_blob"]) if row["quality_blob"] else None
    evaluation = EvalReport.model_validate_json(row["eval_blob"]) if row["eval_blob"] else None
    # A historical round (round_n < current_round) always has a persisted
    # candidate; the placeholders here only matter for partially-written rows
    # observed mid-write during resume.
    return Round(
        refinement_round=int(row["round_n"]),
        candidate=candidate or _PLACEHOLDER_CANDIDATE,
        quality=quality or QualityCheck(passed=False),
        evaluation=evaluation,
        reflection=row["reflection"],
    )


def request_to_row(r: LLMRequest) -> dict:
    """Serialize an outbound LLMRequest into the kwargs Store.insert_requests expects."""
    return {
        "request_id": r.request_id,
        "item_id": r.item_id,
        "round_n": r.round_n,
        "role": r.role,
        "model_key": r.model_key,
        "attempt": r.attempt,
        "messages": r.messages,
        "json_mode": r.json_mode,
        "parent_response_id": r.parent_response_id,
        "temperature": r.temperature,
        "max_tokens": r.max_tokens,
    }


def row_to_llm_request(r: RequestRow, domain: DomainAdapter | None = None) -> LLMRequest:
    """Hydrate a stored RequestRow back into an LLMRequest for fulfillment.

    The strict challenger `response_schema` is intentionally not persisted (a raw
    Pydantic class in SQLite would be a serialization hazard); it's rebuilt here
    from ``role`` + ``domain.payload_model()`` so resumed/re-claimed challenger
    requests enforce the same shape as fresh ones. ``domain=None`` (e.g. the
    batch path, which sends plain JSON mode) yields a schema-less request.
    """
    response_schema = None
    if domain is not None and r.role == "challenger":
        response_schema = challenger_schema_for(domain.payload_model())
    return LLMRequest(
        request_id=r.request_id,
        item_id=r.item_id,
        round_n=r.round_n,
        role=r.role,
        model_key=r.model_key,
        messages=r.messages,
        json_mode=r.json_mode,
        attempt=r.attempt,
        parent_response_id=r.parent_response_id,
        temperature=r.temperature,
        max_tokens=r.max_tokens,
        response_schema=response_schema,
    )


def accepted_extras(item: ItemState, round_obj: Round) -> dict:
    """Compute the trailing metadata fields written into an accepted record."""
    ev = round_obj.evaluation
    return {
        "run_id": item.run_id,
        "item_id": item.item_id,
        "source_id": item.source_id,
        "refinement_round": round_obj.refinement_round,
        "weak_avg": ev.weak_avg if ev else None,
        "strong_avg": ev.strong_avg if ev else None,
        "gap": ev.gap if ev else None,
        "weak_scores": [s.model_dump() for s in (ev.weak_scores if ev else [])],
        "strong_scores": [s.model_dump() for s in (ev.strong_scores if ev else [])],
        "acceptance_rationale": ev.acceptance_rationale if ev else None,
    }
