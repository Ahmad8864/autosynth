"""Row DTOs for the store DAO.

The only DAO outputs that aren't bare ``sqlite3.Row``. They mirror the
request/response tables exactly; conversion to/from them is local to the DAO
and to the dispatcher's hydration layer.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from autodata.store.schema import _loads


@dataclass
class RequestRow:
    request_id: str
    item_id: str
    round_n: int
    role: str
    model_key: str
    attempt: int
    messages: list[dict[str, str]]
    json_mode: bool
    parent_response_id: str | None
    status: str
    submitted_at: str | None
    completed_at: str | None
    batch_id: str | None
    failure_count: int
    last_error: str | None
    temperature: float | None
    max_tokens: int | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RequestRow:
        return cls(
            request_id=row["request_id"],
            item_id=row["item_id"],
            round_n=row["round_n"],
            role=row["role"],
            model_key=row["model_key"],
            attempt=row["attempt"],
            messages=_loads(row["messages_blob"]) or [],
            json_mode=bool(row["json_mode"]),
            parent_response_id=row["parent_response_id"],
            status=row["status"],
            submitted_at=row["submitted_at"],
            completed_at=row["completed_at"],
            batch_id=row["batch_id"],
            failure_count=row["failure_count"],
            last_error=row["last_error"],
            temperature=row["temperature"],
            max_tokens=row["max_tokens"],
        )


@dataclass
class ResponseRow:
    response_id: str
    request_id: str
    model: str
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    duration_ms: int | None
    received_at: str
