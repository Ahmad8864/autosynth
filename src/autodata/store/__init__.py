"""SQLite-backed store for an autodata run.

One ``run.db`` per run. WAL mode, synchronous=NORMAL, foreign keys on.
All write operations go through :py:meth:`Store.tx` which takes the write
lock and a BEGIN IMMEDIATE transaction.

The schema is the canonical record of a run — the database *is* the run.
JSONL and HF exports are produced lazily from ``accepted`` rows via
:py:meth:`Store.export_jsonl` / :py:meth:`Store.export_hf`.

Layout:
  - :mod:`autodata.store.schema` — table DDL, status/state constants, JSON helpers
  - :mod:`autodata.store.types`  — ``RequestRow`` / ``ResponseRow`` DTOs
  - :mod:`autodata.store.dao`    — the ``Store`` class itself
"""
from __future__ import annotations

from autodata.store.dao import Store
from autodata.store.schema import (
    ITEM_ACCEPTED,
    ITEM_NEED_CANDIDATE,
    ITEM_NEED_QUALITY,
    ITEM_NEED_REFLECTION,
    ITEM_NEED_SCORES,
    ITEM_PENDING,
    ITEM_REJECTED,
    REQ_DONE,
    REQ_FAILED,
    REQ_IN_FLIGHT,
    REQ_PENDING,
    RUN_STATUS_ABORTED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_RUNNING,
    SCHEMA_VERSION,
    TERMINAL_ITEM_STATES,
)
from autodata.store.types import RequestRow, ResponseRow

__all__ = [
    "Store",
    "RequestRow",
    "ResponseRow",
    "SCHEMA_VERSION",
    "RUN_STATUS_RUNNING",
    "RUN_STATUS_COMPLETED",
    "RUN_STATUS_ABORTED",
    "REQ_PENDING",
    "REQ_IN_FLIGHT",
    "REQ_DONE",
    "REQ_FAILED",
    "ITEM_PENDING",
    "ITEM_NEED_CANDIDATE",
    "ITEM_NEED_QUALITY",
    "ITEM_NEED_SCORES",
    "ITEM_NEED_REFLECTION",
    "ITEM_ACCEPTED",
    "ITEM_REJECTED",
    "TERMINAL_ITEM_STATES",
]
