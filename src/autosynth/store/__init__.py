"""SQLite persistence for autosynth runs."""

from __future__ import annotations

from autosynth.store.dao import Store
from autosynth.store.schema import (
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
from autosynth.store.types import RequestRow, ResponseRow

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
