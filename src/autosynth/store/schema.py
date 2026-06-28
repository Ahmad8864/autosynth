"""SQLite schema, status/state constants, and JSON serialization helpers.

The DAO in :mod:`autosynth.store.dao` is the only consumer. These live in their
own module so the schema can be read top-to-bottom without scrolling past
~600 lines of methods, and so other subpackages that only need the constants
(``REQ_PENDING``, ``ITEM_ACCEPTED``, etc.) don't have to import the DAO class.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 3

_SCHEMA_SQL = """
CREATE TABLE runs (
    run_id            TEXT PRIMARY KEY,
    config_blob       TEXT NOT NULL,
    harness_blob      TEXT,
    started_at        TEXT NOT NULL,
    last_active_at    TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL,
    cost_usd_cap      REAL,
    cost_usd_actual   REAL NOT NULL DEFAULT 0
);

CREATE TABLE items (
    item_id           TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    domain            TEXT NOT NULL,
    state             TEXT NOT NULL,
    current_round     INTEGER NOT NULL DEFAULT 1,
    source_metadata   TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    final_round       INTEGER,
    rejection_reasons TEXT,
    -- Watermark: the max responses.rowid this item has consumed. The dispatcher
    -- advances an item when a response with rowid > consumed_seq exists. rowid is
    -- a strictly-monotonic integer (responses are append-only), unlike a
    -- wall-clock received_at which can tie at microsecond resolution. (Kept last
    -- to match the v3 ALTER ... ADD COLUMN position — see _MIGRATIONS.)
    consumed_seq      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, source_id)
);
CREATE INDEX items_run_state ON items(run_id, state, updated_at);

CREATE TABLE rounds (
    round_id          TEXT PRIMARY KEY,
    item_id           TEXT NOT NULL REFERENCES items(item_id),
    round_n           INTEGER NOT NULL,
    candidate_blob    TEXT,
    quality_blob      TEXT,
    eval_blob         TEXT,
    reflection        TEXT,
    started_at        TEXT NOT NULL,
    ended_at          TEXT,
    accepted          INTEGER NOT NULL DEFAULT 0,
    UNIQUE(item_id, round_n)
);
CREATE INDEX rounds_item ON rounds(item_id);

CREATE TABLE requests (
    request_id         TEXT PRIMARY KEY,
    item_id            TEXT NOT NULL REFERENCES items(item_id),
    round_n            INTEGER NOT NULL,
    role               TEXT NOT NULL,
    model_key          TEXT NOT NULL,
    attempt            INTEGER NOT NULL DEFAULT 0,
    messages_blob      TEXT NOT NULL,
    json_mode          INTEGER NOT NULL DEFAULT 0,
    parent_response_id TEXT,
    status             TEXT NOT NULL,
    submitted_at       TEXT,
    completed_at       TEXT,
    batch_id           TEXT,
    failure_count      INTEGER NOT NULL DEFAULT 0,
    last_error         TEXT,
    -- Per-call sampling values from the role's ModelConfig at build time.
    -- Persisted so resume reproduces the original call exactly.
    temperature        REAL,
    max_tokens         INTEGER
);
CREATE INDEX requests_status ON requests(status);
CREATE INDEX requests_item ON requests(item_id, round_n);
CREATE INDEX requests_batch ON requests(batch_id) WHERE batch_id IS NOT NULL;

CREATE TABLE responses (
    response_id        TEXT PRIMARY KEY,
    request_id         TEXT NOT NULL REFERENCES requests(request_id),
    model              TEXT NOT NULL,
    text               TEXT NOT NULL,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    cost_usd           REAL,
    duration_ms        INTEGER,
    received_at        TEXT NOT NULL
);
CREATE INDEX responses_received ON responses(received_at);
CREATE INDEX responses_request ON responses(request_id);

CREATE TABLE solver_scores (
    score_id           TEXT PRIMARY KEY,
    round_id           TEXT NOT NULL REFERENCES rounds(round_id),
    solver             TEXT NOT NULL,
    attempt            INTEGER NOT NULL,
    total              REAL NOT NULL,
    per_criterion      TEXT,
    failure_modes      TEXT,
    raw_response       TEXT,
    solver_response_id TEXT NOT NULL REFERENCES responses(response_id),
    -- In verifiable mode there is no judge: judge_response_id self-references
    -- the solver response, and correct holds the verify() verdict (else NULL).
    judge_response_id  TEXT NOT NULL REFERENCES responses(response_id),
    correct            INTEGER,
    UNIQUE(round_id, solver, attempt)
);
CREATE INDEX scores_round ON solver_scores(round_id, solver);

CREATE TABLE accepted (
    accepted_id        TEXT PRIMARY KEY,
    item_id            TEXT NOT NULL REFERENCES items(item_id),
    round_id           TEXT NOT NULL REFERENCES rounds(round_id),
    payload_blob       TEXT NOT NULL,
    accepted_at        TEXT NOT NULL
);
"""


# (target_version, ddl) applied in order to upgrade old dbs; fresh dbs get _SCHEMA_SQL.
_MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (2, ("ALTER TABLE solver_scores ADD COLUMN correct INTEGER",)),
    (3, ("ALTER TABLE items ADD COLUMN consumed_seq INTEGER NOT NULL DEFAULT 0",)),
)


RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_ABORTED = "aborted"

REQ_PENDING = "pending"
REQ_IN_FLIGHT = "in_flight"
REQ_DONE = "done"
REQ_FAILED = "failed"

# Item states. Defined here (not imported from pipeline.State) so the store
# can stay free of pipeline dependencies and so raw SQL strings reference a
# single source of truth. pipeline.State mirrors these values.
ITEM_PENDING = "PENDING"
ITEM_NEED_CANDIDATE = "NEED_CANDIDATE"
ITEM_NEED_QUALITY = "NEED_QUALITY"
ITEM_NEED_SCORES = "NEED_SCORES"
ITEM_NEED_DECISION = "NEED_DECISION"
ITEM_NEED_REFLECTION = "NEED_REFLECTION"
ITEM_ACCEPTED = "ACCEPTED"
ITEM_REJECTED = "REJECTED"
TERMINAL_ITEM_STATES = (ITEM_ACCEPTED, ITEM_REJECTED)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json()
    return json.dumps(obj, default=str)


def _loads(text: str | None) -> Any:
    if text is None:
        return None
    return json.loads(text)
