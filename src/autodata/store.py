"""SQLite-backed store for an autodata run.

One ``run.db`` per run. WAL mode, synchronous=NORMAL, foreign keys on.
All write operations go through :py:meth:`Store.tx` which takes the write
lock and a BEGIN IMMEDIATE transaction.

The schema is the canonical record of a run — the database *is* the run.
JSONL and HF exports are produced lazily from ``accepted`` rows via
:py:meth:`Store.export_jsonl` / :py:meth:`Store.export_hf`.

See MIGRATION_PLAN.md §3 for the spec.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from autodata.schemas import Candidate, EvalReport, QualityCheck, SolverScore
from autodata.utils import stable_id


SCHEMA_VERSION = 1

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
    last_error         TEXT
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
    judge_response_id  TEXT NOT NULL REFERENCES responses(response_id),
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


# Item / Run states are strings; we don't import the enum here to keep store
# free of pipeline dependencies. The pipeline module is the source of truth.
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_ABORTED = "aborted"

REQ_PENDING = "pending"
REQ_IN_FLIGHT = "in_flight"
REQ_DONE = "done"
REQ_FAILED = "failed"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json()
    return json.dumps(obj, default=str)


def _loads(text: Optional[str]) -> Any:
    if text is None:
        return None
    return json.loads(text)


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
    parent_response_id: Optional[str]
    status: str
    submitted_at: Optional[str]
    completed_at: Optional[str]
    batch_id: Optional[str]
    failure_count: int
    last_error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RequestRow":
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
        )


@dataclass
class ResponseRow:
    response_id: str
    request_id: str
    model: str
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cost_usd: Optional[float]
    duration_ms: Optional[int]
    received_at: str


class Store:
    """Thread-safe SQLite DAO for one run.

    A single connection is shared across threads with check_same_thread=False;
    writes are serialized via an RLock. WAL mode lets readers proceed
    concurrently with the single writer.
    """

    def __init__(self, db_path: Path | str):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        with self._lock:
            version = self.conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                self.conn.executescript(_SCHEMA_SQL)
                self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                self.conn.commit()
            elif version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported schema version {version}; expected {SCHEMA_VERSION}"
                )

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Cursor]:
        """Exclusive write transaction. Use for any mutating sequence."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                yield cur
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(self, run_id: str, *, config: Any, harness: Any = None,
                   cost_usd_cap: Optional[float] = None) -> None:
        now = _utcnow()
        with self.tx() as cur:
            cur.execute(
                """INSERT INTO runs (run_id, config_blob, harness_blob, started_at,
                                     last_active_at, status, cost_usd_cap)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, _dumps(config), _dumps(harness), now, now,
                 RUN_STATUS_RUNNING, cost_usd_cap),
            )

    def update_run_status(self, run_id: str, status: str) -> None:
        now = _utcnow()
        finished = now if status in (RUN_STATUS_COMPLETED, RUN_STATUS_ABORTED) else None
        with self.tx() as cur:
            cur.execute(
                "UPDATE runs SET status=?, last_active_at=?, finished_at=COALESCE(?, finished_at) WHERE run_id=?",
                (status, now, finished, run_id),
            )

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def touch_run(self, run_id: str) -> None:
        with self.tx() as cur:
            cur.execute("UPDATE runs SET last_active_at=? WHERE run_id=?", (_utcnow(), run_id))

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def insert_item(self, *, run_id: str, source_id: str, domain: str,
                    state: str, source_metadata: Optional[dict] = None) -> str:
        item_id = stable_id(run_id, source_id)
        now = _utcnow()
        with self.tx() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO items
                   (item_id, run_id, source_id, domain, state, current_round,
                    source_metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (item_id, run_id, source_id, domain, state,
                 _dumps(source_metadata), now, now),
            )
        return item_id

    def update_item(self, item_id: str, *,
                    state: Optional[str] = None,
                    current_round: Optional[int] = None,
                    final_round: Optional[int] = None,
                    rejection_reasons: Optional[list[str]] = None) -> None:
        sets: list[str] = ["updated_at = ?"]
        vals: list[Any] = [_utcnow()]
        if state is not None:
            sets.append("state = ?")
            vals.append(state)
        if current_round is not None:
            sets.append("current_round = ?")
            vals.append(current_round)
        if final_round is not None:
            sets.append("final_round = ?")
            vals.append(final_round)
        if rejection_reasons is not None:
            sets.append("rejection_reasons = ?")
            vals.append(_dumps(rejection_reasons))
        vals.append(item_id)
        with self.tx() as cur:
            cur.execute(f"UPDATE items SET {', '.join(sets)} WHERE item_id = ?", vals)

    def get_item(self, item_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()

    def items_ready_to_advance(self, run_id: str, limit: int = 100) -> list[sqlite3.Row]:
        """Items with unconsumed responses, ready for a step()."""
        return self.conn.execute(
            """SELECT i.* FROM items i
               WHERE i.run_id = ?
                 AND i.state NOT IN ('ACCEPTED','REJECTED')
                 AND EXISTS (
                    SELECT 1 FROM responses r
                    JOIN requests q ON q.request_id = r.request_id
                    WHERE q.item_id = i.item_id AND r.received_at > i.updated_at
                 )
               ORDER BY i.updated_at
               LIMIT ?""",
            (run_id, limit),
        ).fetchall()

    def items_pending_first_step(self, run_id: str, limit: int = 100) -> list[sqlite3.Row]:
        """Items in PENDING state that need their first step."""
        return self.conn.execute(
            "SELECT * FROM items WHERE run_id=? AND state='PENDING' ORDER BY created_at LIMIT ?",
            (run_id, limit),
        ).fetchall()

    def items_terminal_counts(self, run_id: str) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT state, COUNT(*) AS n FROM items WHERE run_id=? GROUP BY state", (run_id,)
        )
        return {row["state"]: row["n"] for row in cur.fetchall()}

    def has_non_terminal_items(self, run_id: str) -> bool:
        row = self.conn.execute(
            """SELECT 1 FROM items WHERE run_id=? AND state NOT IN ('ACCEPTED','REJECTED') LIMIT 1""",
            (run_id,),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Rounds
    # ------------------------------------------------------------------

    @staticmethod
    def round_id(item_id: str, round_n: int) -> str:
        return stable_id("round", item_id, round_n)

    def upsert_round(self, *, item_id: str, round_n: int,
                     candidate: Optional[Candidate] = None,
                     quality: Optional[QualityCheck] = None,
                     evaluation: Optional[EvalReport] = None,
                     reflection: Optional[str] = None) -> str:
        rid = self.round_id(item_id, round_n)
        now = _utcnow()
        with self.tx() as cur:
            existing = cur.execute("SELECT round_id FROM rounds WHERE round_id=?", (rid,)).fetchone()
            if existing is None:
                cur.execute(
                    """INSERT INTO rounds
                       (round_id, item_id, round_n, candidate_blob, quality_blob,
                        eval_blob, reflection, started_at, ended_at, accepted)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)""",
                    (rid, item_id, round_n, _dumps(candidate), _dumps(quality),
                     _dumps(evaluation), reflection, now),
                )
            else:
                sets = []
                vals: list[Any] = []
                if candidate is not None:
                    sets.append("candidate_blob = ?")
                    vals.append(_dumps(candidate))
                if quality is not None:
                    sets.append("quality_blob = ?")
                    vals.append(_dumps(quality))
                if evaluation is not None:
                    sets.append("eval_blob = ?")
                    vals.append(_dumps(evaluation))
                if reflection is not None:
                    sets.append("reflection = ?")
                    vals.append(reflection)
                if sets:
                    vals.append(rid)
                    cur.execute(f"UPDATE rounds SET {', '.join(sets)} WHERE round_id = ?", vals)
        return rid

    def finalize_round(self, item_id: str, round_n: int, accepted: bool) -> None:
        rid = self.round_id(item_id, round_n)
        with self.tx() as cur:
            cur.execute(
                "UPDATE rounds SET accepted=?, ended_at=? WHERE round_id=?",
                (1 if accepted else 0, _utcnow(), rid),
            )

    def rounds_for_item(self, item_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM rounds WHERE item_id=? ORDER BY round_n", (item_id,)
        ).fetchall()

    def get_round(self, item_id: str, round_n: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM rounds WHERE item_id=? AND round_n=?", (item_id, round_n)
        ).fetchone()

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def insert_requests(self, requests: Iterable[dict]) -> int:
        """Bulk-insert requests, all in 'pending' status.

        Each dict must have: request_id, item_id, round_n, role, model_key,
        attempt, messages, json_mode; optional: parent_response_id.
        """
        rows = list(requests)
        if not rows:
            return 0
        with self.tx() as cur:
            cur.executemany(
                """INSERT OR IGNORE INTO requests
                   (request_id, item_id, round_n, role, model_key, attempt,
                    messages_blob, json_mode, parent_response_id, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (r["request_id"], r["item_id"], r["round_n"], r["role"],
                     r["model_key"], r.get("attempt", 0),
                     _dumps(r["messages"]), 1 if r.get("json_mode") else 0,
                     r.get("parent_response_id"), REQ_PENDING)
                    for r in rows
                ],
            )
        return len(rows)

    def claim_pending(self, limit: int, *, batch_id: Optional[str] = None) -> list[RequestRow]:
        """Atomically transition up to `limit` pending requests to in_flight.

        Returns the claimed rows. With `batch_id`, also tags them for batch dispatch.
        """
        now = _utcnow()
        with self.tx() as cur:
            # UPDATE ... RETURNING is supported on SQLite >= 3.35.
            cur.execute(
                """UPDATE requests
                   SET status=?, submitted_at=?, batch_id=COALESCE(?, batch_id)
                   WHERE request_id IN (
                       SELECT request_id FROM requests
                       WHERE status=? ORDER BY rowid LIMIT ?
                   )
                   RETURNING *""",
                (REQ_IN_FLIGHT, now, batch_id, REQ_PENDING, limit),
            )
            return [RequestRow.from_row(row) for row in cur.fetchall()]

    def pending_count(self, run_id: Optional[str] = None) -> int:
        if run_id:
            row = self.conn.execute(
                """SELECT COUNT(*) FROM requests q JOIN items i ON i.item_id = q.item_id
                   WHERE q.status=? AND i.run_id=?""",
                (REQ_PENDING, run_id),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM requests WHERE status=?", (REQ_PENDING,)
            ).fetchone()
        return int(row[0])

    def in_flight_count(self, run_id: Optional[str] = None) -> int:
        if run_id:
            row = self.conn.execute(
                """SELECT COUNT(*) FROM requests q JOIN items i ON i.item_id = q.item_id
                   WHERE q.status=? AND i.run_id=?""",
                (REQ_IN_FLIGHT, run_id),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM requests WHERE status=?", (REQ_IN_FLIGHT,)
            ).fetchone()
        return int(row[0])

    def outstanding_batch_ids(self, run_id: str) -> list[str]:
        cur = self.conn.execute(
            """SELECT DISTINCT q.batch_id FROM requests q
               JOIN items i ON i.item_id = q.item_id
               WHERE q.status=? AND q.batch_id IS NOT NULL AND i.run_id=?""",
            (REQ_IN_FLIGHT, run_id),
        )
        return [row["batch_id"] for row in cur.fetchall()]

    def get_request(self, request_id: str) -> Optional[RequestRow]:
        row = self.conn.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
        return RequestRow.from_row(row) if row else None

    def requests_for_item(self, item_id: str, *, round_n: Optional[int] = None) -> list[RequestRow]:
        if round_n is None:
            cur = self.conn.execute("SELECT * FROM requests WHERE item_id=? ORDER BY rowid", (item_id,))
        else:
            cur = self.conn.execute(
                "SELECT * FROM requests WHERE item_id=? AND round_n=? ORDER BY rowid", (item_id, round_n),
            )
        return [RequestRow.from_row(r) for r in cur.fetchall()]

    def pending_request_ids_for_item(self, item_id: str) -> list[str]:
        cur = self.conn.execute(
            "SELECT request_id FROM requests WHERE item_id=? AND status IN (?, ?)",
            (item_id, REQ_PENDING, REQ_IN_FLIGHT),
        )
        return [row["request_id"] for row in cur.fetchall()]

    def mark_request_failed(self, request_id: str, error: str) -> RequestRow:
        with self.tx() as cur:
            cur.execute(
                """UPDATE requests SET status=?, failure_count=failure_count+1,
                                       last_error=?, completed_at=?
                   WHERE request_id=?
                   RETURNING *""",
                (REQ_FAILED, error, _utcnow(), request_id),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"unknown request_id {request_id!r}")
            return RequestRow.from_row(row)

    # ------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------

    def insert_response(self, *, request_id: str, model: str, text: str,
                        prompt_tokens: Optional[int] = None,
                        completion_tokens: Optional[int] = None,
                        cost_usd: Optional[float] = None,
                        duration_ms: Optional[int] = None) -> None:
        """Insert a response and atomically mark the request done.

        Idempotent on request_id (skipped if a response row already exists).
        """
        now = _utcnow()
        with self.tx() as cur:
            existing = cur.execute(
                "SELECT 1 FROM responses WHERE response_id=?", (request_id,)
            ).fetchone()
            if existing is not None:
                return
            cur.execute(
                """INSERT INTO responses
                   (response_id, request_id, model, text, prompt_tokens,
                    completion_tokens, cost_usd, duration_ms, received_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (request_id, request_id, model, text, prompt_tokens,
                 completion_tokens, cost_usd, duration_ms, now),
            )
            cur.execute(
                "UPDATE requests SET status=?, completed_at=? WHERE request_id=?",
                (REQ_DONE, now, request_id),
            )
            if cost_usd is not None:
                cur.execute(
                    """UPDATE runs SET cost_usd_actual = cost_usd_actual + ?
                       WHERE run_id = (SELECT i.run_id FROM items i
                                       JOIN requests q ON q.item_id = i.item_id
                                       WHERE q.request_id = ?)""",
                    (cost_usd, request_id),
                )

    def get_response(self, response_id: str) -> Optional[ResponseRow]:
        row = self.conn.execute("SELECT * FROM responses WHERE response_id=?", (response_id,)).fetchone()
        if row is None:
            return None
        return ResponseRow(**dict(row))

    def responses_since(self, item_id: str, since_ts: str) -> list[ResponseRow]:
        cur = self.conn.execute(
            """SELECT r.* FROM responses r
               JOIN requests q ON q.request_id = r.request_id
               WHERE q.item_id=? AND r.received_at > ?
               ORDER BY r.request_id""",
            (item_id, since_ts),
        )
        return [ResponseRow(**dict(row)) for row in cur.fetchall()]

    def cost_so_far(self, run_id: str) -> float:
        row = self.conn.execute(
            "SELECT cost_usd_actual FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return float(row[0]) if row else 0.0

    # ------------------------------------------------------------------
    # Solver scores
    # ------------------------------------------------------------------

    def insert_score(self, *, item_id: str, round_n: int, score: SolverScore,
                     solver_response_id: str, judge_response_id: str) -> str:
        round_id = self.round_id(item_id, round_n)
        score_id = stable_id("score", round_id, score.solver, score.attempt)
        with self.tx() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO solver_scores
                   (score_id, round_id, solver, attempt, total, per_criterion,
                    failure_modes, raw_response, solver_response_id, judge_response_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (score_id, round_id, score.solver, score.attempt, score.total,
                 _dumps(score.per_criterion), _dumps(score.failure_modes),
                 score.raw_response, solver_response_id, judge_response_id),
            )
        return score_id

    def scores_for_round(self, item_id: str, round_n: int) -> list[SolverScore]:
        round_id = self.round_id(item_id, round_n)
        cur = self.conn.execute(
            """SELECT solver, attempt, total, per_criterion, failure_modes, raw_response
               FROM solver_scores WHERE round_id=? ORDER BY solver, attempt""",
            (round_id,),
        )
        out: list[SolverScore] = []
        for row in cur.fetchall():
            out.append(SolverScore(
                solver=row["solver"],
                attempt=row["attempt"],
                raw_response=row["raw_response"] or "",
                total=row["total"],
                per_criterion=_loads(row["per_criterion"]) or {},
                failure_modes=_loads(row["failure_modes"]) or [],
            ))
        return out

    # ------------------------------------------------------------------
    # Accepted dataset
    # ------------------------------------------------------------------

    def insert_accepted(self, *, item_id: str, round_n: int, payload: dict) -> str:
        round_id = self.round_id(item_id, round_n)
        accepted_id = stable_id("accepted", item_id, round_id)
        with self.tx() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO accepted
                   (accepted_id, item_id, round_id, payload_blob, accepted_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (accepted_id, item_id, round_id, _dumps(payload), _utcnow()),
            )
        return accepted_id

    def count_accepted(self, run_id: str) -> int:
        row = self.conn.execute(
            """SELECT COUNT(*) FROM accepted a
               JOIN items i ON i.item_id = a.item_id WHERE i.run_id=?""",
            (run_id,),
        ).fetchone()
        return int(row[0])

    def accepted_records(self, run_id: str) -> Iterator[dict]:
        cur = self.conn.execute(
            """SELECT a.payload_blob FROM accepted a
               JOIN items i ON i.item_id = a.item_id WHERE i.run_id=?
               ORDER BY a.accepted_at""",
            (run_id,),
        )
        for row in cur.fetchall():
            yield _loads(row["payload_blob"])

    # ------------------------------------------------------------------
    # Resume normalization (§4.2)
    # ------------------------------------------------------------------

    def normalize_for_resume(self, *, run_id: str, max_request_failures: int) -> dict[str, int]:
        """Reconcile request states after a restart. See §4.2 table.

        - in_flight w/o batch_id  → pending (local fulfill lost its work)
        - in_flight w/ batch_id   → leave (batch poll will fetch)
        - done w/o response row   → pending (crash between insert_response and mark_done)
        - failed w/ count < cap   → pending (let it retry)
        - failed w/ count >= cap  → leave; item goes REJECTED("unrecoverable") elsewhere
        """
        counts = {"in_flight_to_pending": 0, "done_to_pending": 0, "failed_to_pending": 0}
        with self.tx() as cur:
            cur.execute(
                """UPDATE requests SET status=? WHERE status=? AND batch_id IS NULL
                   AND request_id IN (
                       SELECT q.request_id FROM requests q
                       JOIN items i ON i.item_id=q.item_id WHERE i.run_id=?)""",
                (REQ_PENDING, REQ_IN_FLIGHT, run_id),
            )
            counts["in_flight_to_pending"] = cur.rowcount

            cur.execute(
                """UPDATE requests SET status=? WHERE status=?
                   AND request_id NOT IN (SELECT request_id FROM responses)
                   AND request_id IN (
                       SELECT q.request_id FROM requests q
                       JOIN items i ON i.item_id=q.item_id WHERE i.run_id=?)""",
                (REQ_PENDING, REQ_DONE, run_id),
            )
            counts["done_to_pending"] = cur.rowcount

            cur.execute(
                """UPDATE requests SET status=? WHERE status=? AND failure_count < ?
                   AND request_id IN (
                       SELECT q.request_id FROM requests q
                       JOIN items i ON i.item_id=q.item_id WHERE i.run_id=?)""",
                (REQ_PENDING, REQ_FAILED, max_request_failures, run_id),
            )
            counts["failed_to_pending"] = cur.rowcount
        return counts

    def unrecoverable_items(self, run_id: str, max_request_failures: int) -> list[str]:
        """Items owning a request that has hit the failure cap."""
        cur = self.conn.execute(
            """SELECT DISTINCT i.item_id FROM items i
               JOIN requests q ON q.item_id = i.item_id
               WHERE i.run_id = ? AND q.status = ? AND q.failure_count >= ?
                 AND i.state NOT IN ('ACCEPTED','REJECTED')""",
            (run_id, REQ_FAILED, max_request_failures),
        )
        return [row["item_id"] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(self, run_id: str, out_path: Path) -> int:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for rec in self.accepted_records(run_id):
                f.write(json.dumps(rec, default=str))
                f.write("\n")
                n += 1
        return n

    def export_hf(self, run_id: str, out_dir: Path) -> Optional[Path]:
        try:
            from datasets import Dataset  # type: ignore
        except ImportError:
            return None
        records = list(self.accepted_records(run_id))
        if not records:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(records).save_to_disk(str(out_dir))
        return out_dir
