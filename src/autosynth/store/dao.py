"""Thread-safe SQLite data access for one autosynth run."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from autosynth.schemas import Candidate, EvalReport, QualityCheck, SolverScore
from autosynth.store.schema import (
    _MIGRATIONS,
    _SCHEMA_SQL,
    ITEM_PENDING,
    REQ_DONE,
    REQ_FAILED,
    REQ_IN_FLIGHT,
    REQ_PENDING,
    RUN_STATUS_ABORTED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_RUNNING,
    SCHEMA_VERSION,
    TERMINAL_ITEM_STATES,
    _dumps,
    _loads,
    _utcnow,
)
from autosynth.store.types import RequestRow, ResponseRow
from autosynth.utils import stable_id


class Store:
    """One shared SQLite connection with serialized writes."""

    def __init__(self, db_path: Path | str):
        if sqlite3.sqlite_version_info < (3, 35):
            raise RuntimeError(
                f"SQLite >= 3.35 required (UPDATE ... RETURNING); have {sqlite3.sqlite_version}"
            )
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

    # Schema

    def _migrate(self) -> None:
        with self._lock:
            version = self.conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                self.conn.executescript(_SCHEMA_SQL)
                self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                self.conn.commit()
                return
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"unsupported schema version {version}; expected {SCHEMA_VERSION}")
            # Each step's DDL and its user_version bump run in one transaction,
            # so an interrupted upgrade rolls back and re-runs on the next open.
            for target, statements in _MIGRATIONS:
                if version < target:
                    with self.tx() as cur:
                        for sql in statements:
                            cur.execute(sql)
                        cur.execute(f"PRAGMA user_version = {target}")
                    version = target
            if version != SCHEMA_VERSION:
                raise RuntimeError(f"unsupported schema version {version}; expected {SCHEMA_VERSION}")

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

    # Runs

    def create_run(
        self, run_id: str, *, config: Any, harness: Any = None, cost_usd_cap: float | None = None
    ) -> None:
        now = _utcnow()
        with self.tx() as cur:
            cur.execute(
                """INSERT INTO runs (run_id, config_blob, harness_blob, started_at,
                                     last_active_at, status, cost_usd_cap)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, _dumps(config), _dumps(harness), now, now, RUN_STATUS_RUNNING, cost_usd_cap),
            )

    def update_run_status(self, run_id: str, status: str) -> None:
        now = _utcnow()
        finished = now if status in (RUN_STATUS_COMPLETED, RUN_STATUS_ABORTED) else None
        with self.tx() as cur:
            cur.execute(
                "UPDATE runs SET status=?, last_active_at=?, finished_at=COALESCE(?, finished_at) WHERE run_id=?",
                (status, now, finished, run_id),
            )

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def first_run(self) -> sqlite3.Row | None:
        """Return any one run row, or None. Convenient for single-run db files."""
        return self.conn.execute("SELECT * FROM runs LIMIT 1").fetchone()

    def touch_run(self, run_id: str) -> None:
        with self.tx() as cur:
            cur.execute("UPDATE runs SET last_active_at=? WHERE run_id=?", (_utcnow(), run_id))

    # Items

    def insert_item(
        self, *, run_id: str, source_id: str, domain: str, state: str, source_metadata: dict | None = None
    ) -> str:
        item_id = stable_id(run_id, source_id)
        now = _utcnow()
        with self.tx() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO items
                   (item_id, run_id, source_id, domain, state, current_round,
                    source_metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (item_id, run_id, source_id, domain, state, _dumps(source_metadata), now, now),
            )
        return item_id

    def update_item(
        self,
        item_id: str,
        *,
        state: str | None = None,
        current_round: int | None = None,
        final_round: int | None = None,
        rejection_reasons: list[str] | None = None,
        consumed_seq: int | None = None,
    ) -> None:
        """Update an item and optionally advance its response watermark."""
        sets: list[str] = ["updated_at = ?"]
        vals: list[Any] = [_utcnow()]
        if consumed_seq is not None:
            sets.append("consumed_seq = ?")
            vals.append(consumed_seq)
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

    def get_item(self, item_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()

    def items_ready_to_advance(self, run_id: str, limit: int = 100) -> list[sqlite3.Row]:
        """Items with unconsumed responses (rowid > consumed_seq), ready for a step()."""
        return self.conn.execute(
            f"""SELECT i.* FROM items i
               WHERE i.run_id = ?
                 AND i.state NOT IN ({",".join("?" * len(TERMINAL_ITEM_STATES))})
                 AND EXISTS (
                    SELECT 1 FROM responses r
                    JOIN requests q ON q.request_id = r.request_id
                    WHERE q.item_id = i.item_id AND r.rowid > i.consumed_seq
                 )
               ORDER BY i.updated_at
               LIMIT ?""",
            (run_id, *TERMINAL_ITEM_STATES, limit),
        ).fetchall()

    def items_pending_first_step(self, run_id: str, limit: int = 100) -> list[sqlite3.Row]:
        """Items in PENDING state that need their first step."""
        return self.conn.execute(
            "SELECT * FROM items WHERE run_id=? AND state=? ORDER BY created_at LIMIT ?",
            (run_id, ITEM_PENDING, limit),
        ).fetchall()

    def items_terminal_counts(self, run_id: str) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT state, COUNT(*) AS n FROM items WHERE run_id=? GROUP BY state", (run_id,)
        )
        return {row["state"]: row["n"] for row in cur.fetchall()}

    def items_for_run(self, run_id: str, *, stuck_only: bool = False) -> list[sqlite3.Row]:
        """List items for a run; with `stuck_only`, exclude terminal states."""
        if stuck_only:
            return self.conn.execute(
                f"""SELECT item_id, source_id, state, current_round, final_round, rejection_reasons
                    FROM items WHERE run_id=?
                    AND state NOT IN ({",".join("?" * len(TERMINAL_ITEM_STATES))})
                    ORDER BY updated_at""",
                (run_id, *TERMINAL_ITEM_STATES),
            ).fetchall()
        return self.conn.execute(
            """SELECT item_id, source_id, state, current_round, final_round, rejection_reasons
               FROM items WHERE run_id=? ORDER BY updated_at""",
            (run_id,),
        ).fetchall()

    def has_non_terminal_items(self, run_id: str) -> bool:
        row = self.conn.execute(
            f"""SELECT 1 FROM items WHERE run_id=?
                AND state NOT IN ({",".join("?" * len(TERMINAL_ITEM_STATES))}) LIMIT 1""",
            (run_id, *TERMINAL_ITEM_STATES),
        ).fetchone()
        return row is not None

    # Rounds

    @staticmethod
    def round_id(item_id: str, round_n: int) -> str:
        return stable_id("round", item_id, round_n)

    def upsert_round(
        self,
        *,
        item_id: str,
        round_n: int,
        candidate: Candidate | None = None,
        quality: QualityCheck | None = None,
        evaluation: EvalReport | None = None,
        reflection: str | None = None,
    ) -> str:
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
                    (
                        rid,
                        item_id,
                        round_n,
                        _dumps(candidate),
                        _dumps(quality),
                        _dumps(evaluation),
                        reflection,
                        now,
                    ),
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

    def get_round(self, item_id: str, round_n: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM rounds WHERE item_id=? AND round_n=?", (item_id, round_n)
        ).fetchone()

    # Requests

    def insert_requests(self, requests: Iterable[dict]) -> int:
        """Insert requests in the pending state."""
        rows = list(requests)
        if not rows:
            return 0
        with self.tx() as cur:
            cur.executemany(
                """INSERT OR IGNORE INTO requests
                   (request_id, item_id, round_n, role, model_key, attempt,
                    messages_blob, json_mode, parent_response_id, status,
                    temperature, max_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        r["request_id"],
                        r["item_id"],
                        r["round_n"],
                        r["role"],
                        r["model_key"],
                        r.get("attempt", 0),
                        _dumps(r["messages"]),
                        1 if r.get("json_mode") else 0,
                        r.get("parent_response_id"),
                        REQ_PENDING,
                        r.get("temperature"),
                        r.get("max_tokens"),
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def claim_pending(self, limit: int, *, batch_id: str | None = None) -> list[RequestRow]:
        """Claim up to ``limit`` pending requests, optionally for a batch."""
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

    def pending_count(self, run_id: str | None = None) -> int:
        if run_id:
            row = self.conn.execute(
                """SELECT COUNT(*) FROM requests q JOIN items i ON i.item_id = q.item_id
                   WHERE q.status=? AND i.run_id=?""",
                (REQ_PENDING, run_id),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM requests WHERE status=?", (REQ_PENDING,)).fetchone()
        return int(row[0])

    def in_flight_count(self, run_id: str | None = None) -> int:
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

    def tag_batch(self, batch_id: str, request_ids: list[str]) -> None:
        """Stamp a batch_id on the given requests so the poll loop can find them."""
        if not request_ids:
            return
        with self.tx() as cur:
            cur.execute(
                f"UPDATE requests SET batch_id=? WHERE request_id IN ({','.join('?' * len(request_ids))})",
                (batch_id, *request_ids),
            )

    def request_ids_for_batch(self, batch_id: str) -> list[str]:
        """In-flight request ids tagged to a batch — used to reconcile any the
        provider's results omit before the tag is cleared."""
        cur = self.conn.execute(
            "SELECT request_id FROM requests WHERE batch_id=? AND status=?",
            (batch_id, REQ_IN_FLIGHT),
        )
        return [row["request_id"] for row in cur.fetchall()]

    def clear_batch(self, batch_id: str) -> None:
        """Remove the batch_id tag from all requests once results are processed."""
        with self.tx() as cur:
            cur.execute("UPDATE requests SET batch_id=NULL WHERE batch_id=?", (batch_id,))

    def get_request(self, request_id: str) -> RequestRow | None:
        row = self.conn.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
        return RequestRow.from_row(row) if row else None

    def requests_for_item(self, item_id: str, *, round_n: int | None = None) -> list[RequestRow]:
        if round_n is None:
            cur = self.conn.execute("SELECT * FROM requests WHERE item_id=? ORDER BY rowid", (item_id,))
        else:
            cur = self.conn.execute(
                "SELECT * FROM requests WHERE item_id=? AND round_n=? ORDER BY rowid",
                (item_id, round_n),
            )
        return [RequestRow.from_row(r) for r in cur.fetchall()]

    def pending_request_ids_for_item(self, item_id: str) -> list[str]:
        cur = self.conn.execute(
            "SELECT request_id FROM requests WHERE item_id=? AND status IN (?, ?)",
            (item_id, REQ_PENDING, REQ_IN_FLIGHT),
        )
        return [row["request_id"] for row in cur.fetchall()]

    def mark_request_failed(
        self,
        request_id: str,
        error: str,
        *,
        max_failures: int,
    ) -> RequestRow:
        """Requeue a failed request, or terminate it when it reaches the cap."""
        now = _utcnow()
        with self.tx() as cur:
            cur.execute(
                """UPDATE requests
                   SET failure_count = failure_count + 1,
                       last_error = ?,
                       status = CASE
                           WHEN failure_count + 1 >= ? THEN ?
                           ELSE ?
                       END,
                       completed_at = CASE
                           WHEN failure_count + 1 >= ? THEN ?
                           ELSE NULL
                       END
                   WHERE request_id = ?
                   RETURNING *""",
                (error, max_failures, REQ_FAILED, REQ_PENDING, max_failures, now, request_id),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"unknown request_id {request_id!r}")
            return RequestRow.from_row(row)

    # Responses

    def insert_response(
        self,
        *,
        request_id: str,
        model: str,
        text: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Insert a response and atomically mark its request done.

        Duplicate request IDs are ignored. Response rowids drive the watermark.
        """
        with self.tx() as cur:
            now = _utcnow()
            existing = cur.execute("SELECT 1 FROM responses WHERE response_id=?", (request_id,)).fetchone()
            if existing is not None:
                return
            cur.execute(
                """INSERT INTO responses
                   (response_id, request_id, model, text, prompt_tokens,
                    completion_tokens, cost_usd, duration_ms, received_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    request_id,
                    model,
                    text,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd,
                    duration_ms,
                    now,
                ),
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

    def get_response(self, response_id: str) -> ResponseRow | None:
        row = self.conn.execute("SELECT * FROM responses WHERE response_id=?", (response_id,)).fetchone()
        if row is None:
            return None
        return ResponseRow(**dict(row))

    def hydrate_responses(self, item_id: str, since_seq: int) -> list[sqlite3.Row]:
        """Return responses after ``since_seq`` with request and parent fields."""
        cur = self.conn.execute(
            """SELECT r.request_id            AS request_id,
                      r.rowid                  AS seq,
                      q.role                   AS role,
                      q.round_n                AS round_n,
                      q.attempt                AS attempt,
                      r.text                   AS text,
                      q.parent_response_id     AS parent_response_id,
                      pq.role                  AS parent_role,
                      pr.text                  AS parent_text
               FROM responses r
               JOIN requests q  ON q.request_id  = r.request_id
               LEFT JOIN requests  pq ON pq.request_id  = q.parent_response_id
               LEFT JOIN responses pr ON pr.response_id = q.parent_response_id
               WHERE q.item_id = ? AND r.rowid > ?
               ORDER BY r.request_id""",
            (item_id, since_seq),
        )
        return cur.fetchall()

    def cost_so_far(self, run_id: str) -> float:
        row = self.conn.execute("SELECT cost_usd_actual FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return float(row[0]) if row else 0.0

    # Solver scores

    def insert_score(
        self,
        *,
        item_id: str,
        round_n: int,
        score: SolverScore,
        solver_response_id: str,
        judge_response_id: str,
    ) -> str:
        round_id = self.round_id(item_id, round_n)
        score_id = stable_id("score", round_id, score.solver, score.attempt)
        with self.tx() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO solver_scores
                   (score_id, round_id, solver, attempt, total, per_criterion,
                    failure_modes, raw_response, solver_response_id, judge_response_id, correct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    score_id,
                    round_id,
                    score.solver,
                    score.attempt,
                    score.total,
                    _dumps(score.per_criterion),
                    _dumps(score.failure_modes),
                    score.raw_response,
                    solver_response_id,
                    judge_response_id,
                    None if score.correct is None else int(score.correct),
                ),
            )
        return score_id

    def scores_for_round(self, item_id: str, round_n: int) -> list[SolverScore]:
        round_id = self.round_id(item_id, round_n)
        cur = self.conn.execute(
            """SELECT solver, attempt, total, per_criterion, failure_modes, raw_response, correct
               FROM solver_scores WHERE round_id=? ORDER BY solver, attempt""",
            (round_id,),
        )
        out: list[SolverScore] = []
        for row in cur.fetchall():
            out.append(
                SolverScore(
                    solver=row["solver"],
                    attempt=row["attempt"],
                    raw_response=row["raw_response"] or "",
                    total=row["total"],
                    per_criterion=_loads(row["per_criterion"]) or {},
                    failure_modes=_loads(row["failure_modes"]) or [],
                    correct=None if row["correct"] is None else bool(row["correct"]),
                )
            )
        return out

    # Accepted dataset

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

    # Resume normalization

    def normalize_for_resume(self, *, run_id: str, max_request_failures: int) -> dict[str, int]:
        """Requeue interrupted local work and incomplete requests after restart."""
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

    def unrecoverable_items(
        self,
        run_id: str,
        max_request_failures: int,
    ) -> list[tuple[str, str | None]]:
        """Return items whose requests reached the failure cap."""
        cur = self.conn.execute(
            f"""SELECT i.item_id AS item_id,
                      MAX(q.last_error) AS last_error
               FROM items i
               JOIN requests q ON q.item_id = i.item_id
               WHERE i.run_id = ? AND q.status = ? AND q.failure_count >= ?
                 AND i.state NOT IN ({",".join("?" * len(TERMINAL_ITEM_STATES))})
               GROUP BY i.item_id""",
            (run_id, REQ_FAILED, max_request_failures, *TERMINAL_ITEM_STATES),
        )
        return [(row["item_id"], row["last_error"]) for row in cur.fetchall()]

    def failure_rounds(self, run_id: str) -> list[sqlite3.Row]:
        """Rows for failure aggregation: candidate, quality, evaluation, accepted."""
        cur = self.conn.execute(
            """SELECT r.candidate_blob, r.quality_blob, r.eval_blob, r.accepted
               FROM rounds r JOIN items i ON i.item_id = r.item_id
               WHERE i.run_id = ?""",
            (run_id,),
        )
        return cur.fetchall()

    # Export

    def export_jsonl(self, run_id: str, out_path: Path) -> int:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for rec in self.accepted_records(run_id):
                f.write(json.dumps(rec, default=str))
                f.write("\n")
                n += 1
        return n

    def export_hf(self, run_id: str, out_dir: Path) -> Path | None:
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
