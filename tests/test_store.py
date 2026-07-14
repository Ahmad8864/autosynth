"""SQLite schema, transactions, resume handling, and exports."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

import pytest

from autosynth.schemas import Candidate, EvalReport, QualityCheck, RubricCriterion, SolverScore
from autosynth.store import (
    REQ_DONE,
    REQ_FAILED,
    REQ_IN_FLIGHT,
    REQ_PENDING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_RUNNING,
    SCHEMA_VERSION,
    Store,
)
from autosynth.store.schema import _SCHEMA_SQL

# fixtures


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "run.db")
    s.create_run("r1", config={"foo": "bar"}, harness={"v": 1}, cost_usd_cap=10.0)
    return s


def _candidate(source_id: str = "s1") -> Candidate:
    return Candidate(
        candidate_id="cand-1",
        domain="d",
        source_id=source_id,
        payload={"q": "?"},
        rubric=[RubricCriterion(id="c1", description="x", weight=3)],
        reference_output="r",
    )


def _request(item_id: str, *, request_id: str, role: str = "weak", round_n: int = 1) -> dict:
    return {
        "request_id": request_id,
        "item_id": item_id,
        "round_n": round_n,
        "role": role,
        "model_key": "weak_solver",
        "attempt": 0,
        "messages": [{"role": "user", "content": "x"}],
        "json_mode": False,
    }


# schema


def test_user_version_set(store: Store):
    v = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert v == SCHEMA_VERSION


def test_wal_mode(store: Store):
    mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_foreign_keys_on(store: Store):
    fk = store.conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_all_tables_created(store: Store):
    names = {
        r["name"] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"runs", "items", "rounds", "requests", "responses", "solver_scores", "accepted"}.issubset(names)


def test_unsupported_version_raises(tmp_path: Path):
    s = Store(tmp_path / "x.db")
    s.conn.execute("PRAGMA user_version = 999")
    s.conn.commit()
    s.close()
    with pytest.raises(RuntimeError, match="unsupported schema version"):
        Store(tmp_path / "x.db")


# runs + items + rounds


def test_create_and_get_run(store: Store):
    row = store.get_run("r1")
    assert row is not None
    assert row["status"] == RUN_STATUS_RUNNING
    assert row["cost_usd_cap"] == 10.0
    assert json.loads(row["config_blob"]) == {"foo": "bar"}


def test_run_status_transition(store: Store):
    store.update_run_status("r1", RUN_STATUS_COMPLETED)
    row = store.get_run("r1")
    assert row is not None
    assert row["status"] == RUN_STATUS_COMPLETED
    assert row["finished_at"] is not None


def test_insert_item_idempotent(store: Store):
    a = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="PENDING")
    b = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="PENDING")
    assert a == b


def test_update_item_state_and_round(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="PENDING")
    store.update_item(iid, state="NEED_CANDIDATE", current_round=2)
    row = store.get_item(iid)
    assert row is not None
    assert row["state"] == "NEED_CANDIDATE"
    assert row["current_round"] == 2


def test_update_item_rejection_reasons(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    store.update_item(
        iid, state="REJECTED", final_round=3, rejection_reasons=["exhausted_rounds", "gap_too_small"]
    )
    row = store.get_item(iid)
    assert row is not None
    assert json.loads(row["rejection_reasons"]) == ["exhausted_rounds", "gap_too_small"]
    assert row["final_round"] == 3


def test_round_materialization_lifecycle(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_CANDIDATE")
    store.upsert_round(item_id=iid, round_n=1, candidate=_candidate())
    row = store.get_round(iid, 1)
    assert row is not None
    assert row["candidate_blob"] is not None
    assert row["quality_blob"] is None
    assert row["accepted"] == 0
    store.upsert_round(item_id=iid, round_n=1, quality=QualityCheck(passed=True))
    row = store.get_round(iid, 1)
    assert row is not None
    assert json.loads(row["quality_blob"])["passed"] is True
    assert row["candidate_blob"] is not None
    store.upsert_round(item_id=iid, round_n=1, evaluation=EvalReport(accepted=True, gap=0.5))
    store.finalize_round(iid, 1, accepted=True)
    row = store.get_round(iid, 1)
    assert row is not None
    assert row["accepted"] == 1
    assert row["ended_at"] is not None
    assert json.loads(row["eval_blob"])["accepted"] is True


def test_items_terminal_counts(store: Store):
    store.insert_item(run_id="r1", source_id="a", domain="qa", state="ACCEPTED")
    store.insert_item(run_id="r1", source_id="b", domain="qa", state="ACCEPTED")
    store.insert_item(run_id="r1", source_id="c", domain="qa", state="REJECTED")
    store.insert_item(run_id="r1", source_id="d", domain="qa", state="NEED_SCORES")
    counts = store.items_terminal_counts("r1")
    assert counts == {"ACCEPTED": 2, "REJECTED": 1, "NEED_SCORES": 1}


# requests + responses


def test_insert_requests_and_pending_count(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    n = store.insert_requests([_request(iid, request_id=f"q{i}") for i in range(5)])
    assert n == 5
    assert store.pending_count("r1") == 5


def test_insert_requests_idempotent_on_request_id(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.insert_requests([_request(iid, request_id="q1")])
    assert store.pending_count("r1") == 1


def test_claim_pending_returns_in_order(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id=f"q{i}") for i in range(10)])
    claimed = store.claim_pending(3)
    assert len(claimed) == 3
    assert all(c.status == REQ_IN_FLIGHT for c in claimed)
    assert store.pending_count("r1") == 7
    assert store.in_flight_count("r1") == 3


def test_claim_pending_atomic_under_threads(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id=f"q{i}") for i in range(50)])

    claimed_ids: list[str] = []
    lock = threading.Lock()

    def worker():
        rows = store.claim_pending(10)
        with lock:
            claimed_ids.extend(r.request_id for r in rows)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed_ids) == 50
    assert len(set(claimed_ids)) == 50
    assert store.pending_count("r1") == 0
    assert store.in_flight_count("r1") == 50


def test_insert_response_marks_request_done_and_charges_cost(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    store.insert_requests([_request(iid, request_id="q1", role="quality")])
    store.claim_pending(1)
    store.insert_response(request_id="q1", model="gpt-x", text="ok", cost_usd=0.03)
    req = store.get_request("q1")
    assert req is not None
    assert req.status == REQ_DONE
    resp = store.get_response("q1")
    assert resp is not None
    assert resp.text == "ok"
    assert store.cost_so_far("r1") == pytest.approx(0.03)


def test_insert_response_idempotent(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    store.insert_requests([_request(iid, request_id="q1")])
    store.insert_response(request_id="q1", model="m", text="t", cost_usd=0.01)
    store.insert_response(request_id="q1", model="m", text="t-again", cost_usd=0.99)
    resp = store.get_response("q1")
    assert resp is not None and resp.text == "t"
    assert store.cost_so_far("r1") == pytest.approx(0.01)


def test_pending_request_ids_for_item(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1"), _request(iid, request_id="q2")])
    store.claim_pending(1)
    store.insert_response(request_id="q1", model="m", text="t")
    ids = store.pending_request_ids_for_item(iid)
    assert set(ids) == {"q2"}


def test_mark_request_failed_requeues_below_cap(store: Store):
    """Failures below the cap return to pending without completing."""
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)
    row = store.mark_request_failed("q1", "timeout", max_failures=3)
    assert row.status == REQ_PENDING
    assert row.failure_count == 1
    assert row.last_error == "timeout"
    assert row.completed_at is None


def test_mark_request_failed_terminates_at_cap(store: Store):
    """A request at the cap becomes failed and marks its item unrecoverable."""
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)
    for _ in range(2):
        store.mark_request_failed("q1", "deterministic", max_failures=3)
    row = store.mark_request_failed("q1", "deterministic", max_failures=3)
    assert row.status == REQ_FAILED
    assert row.failure_count == 3
    assert row.completed_at is not None
    assert store.unrecoverable_items("r1", max_request_failures=3) == [
        (iid, "deterministic"),
    ]


# solver scores


def test_insert_score_references_two_responses(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.upsert_round(item_id=iid, round_n=1, candidate=_candidate())
    store.insert_requests([_request(iid, request_id="sol1"), _request(iid, request_id="jud1")])
    store.claim_pending(2)
    store.insert_response(request_id="sol1", model="m", text="weak attempt")
    store.insert_response(request_id="jud1", model="m", text="{}")

    score = SolverScore(
        solver="weak",
        attempt=0,
        raw_response="weak attempt",
        total=0.2,
        per_criterion={"c1": 0.2},
        failure_modes=["generic"],
    )
    score_id = store.insert_score(
        item_id=iid, round_n=1, score=score, solver_response_id="sol1", judge_response_id="jud1"
    )
    row = store.conn.execute("SELECT * FROM solver_scores WHERE score_id=?", (score_id,)).fetchone()
    assert row["solver_response_id"] == "sol1"
    assert row["judge_response_id"] == "jud1"
    assert row["total"] == pytest.approx(0.2)


def test_scores_for_round_round_trip(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.upsert_round(item_id=iid, round_n=1, candidate=_candidate())
    store.insert_requests([_request(iid, request_id=f"r{i}") for i in range(4)])
    store.claim_pending(4)
    for rid in ("r0", "r1", "r2", "r3"):
        store.insert_response(request_id=rid, model="m", text="t")
    for s in ("weak", "strong"):
        for a in (0, 1):
            score = SolverScore(solver=s, attempt=a, raw_response="x", total=0.5)
            store.insert_score(
                item_id=iid, round_n=1, score=score, solver_response_id="r0", judge_response_id="r1"
            )
    out = store.scores_for_round(iid, 1)
    assert len(out) == 4
    assert {s.solver for s in out} == {"weak", "strong"}


def test_score_persists_correct(store: Store):
    """Verifiable scores round-trip the correct verdict (True/False/None)."""
    iid = store.insert_item(run_id="r1", source_id="s1", domain="math", state="NEED_SCORES")
    store.upsert_round(item_id=iid, round_n=1, candidate=_candidate())
    store.insert_requests([_request(iid, request_id="sol0")])
    store.claim_pending(1)
    store.insert_response(request_id="sol0", model="m", text="ANSWER: 42")
    for solver, attempt, correct in [("weak", 0, False), ("weak", 1, None), ("strong", 0, True)]:
        score = SolverScore(
            solver=solver,
            attempt=attempt,
            raw_response="x",
            total=1.0 if correct else 0.0,
            correct=correct,
        )
        # Direct verification has no separate judge response.
        store.insert_score(
            item_id=iid, round_n=1, score=score, solver_response_id="sol0", judge_response_id="sol0"
        )
    out = {(s.solver, s.attempt): s.correct for s in store.scores_for_round(iid, 1)}
    assert out[("weak", 0)] is False
    assert out[("weak", 1)] is None
    assert out[("strong", 0)] is True


def test_migration_v1_to_v2_adds_correct(tmp_path: Path):
    """A v1 db gains the correct column on open, preserving old rows; reopen is idempotent."""
    db = tmp_path / "v1.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE items (item_id TEXT PRIMARY KEY);"
        "CREATE TABLE solver_scores (score_id TEXT PRIMARY KEY, total REAL NOT NULL);"
        "INSERT INTO solver_scores (score_id, total) VALUES ('old', 0.5);"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    store = Store(db)
    try:
        cols = {r[1] for r in store.conn.execute("PRAGMA table_info(solver_scores)")}
        assert "correct" in cols
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        row = store.conn.execute("SELECT total, correct FROM solver_scores WHERE score_id='old'").fetchone()
        assert row["total"] == pytest.approx(0.5)
        assert row["correct"] is None
    finally:
        store.close()
    Store(db).close()


def test_migration_v2_to_v3_adds_consumed_seq(tmp_path: Path):
    """A v2 db gains items.consumed_seq (the rowid watermark) on open."""
    db = tmp_path / "v2.db"
    conn = sqlite3.connect(db)
    conn.executescript(re.sub(r"\n\s*consumed_seq\s+INTEGER NOT NULL DEFAULT 0,", "", _SCHEMA_SQL))
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    assert "consumed_seq" not in {r[1] for r in conn.execute("PRAGMA table_info(items)")}
    conn.close()

    store = Store(db)
    try:
        assert "consumed_seq" in {r[1] for r in store.conn.execute("PRAGMA table_info(items)")}
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        store.close()


def test_hydrate_responses_uses_rowid_watermark(store: Store):
    """Hydration returns only responses beyond the rowid watermark."""
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id=f"q{i}") for i in range(2)])
    store.claim_pending(2)
    store.insert_response(request_id="q0", model="m", text="a")
    store.insert_response(request_id="q1", model="m", text="b")

    rows = store.hydrate_responses(iid, 0)
    assert {r["request_id"] for r in rows} == {"q0", "q1"}
    max_seq = max(r["seq"] for r in rows)
    assert store.hydrate_responses(iid, max_seq) == []


def _schema_fingerprint(conn: sqlite3.Connection) -> dict:
    """Return each table's columns and indexes without generated index names."""
    tables = sorted(
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    )
    fp: dict = {}
    for t in tables:
        cols = [
            (r["name"], r["type"], r["notnull"], r["pk"]) for r in conn.execute(f"PRAGMA table_info({t})")
        ]
        idx = []
        for ix in conn.execute(f"PRAGMA index_list({t})"):
            members = tuple(r["name"] for r in conn.execute(f"PRAGMA index_info({ix['name']})"))
            idx.append((ix["unique"], ix["origin"], members))
        fp[t] = {"columns": cols, "indexes": sorted(idx)}
    return fp


def test_fresh_and_migrated_schema_converge(tmp_path: Path):
    """Fresh and fully migrated databases must have identical schemas."""
    fresh = Store(tmp_path / "fresh.db")
    try:
        expected = _schema_fingerprint(fresh.conn)
    finally:
        fresh.close()

    v1_sql = re.sub(r"\n\s*correct\s+INTEGER,", "", _SCHEMA_SQL)
    v1_sql = re.sub(r"\n\s*consumed_seq\s+INTEGER NOT NULL DEFAULT 0,", "", v1_sql)
    db = tmp_path / "v1.db"
    conn = sqlite3.connect(db)
    conn.executescript(v1_sql)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    score_cols = {r[1] for r in conn.execute("PRAGMA table_info(solver_scores)")}
    item_cols = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
    conn.close()
    assert "correct" not in score_cols and "consumed_seq" not in item_cols

    migrated = Store(db)
    try:
        assert migrated.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert _schema_fingerprint(migrated.conn) == expected
    finally:
        migrated.close()


# accepted + export


def test_accepted_and_export_jsonl(store: Store, tmp_path: Path):
    iid1 = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    iid2 = store.insert_item(run_id="r1", source_id="s2", domain="qa", state="NEED_QUALITY")
    store.upsert_round(item_id=iid1, round_n=1, candidate=_candidate("s1"))
    store.upsert_round(item_id=iid2, round_n=1, candidate=_candidate("s2"))
    store.insert_accepted(item_id=iid1, round_n=1, payload={"input": "a", "gap": 0.5})
    store.insert_accepted(item_id=iid2, round_n=1, payload={"input": "b", "gap": 0.6})
    assert store.count_accepted("r1") == 2

    out = tmp_path / "exp" / "accepted.jsonl"
    n = store.export_jsonl("r1", out)
    assert n == 2
    lines = out.read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert {r["input"] for r in parsed} == {"a", "b"}


def test_insert_accepted_idempotent_on_round(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    store.upsert_round(item_id=iid, round_n=1, candidate=_candidate())
    a = store.insert_accepted(item_id=iid, round_n=1, payload={"x": 1})
    b = store.insert_accepted(item_id=iid, round_n=1, payload={"x": 2})
    assert a == b
    assert store.count_accepted("r1") == 1


# resume normalization (§4.2)


def test_resume_in_flight_local_to_pending(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["in_flight_to_pending"] == 1
    req = store.get_request("q1")
    assert req is not None and req.status == REQ_PENDING


def test_resume_in_flight_batch_unchanged(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1, batch_id="batch-xyz")
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["in_flight_to_pending"] == 0
    req = store.get_request("q1")
    assert req is not None
    assert req.status == REQ_IN_FLIGHT
    assert req.batch_id == "batch-xyz"


def test_resume_done_without_response_reverts_to_pending(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    # Simulate a crash between marking done and writing the response.
    with store.tx() as cur:
        cur.execute("UPDATE requests SET status=? WHERE request_id='q1'", (REQ_DONE,))
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["done_to_pending"] == 1
    req = store.get_request("q1")
    assert req is not None and req.status == REQ_PENDING


def test_resume_failed_under_cap_reverts_to_pending(store: Store):
    """Resume requeues an under-cap request left in the failed state."""
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)
    # Bypass the normal in-run requeue to reproduce the crash state.
    with store.tx() as cur:
        cur.execute(
            "UPDATE requests SET status=?, failure_count=1 WHERE request_id='q1'",
            (REQ_FAILED,),
        )
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["failed_to_pending"] == 1
    req = store.get_request("q1")
    assert req is not None and req.status == REQ_PENDING


def test_resume_failed_at_cap_stays_failed(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)
    for _ in range(3):
        store.mark_request_failed("q1", "err", max_failures=3)
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["failed_to_pending"] == 0
    req = store.get_request("q1")
    assert req is not None
    assert req.failure_count == 3
    assert req.status == REQ_FAILED
    assert (iid, "err") in store.unrecoverable_items("r1", max_request_failures=3)
