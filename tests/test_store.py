"""Tests for the SQLite Store.

Covers the schema, transactional invariants (claim_pending atomicity under
threads), round materialization timing, the resume normalization table
from MIGRATION_PLAN.md §4.2, and JSONL export.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from autodata.schemas import Candidate, EvalReport, QualityCheck, RubricCriterion, SolverScore
from autodata.store import (
    REQ_DONE,
    REQ_FAILED,
    REQ_IN_FLIGHT,
    REQ_PENDING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_RUNNING,
    SCHEMA_VERSION,
    Store,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "run.db")
    s.create_run("r1", config={"foo": "bar"}, harness={"v": 1}, cost_usd_cap=10.0)
    return s


def _candidate(source_id: str = "s1") -> Candidate:
    return Candidate(
        candidate_id="cand-1", domain="d", source_id=source_id, payload={"q": "?"},
        rubric=[RubricCriterion(id="c1", description="x", weight=3)],
        reference_output="r",
    )


def _request(item_id: str, *, request_id: str, role: str = "weak", round_n: int = 1) -> dict:
    return {
        "request_id": request_id, "item_id": item_id, "round_n": round_n,
        "role": role, "model_key": "weak_solver", "attempt": 0,
        "messages": [{"role": "user", "content": "x"}], "json_mode": False,
    }


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

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
    names = {r["name"] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"runs", "items", "rounds", "requests", "responses",
            "solver_scores", "accepted"}.issubset(names)


def test_unsupported_version_raises(tmp_path: Path):
    s = Store(tmp_path / "x.db")
    s.conn.execute("PRAGMA user_version = 999")
    s.conn.commit()
    s.close()
    with pytest.raises(RuntimeError, match="unsupported schema version"):
        Store(tmp_path / "x.db")


# ---------------------------------------------------------------------------
# runs + items + rounds
# ---------------------------------------------------------------------------

def test_create_and_get_run(store: Store):
    row = store.get_run("r1")
    assert row["status"] == RUN_STATUS_RUNNING
    assert row["cost_usd_cap"] == 10.0
    assert json.loads(row["config_blob"]) == {"foo": "bar"}


def test_run_status_transition(store: Store):
    store.update_run_status("r1", RUN_STATUS_COMPLETED)
    row = store.get_run("r1")
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
    assert row["state"] == "NEED_CANDIDATE"
    assert row["current_round"] == 2


def test_update_item_rejection_reasons(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    store.update_item(iid, state="REJECTED", final_round=3,
                      rejection_reasons=["exhausted_rounds", "gap_too_small"])
    row = store.get_item(iid)
    assert json.loads(row["rejection_reasons"]) == ["exhausted_rounds", "gap_too_small"]
    assert row["final_round"] == 3


def test_round_materialization_lifecycle(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_CANDIDATE")
    # 1. insert with candidate only
    store.upsert_round(item_id=iid, round_n=1, candidate=_candidate())
    row = store.get_round(iid, 1)
    assert row["candidate_blob"] is not None
    assert row["quality_blob"] is None
    assert row["accepted"] == 0
    # 2. attach quality
    store.upsert_round(item_id=iid, round_n=1, quality=QualityCheck(passed=True))
    row = store.get_round(iid, 1)
    assert json.loads(row["quality_blob"])["passed"] is True
    assert row["candidate_blob"] is not None   # not overwritten
    # 3. attach eval + finalize accepted
    store.upsert_round(item_id=iid, round_n=1, evaluation=EvalReport(accepted=True, gap=0.5))
    store.finalize_round(iid, 1, accepted=True)
    row = store.get_round(iid, 1)
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


# ---------------------------------------------------------------------------
# requests + responses
# ---------------------------------------------------------------------------

def test_insert_requests_and_pending_count(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    n = store.insert_requests([_request(iid, request_id=f"q{i}") for i in range(5)])
    assert n == 5
    assert store.pending_count("r1") == 5


def test_insert_requests_idempotent_on_request_id(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.insert_requests([_request(iid, request_id="q1")])   # duplicate
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

    assert len(claimed_ids) == 50              # exactly the 50 we inserted
    assert len(set(claimed_ids)) == 50         # no duplicates
    assert store.pending_count("r1") == 0
    assert store.in_flight_count("r1") == 50


def test_insert_response_marks_request_done_and_charges_cost(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    store.insert_requests([_request(iid, request_id="q1", role="quality")])
    store.claim_pending(1)
    store.insert_response(request_id="q1", model="gpt-x", text="ok", cost_usd=0.03)
    req = store.get_request("q1")
    assert req.status == REQ_DONE
    resp = store.get_response("q1")
    assert resp is not None
    assert resp.text == "ok"
    assert store.cost_so_far("r1") == pytest.approx(0.03)


def test_insert_response_idempotent(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    store.insert_requests([_request(iid, request_id="q1")])
    store.insert_response(request_id="q1", model="m", text="t", cost_usd=0.01)
    store.insert_response(request_id="q1", model="m", text="t-again", cost_usd=0.99)  # ignored
    assert store.get_response("q1").text == "t"
    assert store.cost_so_far("r1") == pytest.approx(0.01)


def test_responses_since(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_QUALITY")
    store.insert_requests([_request(iid, request_id=f"q{i}") for i in range(3)])
    for i in range(3):
        store.insert_response(request_id=f"q{i}", model="m", text=f"r{i}")
    item = store.get_item(iid)
    new = store.responses_since(iid, item["updated_at"])
    assert {r.request_id for r in new} == {"q0", "q1", "q2"}
    # ordered deterministically by request_id
    assert [r.request_id for r in new] == ["q0", "q1", "q2"]


def test_pending_request_ids_for_item(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1"), _request(iid, request_id="q2")])
    store.claim_pending(1)        # q1 -> in_flight, still counts as outstanding
    store.insert_response(request_id="q1", model="m", text="t")
    ids = store.pending_request_ids_for_item(iid)
    assert set(ids) == {"q2"}     # q1 is done; q2 still pending


def test_mark_request_failed_requeues_below_cap(store: Store):
    """Within-run retries: under the cap, the request goes back to PENDING
    so the dispatcher loop picks it up on the next tick. completed_at stays
    null because the request isn't done — it's just paused between attempts."""
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)
    row = store.mark_request_failed("q1", "timeout", max_failures=3)
    assert row.status == REQ_PENDING
    assert row.failure_count == 1
    assert row.last_error == "timeout"
    assert row.completed_at is None


def test_mark_request_failed_terminates_at_cap(store: Store):
    """At the cap, the request transitions to FAILED with completed_at set;
    the owning item then surfaces through unrecoverable_items with the error."""
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


# ---------------------------------------------------------------------------
# solver scores
# ---------------------------------------------------------------------------

def test_insert_score_references_two_responses(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.upsert_round(item_id=iid, round_n=1, candidate=_candidate())
    # Solver + judge response rows
    store.insert_requests([_request(iid, request_id="sol1"), _request(iid, request_id="jud1")])
    store.claim_pending(2)
    store.insert_response(request_id="sol1", model="m", text="weak attempt")
    store.insert_response(request_id="jud1", model="m", text="{}")

    score = SolverScore(solver="weak", attempt=0, raw_response="weak attempt", total=0.2,
                        per_criterion={"c1": 0.2}, failure_modes=["generic"])
    score_id = store.insert_score(item_id=iid, round_n=1, score=score,
                                  solver_response_id="sol1", judge_response_id="jud1")
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
            store.insert_score(item_id=iid, round_n=1, score=score,
                               solver_response_id="r0", judge_response_id="r1")
    out = store.scores_for_round(iid, 1)
    assert len(out) == 4
    assert {s.solver for s in out} == {"weak", "strong"}


# ---------------------------------------------------------------------------
# accepted + export
# ---------------------------------------------------------------------------

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
    b = store.insert_accepted(item_id=iid, round_n=1, payload={"x": 2})  # ignored
    assert a == b
    assert store.count_accepted("r1") == 1


# ---------------------------------------------------------------------------
# resume normalization (§4.2)
# ---------------------------------------------------------------------------

def test_resume_in_flight_local_to_pending(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)  # local: batch_id stays NULL
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["in_flight_to_pending"] == 1
    assert store.get_request("q1").status == REQ_PENDING


def test_resume_in_flight_batch_unchanged(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1, batch_id="batch-xyz")
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["in_flight_to_pending"] == 0
    assert store.get_request("q1").status == REQ_IN_FLIGHT
    assert store.get_request("q1").batch_id == "batch-xyz"


def test_resume_done_without_response_reverts_to_pending(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    # Manually put it in 'done' without a response row (simulate crash window).
    with store.tx() as cur:
        cur.execute("UPDATE requests SET status=? WHERE request_id='q1'", (REQ_DONE,))
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["done_to_pending"] == 1
    assert store.get_request("q1").status == REQ_PENDING


def test_resume_failed_under_cap_reverts_to_pending(store: Store):
    """A request that crashed in the FAILED state (e.g. process killed mid-write)
    is reset to PENDING on resume so the dispatcher can pick it up again."""
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)
    # Force the FAILED state directly: in-run, mark_request_failed would
    # requeue under-cap requests itself.
    with store.tx() as cur:
        cur.execute(
            "UPDATE requests SET status=?, failure_count=1 WHERE request_id='q1'",
            (REQ_FAILED,),
        )
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["failed_to_pending"] == 1
    assert store.get_request("q1").status == REQ_PENDING


def test_resume_failed_at_cap_stays_failed(store: Store):
    iid = store.insert_item(run_id="r1", source_id="s1", domain="qa", state="NEED_SCORES")
    store.insert_requests([_request(iid, request_id="q1")])
    store.claim_pending(1)
    for _ in range(3):
        store.mark_request_failed("q1", "err", max_failures=3)
    counts = store.normalize_for_resume(run_id="r1", max_request_failures=3)
    assert counts["failed_to_pending"] == 0
    assert store.get_request("q1").failure_count == 3
    assert store.get_request("q1").status == REQ_FAILED
    assert (iid, "err") in store.unrecoverable_items("r1", max_request_failures=3)
