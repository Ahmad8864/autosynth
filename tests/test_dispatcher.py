"""Tests for the Dispatcher.

Covers: end-to-end happy-path drive via local fulfill, claim_pending
concurrency under load, budget abort, resume normalization integration,
and the unrecoverable-failure path.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from autodata.config import (
    AcceptanceConfig,
    DispatcherConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autodata.dispatcher import Dispatcher, fulfill_local
from autodata.domain import GroundingItem
from autodata.domains.qa_from_documents import QAFromDocuments
from autodata.harness import DEFAULT_HARNESS
from autodata.llm import LLMClient, register_mock
from autodata.pipeline import State
from autodata.store import REQ_DONE, REQ_FAILED, REQ_PENDING, Store


# ---------------------------------------------------------------------------
# fixtures + mock scenarios
# ---------------------------------------------------------------------------

@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("Document A contains specific facts about topic A.")
    (d / "b.md").write_text("Document B contains specific facts about topic B.")
    return d


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "run.db")


@pytest.fixture
def cfg(docs_dir, tmp_path) -> RunConfig:
    return RunConfig(
        run_id="r1",
        output_dir=str(tmp_path),
        max_examples=2,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(docs_dir)}),
        loop=LoopConfig(max_rounds=2, weak_samples=2, strong_samples=2),
        acceptance=AcceptanceConfig(forbid_weak_zero=False),
        orchestrator=ModelConfig(provider_model="mock/disp_happy"),
        challenger=ModelConfig(provider_model="mock/disp_happy"),
        weak_solver=ModelConfig(provider_model="mock/disp_happy"),
        strong_solver=ModelConfig(provider_model="mock/disp_happy"),
        judge=ModelConfig(provider_model="mock/disp_happy"),
        dispatcher=DispatcherConfig(concurrency=4, items_per_advance=10, poll_interval_s=0.0),
    )


def _seed_run(store: Store, cfg: RunConfig, docs_dir: Path) -> tuple[Dispatcher, dict[str, GroundingItem]]:
    domain = QAFromDocuments(source_dir=str(docs_dir))
    store.create_run(cfg.run_id, config=cfg.model_dump(mode="json"))
    grounding: dict[str, GroundingItem] = {}
    for item in domain.load_grounding():
        store.insert_item(run_id=cfg.run_id, source_id=item.source_id,
                          domain=domain.name, state=State.PENDING.value,
                          source_metadata=item.metadata)
        grounding[item.source_id] = item
    disp = Dispatcher(
        store=store, llm=LLMClient(), domain=domain, cfg=cfg,
        run_id=cfg.run_id, harness=DEFAULT_HARNESS, grounding=grounding,
    )
    return disp, grounding


# Mock scenario: deterministic accept path.
def _disp_happy(role: str, messages):
    from autodata.llm import _canonical_role, _join_messages
    all_text = _join_messages(messages)
    canon = _canonical_role(role, all_text)
    if canon == "challenger":
        return json.dumps({
            "payload": {"question": "What is documented?", "context": "synthetic"},
            "reference_output": "the documented fact",
            "rubric": [{"id": "c1", "description": "names fact", "weight": 5},
                       {"id": "c2", "description": "cites", "weight": 3}],
        })
    if canon == "quality":
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if canon == "judge":
        if "[solver=weak]" in all_text:
            return json.dumps({"per_criterion": {"c1": 0.2, "c2": 0.1}, "total": 0.16})
        return json.dumps({"per_criterion": {"c1": 0.9, "c2": 0.85}, "total": 0.88})
    if canon == "weak":
        return "vague"
    if canon == "strong":
        return "specific, fact-grounded answer"
    if canon == "reflector":
        return json.dumps({"feedback": ["try harder"], "new_angle": ""})
    return "{}"


register_mock("disp_happy", _disp_happy)


# Mock scenario: every solver/judge request raises (for unrecoverable test).
def _disp_failing(role: str, messages):
    raise RuntimeError("simulated provider outage")


register_mock("disp_failing", _disp_failing)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_dispatcher_happy_path_accepts_both_items(store, cfg, docs_dir):
    disp, _ = _seed_run(store, cfg, docs_dir)
    summary = disp.run()
    assert summary.accepted == 2
    assert summary.rejected == 0
    assert store.count_accepted(cfg.run_id) == 2


def test_dispatcher_writes_solver_scores(store, cfg, docs_dir):
    disp, _ = _seed_run(store, cfg, docs_dir)
    disp.run()
    # Each accepted item has weak_samples + strong_samples = 4 score rows.
    all_items = list(store.conn.execute("SELECT item_id FROM items"))
    for row in all_items:
        scores = store.conn.execute(
            "SELECT solver, attempt FROM solver_scores ss "
            "JOIN rounds r ON ss.round_id = r.round_id WHERE r.item_id=?", (row["item_id"],)
        ).fetchall()
        assert len(scores) == 4


def test_dispatcher_writes_round_blobs(store, cfg, docs_dir):
    disp, _ = _seed_run(store, cfg, docs_dir)
    disp.run()
    rows = store.conn.execute(
        "SELECT candidate_blob, quality_blob, eval_blob, accepted FROM rounds"
    ).fetchall()
    assert rows
    for r in rows:
        assert r["candidate_blob"] is not None
        assert r["quality_blob"] is not None
        assert r["eval_blob"] is not None
        assert r["accepted"] == 1   # all rounds accepted under disp_happy


def test_dispatcher_export_jsonl_after_run(store, cfg, docs_dir, tmp_path):
    disp, _ = _seed_run(store, cfg, docs_dir)
    disp.run()
    out = tmp_path / "out.jsonl"
    n = store.export_jsonl(cfg.run_id, out)
    assert n == 2
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["acceptance_rationale"] for r in records)
    assert all(r["gap"] > 0.2 for r in records)


# ---------------------------------------------------------------------------
# concurrency / claim_pending under load
# ---------------------------------------------------------------------------

def test_dispatcher_concurrent_fulfill_no_duplicate_responses(store, cfg, docs_dir):
    # Use a higher concurrency to stress claim_pending.
    cfg.dispatcher.concurrency = 8
    cfg.loop.weak_samples = 4
    cfg.loop.strong_samples = 4
    disp, _ = _seed_run(store, cfg, docs_dir)
    disp.run()
    # Every request has exactly one response.
    rows = store.conn.execute(
        "SELECT request_id, COUNT(*) AS n FROM responses GROUP BY request_id"
    ).fetchall()
    assert all(r["n"] == 1 for r in rows)


# ---------------------------------------------------------------------------
# unrecoverable failure path
# ---------------------------------------------------------------------------

def test_dispatcher_marks_item_rejected_after_failure_cap(store, cfg, docs_dir):
    cfg.challenger.provider_model = "mock/disp_failing"
    cfg.dispatcher.max_request_failures = 2
    disp, _ = _seed_run(store, cfg, docs_dir)
    # Run twice (so each request hits the failure cap)
    disp.run()
    # After the first run, requests have failure_count=1, status='failed'.
    # We restart the dispatcher; resume normalization reverts failed→pending.
    # The second run drives failure_count to 2 → at cap.
    disp2 = Dispatcher(
        store=store, llm=LLMClient(), domain=disp.domain, cfg=cfg,
        run_id=cfg.run_id, harness=DEFAULT_HARNESS, grounding=disp.grounding,
    )
    disp2.run()
    summary = disp2._summarize()
    assert summary.rejected == 2
    assert summary.accepted == 0
    # Reasons should mention unrecoverable
    rows = store.conn.execute(
        "SELECT rejection_reasons FROM items WHERE rejection_reasons IS NOT NULL"
    ).fetchall()
    assert any("unrecoverable" in r["rejection_reasons"] for r in rows)


# ---------------------------------------------------------------------------
# budget abort
# ---------------------------------------------------------------------------

def test_dispatcher_aborts_when_budget_exceeded(store, cfg, docs_dir):
    cfg.budget_usd = 0.0001
    disp, _ = _seed_run(store, cfg, docs_dir)
    # Pretend we've already spent 1.0 USD; first budget check aborts.
    store.cost_so_far = lambda run_id: 1.0
    disp.run()
    run_row = store.get_run(cfg.run_id)
    assert run_row["status"] == "aborted"


# ---------------------------------------------------------------------------
# resume after kill
# ---------------------------------------------------------------------------

def test_dispatcher_resume_completes_partial_run(store, cfg, docs_dir):
    """Drive the dispatcher, kill it mid-flight, restart, finish."""
    disp, grounding = _seed_run(store, cfg, docs_dir)
    # Run one advancement step's worth, then stop.
    # The simplest way to "interrupt" is to set the stop flag after the first
    # batch lands. We instead emulate kill by partially populating the store:
    # call advance_one to emit challenger requests, leave them in_flight, then
    # restart.
    items_pending = store.items_pending_first_step(cfg.run_id)
    for row in items_pending:
        disp._advance_one(row)
    # Now claim and process exactly one request, then simulate a crash by
    # leaving the rest in 'in_flight'.
    one = store.claim_pending(limit=1)
    if one:
        # Mark it done with a fake response so we have at least partial state.
        from autodata.llm import LLMRequest, Response
        req = one[0]
        request = LLMRequest(
            request_id=req.request_id, item_id=req.item_id, round_n=req.round_n,
            role=req.role, model_key=req.model_key, messages=req.messages,
            json_mode=req.json_mode, attempt=req.attempt,
        )
        resp = LLMClient().complete(request)
        store.insert_response(request_id=req.request_id, model=resp.model,
                              text=resp.text, cost_usd=resp.cost_usd)
    # Restart from scratch.
    disp2 = Dispatcher(
        store=store, llm=LLMClient(), domain=disp.domain, cfg=cfg,
        run_id=cfg.run_id, harness=DEFAULT_HARNESS, grounding=grounding,
    )
    summary = disp2.run()
    assert summary.accepted == 2
