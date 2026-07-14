"""Dispatcher behavior across local execution, failure, and resume."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from threading import Lock

import pytest

from autosynth.config import (
    AcceptanceConfig,
    DispatcherConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autosynth.dispatcher import Dispatcher
from autosynth.domain import GroundingItem
from autosynth.domains.qa_from_documents import QAFromDocuments
from autosynth.harness import DEFAULT_HARNESS
from autosynth.llm import LLMClient, register_mock
from autosynth.pipeline import State
from autosynth.store import Store

# fixtures + mock scenarios


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
    run_id = cfg.run_id
    assert run_id is not None
    domain = QAFromDocuments(source_dir=str(docs_dir))
    store.create_run(run_id, config=cfg.model_dump(mode="json"))
    grounding: dict[str, GroundingItem] = {}
    for item in domain.load_grounding():
        store.insert_item(
            run_id=run_id,
            source_id=item.source_id,
            domain=domain.name,
            state=State.PENDING.value,
            source_metadata=item.metadata,
        )
        grounding[item.source_id] = item
    disp = Dispatcher(
        store=store,
        llm=LLMClient(),
        domain=domain,
        cfg=cfg,
        run_id=run_id,
        harness=DEFAULT_HARNESS,
        grounding=grounding,
    )
    return disp, grounding


# Mock scenario: deterministic accept path.
def _disp_happy(role: str, messages):
    from autosynth.llm.mock import _canonical_role, _join_messages

    all_text = _join_messages(messages)
    canon = _canonical_role(role, all_text)
    if canon == "challenger":
        return json.dumps(
            {
                "payload": {"question": "What is documented?", "context": "synthetic"},
                "reference_output": "the documented fact",
                "rubric": [
                    {"id": "c1", "description": "names fact", "weight": 5},
                    {"id": "c2", "description": "cites", "weight": 3},
                ],
            }
        )
    if canon == "quality":
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if canon == "judge":
        if "SOLVER_RESPONSE: vague" in all_text:
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


# happy path


def test_dispatcher_happy_path_accepts_both_items(store, cfg, docs_dir):
    disp, _ = _seed_run(store, cfg, docs_dir)
    summary = disp.run()
    assert summary.accepted == 2
    assert summary.rejected == 0
    assert store.count_accepted(cfg.run_id) == 2


def test_dispatcher_writes_solver_scores(store, cfg, docs_dir):
    disp, _ = _seed_run(store, cfg, docs_dir)
    disp.run()
    all_items = list(store.conn.execute("SELECT item_id FROM items"))
    for row in all_items:
        scores = store.conn.execute(
            "SELECT solver, attempt FROM solver_scores ss "
            "JOIN rounds r ON ss.round_id = r.round_id WHERE r.item_id=?",
            (row["item_id"],),
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
        assert r["accepted"] == 1


def test_dispatcher_export_jsonl_after_run(store, cfg, docs_dir, tmp_path):
    disp, _ = _seed_run(store, cfg, docs_dir)
    disp.run()
    out = tmp_path / "out.jsonl"
    n = store.export_jsonl(cfg.run_id, out)
    assert n == 2
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["acceptance_rationale"] for r in records)
    assert all(r["gap"] > 0.2 for r in records)


# concurrency / claim_pending under load


def test_dispatcher_concurrent_fulfill_calls_each_request_once(store, cfg, docs_dir, monkeypatch):
    cfg.dispatcher.concurrency = 8
    cfg.loop.weak_samples = 4
    cfg.loop.strong_samples = 4
    disp, _ = _seed_run(store, cfg, docs_dir)

    calls: Counter[str] = Counter()
    calls_lock = Lock()
    complete = disp.llm.complete

    def counted_complete(request):
        with calls_lock:
            calls[request.request_id] += 1
        return complete(request)

    monkeypatch.setattr(disp.llm, "complete", counted_complete)
    disp.run()
    assert calls
    assert set(calls.values()) == {1}
    assert set(calls) == {row["request_id"] for row in store.conn.execute("SELECT request_id FROM requests")}


# unrecoverable failure path


def test_dispatcher_marks_item_rejected_after_failure_cap(store, cfg, docs_dir):
    cfg.challenger.provider_model = "mock/disp_failing"
    cfg.dispatcher.max_request_failures = 2
    disp, _ = _seed_run(store, cfg, docs_dir)
    disp.run()
    # Resume retries the failed requests and drives them to the cap.
    disp2 = Dispatcher(
        store=store,
        llm=LLMClient(),
        domain=disp.domain,
        cfg=cfg,
        run_id=cfg.run_id,
        harness=DEFAULT_HARNESS,
        grounding=disp.grounding,
    )
    disp2.run()
    summary = disp2._summarize()
    assert summary.rejected == 2
    assert summary.accepted == 0
    rows = store.conn.execute(
        "SELECT rejection_reasons FROM items WHERE rejection_reasons IS NOT NULL"
    ).fetchall()
    assert any("unrecoverable" in r["rejection_reasons"] for r in rows)


# budget abort


def test_dispatcher_aborts_when_budget_exceeded(store, cfg, docs_dir):
    cfg.budget_usd = 0.0001
    disp, _ = _seed_run(store, cfg, docs_dir)
    # Start above the configured budget.
    store.cost_so_far = lambda run_id: 1.0
    disp.run()
    run_row = store.get_run(cfg.run_id)
    assert run_row["status"] == "aborted"


# resume after kill


def test_dispatcher_resume_completes_partial_run(store, cfg, docs_dir):
    """Drive the dispatcher, kill it mid-flight, restart, finish."""
    disp, grounding = _seed_run(store, cfg, docs_dir)
    # Leave the first requests in flight to model a killed local worker.
    items_pending = store.items_pending_first_step(cfg.run_id)
    for row in items_pending:
        disp._advance_one(row)
    # Complete one request and leave the rest in flight.
    one = store.claim_pending(limit=1)
    if one:
        from autosynth.llm import LLMRequest

        req = one[0]
        request = LLMRequest(
            request_id=req.request_id,
            item_id=req.item_id,
            round_n=req.round_n,
            role=req.role,
            model_key=req.model_key,
            messages=req.messages,
            json_mode=req.json_mode,
            attempt=req.attempt,
        )
        resp = LLMClient().complete(request)
        store.insert_response(
            request_id=req.request_id, model=resp.model, text=resp.text, cost_usd=resp.cost_usd
        )
    disp2 = Dispatcher(
        store=store,
        llm=LLMClient(),
        domain=disp.domain,
        cfg=cfg,
        run_id=cfg.run_id,
        harness=DEFAULT_HARNESS,
        grounding=grounding,
    )
    summary = disp2.run()
    assert summary.accepted == 2


def test_hydration_reattaches_challenger_schema(store, cfg, docs_dir):
    """Rehydrated challenger requests must recover the domain payload schema."""
    from autosynth.dispatcher.hydration import row_to_llm_request
    from autosynth.domains.qa_from_documents import QAPayload
    from autosynth.llm.response_format import challenger_schema_for

    disp, _ = _seed_run(store, cfg, docs_dir)
    for row in store.items_pending_first_step(cfg.run_id):
        disp._advance_one(row)
    claimed = store.claim_pending(limit=10)
    challengers = [r for r in claimed if r.role == "challenger"]
    assert challengers
    for req_row in challengers:
        assert row_to_llm_request(req_row, disp.domain).response_schema is challenger_schema_for(QAPayload)
        # Batch transport intentionally uses plain JSON mode.
        assert row_to_llm_request(req_row).response_schema is None


def test_resume_after_watermark_reset_is_idempotent(store, cfg, docs_dir):
    """Re-delivered scores after migration must remain idempotent."""
    disp, grounding = _seed_run(store, cfg, docs_dir)
    assert disp.run().accepted == 2
    scores_before = store.conn.execute("SELECT COUNT(*) FROM solver_scores").fetchone()[0]
    assert scores_before > 0

    # Rewind to unfinalized scoring with every response deliverable again.
    with store.tx() as cur:
        cur.execute("DELETE FROM accepted")
        cur.execute("UPDATE rounds SET accepted = 0")
        cur.execute("UPDATE items SET state = 'NEED_SCORES', final_round = NULL, consumed_seq = 0")

    disp2 = Dispatcher(
        store=store,
        llm=LLMClient(),
        domain=disp.domain,
        cfg=cfg,
        run_id=cfg.run_id,
        harness=DEFAULT_HARNESS,
        grounding=grounding,
    )
    assert disp2.run().accepted == 2
    assert store.conn.execute("SELECT COUNT(*) FROM solver_scores").fetchone()[0] == scores_before
    assert store.conn.execute("SELECT COUNT(*) FROM accepted").fetchone()[0] == 2


def test_dispatcher_dead_letters_crashing_step(store, cfg, docs_dir, monkeypatch):
    # A deterministic step error must terminate instead of livelocking.
    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic step crash")

    monkeypatch.setattr("autosynth.dispatcher.core.step", _boom)
    disp, grounding = _seed_run(store, cfg, docs_dir)
    summary = disp.run()
    assert summary.rejected == len(grounding)
    assert summary.accepted == 0
    rows = store.conn.execute("SELECT rejection_reasons FROM items WHERE state='REJECTED'").fetchall()
    assert rows and all("step crashed" in (row[0] or "") for row in rows)
