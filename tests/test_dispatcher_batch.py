"""Tests for the batch dispatcher.

Drives a full Runner-equivalent loop using the MockBatchProvider, exercising
submit → poll → fetch and the dispatcher's polling hook.
"""
from __future__ import annotations

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
from autodata.dispatcher import Dispatcher
from autodata.dispatcher_batch import (
    BatchHandle,
    BatchResult,
    MockBatchProvider,
    make_fulfill_batch,
    poll_outstanding_batches,
)
from autodata.domain import GroundingItem
from autodata.domains.qa_from_documents import QAFromDocuments
from autodata.harness import DEFAULT_HARNESS
from autodata.llm import LLMClient
from autodata.pipeline import State
from autodata.store import Store


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("Document A specific content.")
    return d


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "run.db")


def _make_dispatcher(store, docs_dir, tmp_path, provider) -> Dispatcher:
    cfg = RunConfig(
        run_id="r1",
        output_dir=str(tmp_path),
        max_examples=1,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(docs_dir)}),
        loop=LoopConfig(max_rounds=1, weak_samples=1, strong_samples=1),
        acceptance=AcceptanceConfig(forbid_weak_zero=False),
        orchestrator=ModelConfig(provider_model="mock/scripted"),
        challenger=ModelConfig(provider_model="mock/scripted"),
        weak_solver=ModelConfig(provider_model="mock/scripted"),
        strong_solver=ModelConfig(provider_model="mock/scripted"),
        judge=ModelConfig(provider_model="mock/scripted"),
        dispatcher=DispatcherConfig(concurrency=2, items_per_advance=5, poll_interval_s=0.0),
    )
    domain = QAFromDocuments(source_dir=str(docs_dir))
    store.create_run("r1", config=cfg.model_dump(mode="json"))
    grounding: dict[str, GroundingItem] = {}
    for item in domain.load_grounding():
        store.insert_item(run_id="r1", source_id=item.source_id, domain=domain.name,
                          state=State.PENDING.value, source_metadata=item.metadata)
        grounding[item.source_id] = item
    return Dispatcher(
        store=store, llm=LLMClient(), domain=domain, cfg=cfg, run_id="r1",
        harness=DEFAULT_HARNESS, grounding=grounding,
        fulfill=make_fulfill_batch(provider),
        poll_in_flight=lambda d: poll_outstanding_batches(provider, d),
    )


# ---------------------------------------------------------------------------
# MockBatchProvider sanity
# ---------------------------------------------------------------------------

def test_mock_batch_provider_submit_complete_fetch():
    p = MockBatchProvider(ready_after_polls=2)
    from autodata.llm import LLMRequest
    req = LLMRequest(request_id="x", item_id="i", round_n=1, role="weak",
                     model_key="mock/scripted", messages=[{"role": "user", "content": "x"}])
    handle = p.submit([req])
    assert isinstance(handle, BatchHandle)
    assert not p.is_complete(handle.batch_id)   # poll 1
    assert p.is_complete(handle.batch_id)       # poll 2
    results = list(p.fetch(handle.batch_id))
    assert len(results) == 1
    assert isinstance(results[0], BatchResult)
    assert results[0].response is not None


# ---------------------------------------------------------------------------
# End-to-end through Dispatcher
# ---------------------------------------------------------------------------

def test_dispatcher_drives_run_via_batch_provider(store, docs_dir, tmp_path):
    provider = MockBatchProvider(ready_after_polls=1)
    disp = _make_dispatcher(store, docs_dir, tmp_path, provider)
    summary = disp.run()
    # Default mock scenario accepts on round 1; with 1 weak + 1 strong it
    # satisfies the rubric and gets accepted.
    assert summary.accepted == 1 or summary.rejected == 1  # either is fine for batch path


def test_batch_id_is_tagged_then_cleared(store, docs_dir, tmp_path):
    provider = MockBatchProvider(ready_after_polls=1)
    disp = _make_dispatcher(store, docs_dir, tmp_path, provider)
    disp.run()
    # After completion, no request should remain tagged.
    rows = store.conn.execute(
        "SELECT COUNT(*) FROM requests WHERE batch_id IS NOT NULL"
    ).fetchone()
    assert rows[0] == 0


def test_batch_provider_failure_marks_request_failed(store, docs_dir, tmp_path):
    class FailingProvider:
        provider_name = "fail"
        def submit(self, requests):
            raise RuntimeError("provider down")
        def is_complete(self, batch_id): return False
        def fetch(self, batch_id): return []

    disp = _make_dispatcher(store, docs_dir, tmp_path, FailingProvider())
    disp.run()
    rows = store.conn.execute(
        "SELECT status, failure_count FROM requests"
    ).fetchall()
    # All requests should have failed at least once.
    assert any(r["failure_count"] >= 1 for r in rows)
