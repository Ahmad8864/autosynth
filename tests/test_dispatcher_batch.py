"""Batch provider and dispatcher integration."""

from __future__ import annotations

import json
from pathlib import Path

import litellm
import pytest

from autosynth.config import (
    AcceptanceConfig,
    DispatcherConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autosynth.dispatcher import (
    AnthropicBatchProvider,
    BatchHandle,
    BatchResult,
    Dispatcher,
    LiteLLMBatchProvider,
    MockBatchProvider,
    make_fulfill_batch,
    poll_outstanding_batches,
)
from autosynth.dispatcher.batch import _anthropic_result_to_result, _to_anthropic_request
from autosynth.domain import GroundingItem
from autosynth.domains.qa_from_documents import QAFromDocuments
from autosynth.harness import DEFAULT_HARNESS
from autosynth.llm import LLMClient, LLMRequest
from autosynth.pipeline import State
from autosynth.runner import Runner
from autosynth.store import Store


class _Obj:
    """Minimal attribute bag standing in for LiteLLM's batch/file objects."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


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
        store.insert_item(
            run_id="r1",
            source_id=item.source_id,
            domain=domain.name,
            state=State.PENDING.value,
            source_metadata=item.metadata,
        )
        grounding[item.source_id] = item
    return Dispatcher(
        store=store,
        llm=LLMClient(),
        domain=domain,
        cfg=cfg,
        run_id="r1",
        harness=DEFAULT_HARNESS,
        grounding=grounding,
        fulfill=make_fulfill_batch(provider),
        poll_in_flight=lambda d: poll_outstanding_batches(provider, d),
    )


# MockBatchProvider sanity


def test_mock_batch_provider_submit_complete_fetch():
    p = MockBatchProvider(ready_after_polls=2)
    req = LLMRequest(
        request_id="x",
        item_id="i",
        round_n=1,
        role="weak",
        model_key="mock/scripted",
        messages=[{"role": "user", "content": "x"}],
    )
    handle = p.submit([req])
    assert isinstance(handle, BatchHandle)
    assert not p.is_complete(handle.batch_id)
    assert p.is_complete(handle.batch_id)
    results = list(p.fetch(handle.batch_id))
    assert len(results) == 1
    assert isinstance(results[0], BatchResult)
    assert results[0].response is not None


# End-to-end through Dispatcher


def test_dispatcher_drives_run_via_batch_provider(store, docs_dir, tmp_path):
    provider = MockBatchProvider(ready_after_polls=1)
    disp = _make_dispatcher(store, docs_dir, tmp_path, provider)
    summary = disp.run()
    # The default mock accepts deterministically on the first round.
    assert summary.accepted == 1 and summary.rejected == 0


def test_batch_id_is_tagged_then_cleared(store, docs_dir, tmp_path):
    tagged_counts: list[int] = []

    class InspectingProvider(MockBatchProvider):
        def is_complete(self, batch_id):
            tagged_counts.append(
                store.conn.execute("SELECT COUNT(*) FROM requests WHERE batch_id=?", (batch_id,)).fetchone()[
                    0
                ]
            )
            return super().is_complete(batch_id)

    provider = InspectingProvider(ready_after_polls=1)
    disp = _make_dispatcher(store, docs_dir, tmp_path, provider)
    disp.run()
    assert tagged_counts and all(count > 0 for count in tagged_counts)
    rows = store.conn.execute("SELECT COUNT(*) FROM requests WHERE batch_id IS NOT NULL").fetchone()
    assert rows[0] == 0


def test_batch_provider_failure_marks_request_failed(store, docs_dir, tmp_path):
    class FailingProvider:
        provider_name = "fail"

        def submit(self, requests):
            raise RuntimeError("provider down")

        def is_complete(self, batch_id):
            return False

        def fetch(self, batch_id):
            return []

    disp = _make_dispatcher(store, docs_dir, tmp_path, FailingProvider())
    disp.run()
    rows = store.conn.execute("SELECT status, failure_count FROM requests").fetchall()
    assert any(r["failure_count"] >= 1 for r in rows)


def test_poll_reconciles_requests_missing_from_results(store, docs_dir, tmp_path):
    """Missing batch results must not leave tagged requests in flight."""

    class EmptyFetchProvider:
        provider_name = "empty"

        def __init__(self):
            self._n = 0

        def submit(self, requests):
            self._n += 1
            return BatchHandle(
                batch_id=f"b{self._n}",
                provider="empty",
                request_ids=tuple(r.request_id for r in requests),
            )

        def is_complete(self, batch_id):
            return True

        def fetch(self, batch_id):
            return []

    disp = _make_dispatcher(store, docs_dir, tmp_path, EmptyFetchProvider())
    summary = disp.run()
    assert summary.accepted == 0 and summary.rejected == 1
    errors = [
        r["last_error"]
        for r in store.conn.execute("SELECT last_error FROM requests WHERE last_error IS NOT NULL").fetchall()
    ]
    assert any("without a result" in (e or "") for e in errors)


def test_poll_skips_untagged_results(store, docs_dir, tmp_path):
    """Ignore results whose request IDs were never tagged."""

    class BogusResultProvider:
        provider_name = "bogus"

        def __init__(self):
            self._n = 0

        def submit(self, requests):
            self._n += 1
            return BatchHandle(
                batch_id=f"b{self._n}",
                provider="bogus",
                request_ids=tuple(r.request_id for r in requests),
            )

        def is_complete(self, batch_id):
            return True

        def fetch(self, batch_id):
            # The real tagged request is recovered by the unseen-request sweep.
            return [BatchResult(request_id="", response=None, error="boom")]

    disp = _make_dispatcher(store, docs_dir, tmp_path, BogusResultProvider())
    summary = disp.run()
    assert summary.accepted == 0 and summary.rejected == 1


# LiteLLMBatchProvider (real provider over LiteLLM's batch API, mocked)


def _req(request_id="r1", *, role="quality", json_mode=False, temperature=None):
    return LLMRequest(
        request_id=request_id,
        item_id="i",
        round_n=1,
        role=role,
        model_key="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "give me json"}],
        json_mode=json_mode,
        temperature=temperature,
    )


def test_litellm_provider_submit_serializes_openai_batch(monkeypatch):
    captured: dict = {}

    def fake_create_file(*, file, purpose, custom_llm_provider, **kw):
        _name, buf = file
        captured["bytes"] = buf.read()
        captured["purpose"] = purpose
        captured["provider"] = custom_llm_provider
        return _Obj(id="file-1")

    def fake_create_batch(*, completion_window, endpoint, input_file_id, custom_llm_provider, **kw):
        captured["endpoint"] = endpoint
        captured["input_file_id"] = input_file_id
        return _Obj(id="batch-1", status="validating")

    monkeypatch.setattr(litellm, "create_file", fake_create_file)
    monkeypatch.setattr(litellm, "create_batch", fake_create_batch)

    handle = LiteLLMBatchProvider().submit([_req(json_mode=True, temperature=0.5)])

    assert handle.batch_id == "batch-1" and handle.request_ids == ("r1",)
    assert captured["provider"] == "openai" and captured["purpose"] == "batch"
    assert captured["endpoint"] == "/v1/chat/completions" and captured["input_file_id"] == "file-1"
    line = json.loads(captured["bytes"].decode().strip())
    assert line["custom_id"] == "r1"
    assert line["body"]["model"] == "gpt-4o-mini"
    assert line["body"]["response_format"] == {"type": "json_object"}
    assert line["body"]["temperature"] == 0.5


def test_litellm_provider_polls_until_terminal(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "retrieve_batch",
        lambda batch_id, custom_llm_provider="openai", **kw: _Obj(status="in_progress"),
    )
    assert LiteLLMBatchProvider().is_complete("b") is False
    monkeypatch.setattr(
        litellm,
        "retrieve_batch",
        lambda batch_id, custom_llm_provider="openai", **kw: _Obj(status="completed"),
    )
    assert LiteLLMBatchProvider().is_complete("b") is True


def test_litellm_provider_fetch_parses_output(monkeypatch):
    out_line = json.dumps(
        {
            "custom_id": "r1",
            "response": {
                "status_code": 200,
                "body": {
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            },
            "error": None,
        }
    )
    monkeypatch.setattr(
        litellm,
        "retrieve_batch",
        lambda batch_id, custom_llm_provider="openai", **kw: _Obj(
            id=batch_id, status="completed", output_file_id="out-1", error_file_id=None
        ),
    )
    monkeypatch.setattr(
        litellm,
        "file_content",
        lambda file_id, custom_llm_provider="openai", **kw: _Obj(content=(out_line + "\n").encode()),
    )
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.001)

    results = list(LiteLLMBatchProvider().fetch("batch-1"))

    assert len(results) == 1
    r = results[0]
    assert r.request_id == "r1" and r.error is None
    assert r.response is not None
    assert r.response.text == "hello"
    assert r.response.prompt_tokens == 3 and r.response.completion_tokens == 2
    assert r.response.cost_usd == 0.001


def test_litellm_provider_fetch_surfaces_error_file(monkeypatch):
    err_line = json.dumps({"custom_id": "r2", "response": None, "error": {"code": "x", "message": "boom"}})
    monkeypatch.setattr(
        litellm,
        "retrieve_batch",
        lambda batch_id, custom_llm_provider="openai", **kw: _Obj(
            id=batch_id, status="failed", output_file_id=None, error_file_id="err-1"
        ),
    )
    monkeypatch.setattr(
        litellm,
        "file_content",
        lambda file_id, custom_llm_provider="openai", **kw: _Obj(content=(err_line + "\n").encode()),
    )

    results = list(LiteLLMBatchProvider().fetch("batch-1"))

    assert len(results) == 1
    assert results[0].request_id == "r2" and results[0].response is None
    assert results[0].error is not None and "boom" in results[0].error


# Runner wiring (dispatcher.mode == "batch")


def test_runner_batch_mode_with_mock_provider(docs_dir, tmp_path):
    cfg = RunConfig(
        run_id="rb",
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
        dispatcher=DispatcherConfig(
            concurrency=4, items_per_advance=5, poll_interval_s=0.0, mode="batch", batch_provider="mock"
        ),
    )
    summary = Runner(cfg).run()
    assert summary.accepted == 1 and summary.rejected == 0


# AnthropicBatchProvider (native Message Batches API, HTTP seam stubbed)


def _areq(request_id="r1", *, temperature=None, max_tokens=None, system=False):
    messages = [{"role": "user", "content": "hi"}]
    if system:
        messages = [{"role": "system", "content": "be terse"}, *messages]
    return LLMRequest(
        request_id=request_id,
        item_id="i",
        round_n=1,
        role="quality",
        model_key="anthropic/claude-haiku-4-5",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_anthropic_request_translation_hoists_system_and_defaults_max_tokens():
    out = _to_anthropic_request(_areq(system=True, temperature=0.3), max_tokens_default=1024)
    assert out["custom_id"] == "r1"
    params = out["params"]
    assert params["model"] == "claude-haiku-4-5"
    assert params["max_tokens"] == 1024
    assert params["system"] == "be terse"
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    assert params["temperature"] == 0.3


def test_anthropic_request_keeps_explicit_max_tokens():
    out = _to_anthropic_request(_areq(max_tokens=42), max_tokens_default=1024)
    assert out["params"]["max_tokens"] == 42
    assert "system" not in out["params"]


def test_anthropic_result_parsing_succeeded(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.002)
    line = {
        "custom_id": "r1",
        "result": {
            "type": "succeeded",
            "message": {
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        },
    }
    r = _anthropic_result_to_result(line)
    assert r.error is None and r.response is not None
    assert r.response.text == "hello"
    assert r.response.prompt_tokens == 5 and r.response.completion_tokens == 2
    assert r.response.cost_usd == 0.002


def test_anthropic_result_parsing_errored():
    line = {
        "custom_id": "r2",
        "result": {"type": "errored", "error": {"type": "invalid_request", "message": "bad"}},
    }
    r = _anthropic_result_to_result(line)
    assert r.response is None and r.error is not None and "bad" in r.error


def test_anthropic_provider_submit_poll_fetch(monkeypatch):
    p = AnthropicBatchProvider(api_key="k")
    captured: dict = {}
    state = {"status": "in_progress"}

    def fake_create(payload):
        captured["payload"] = payload
        return {"id": "msgbatch_1", "processing_status": "in_progress"}

    def fake_retrieve(batch_id):
        ended = state["status"] == "ended"
        return {
            "id": batch_id,
            "processing_status": state["status"],
            "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"
            if ended
            else None,
        }

    def fake_download(results_url):
        line = {
            "custom_id": "r1",
            "result": {
                "type": "succeeded",
                "message": {
                    "model": "claude-haiku-4-5",
                    "content": [{"type": "text", "text": "yo"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
        }
        return json.dumps(line) + "\n"

    monkeypatch.setattr(p, "_create_batch", fake_create)
    monkeypatch.setattr(p, "_retrieve_batch", fake_retrieve)
    monkeypatch.setattr(p, "_download_results", fake_download)

    handle = p.submit([_areq()])
    assert handle.batch_id == "msgbatch_1"
    assert captured["payload"]["requests"][0]["custom_id"] == "r1"
    assert p.is_complete("msgbatch_1") is False
    state["status"] = "ended"
    assert p.is_complete("msgbatch_1") is True
    results = list(p.fetch("msgbatch_1"))
    assert len(results) == 1 and results[0].response is not None
    assert results[0].response.text == "yo"


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicBatchProvider()._http("GET", "https://api.anthropic.com/v1/messages/batches/x")


def test_runner_batch_provider_selection(docs_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def cfg(provider):
        return RunConfig(
            run_id=f"r-{provider}",
            output_dir=str(tmp_path),
            max_examples=1,
            domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(docs_dir)}),
            challenger=ModelConfig(provider_model="mock/scripted"),
            weak_solver=ModelConfig(provider_model="mock/scripted"),
            strong_solver=ModelConfig(provider_model="mock/scripted"),
            judge=ModelConfig(provider_model="mock/scripted"),
            dispatcher=DispatcherConfig(mode="batch", batch_provider=provider),
        )

    assert isinstance(Runner(cfg("mock"))._batch_provider(), MockBatchProvider)
    assert isinstance(Runner(cfg("anthropic"))._batch_provider(), AnthropicBatchProvider)
    assert isinstance(Runner(cfg("openai"))._batch_provider(), LiteLLMBatchProvider)
