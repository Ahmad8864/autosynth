"""Batch dispatcher: submit pending requests in chunks to provider batch APIs.

Implements the ``fulfill`` strategy for ``Dispatcher`` that uses provider
batch endpoints instead of streaming HTTP. The pipeline state machine is
unchanged across the batch SLA; resume works because every request carries a
deterministic ``request_id`` and the ``batch_id`` column tags in-flight batch
submissions.

The module defines a small :class:`BatchProvider` protocol, an in-process
:class:`MockBatchProvider` for tests and demos, and :class:`LiteLLMBatchProvider`
— the real OpenAI-style provider (upload file → create batch → poll → download)
over LiteLLM, covering whatever LiteLLM's batch API supports.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from autosynth.dispatcher.hydration import row_to_llm_request
from autosynth.llm import LLMClient, LLMRequest, Response
from autosynth.store import RequestRow, Store

__all__ = [
    "BatchProvider",
    "BatchHandle",
    "BatchResult",
    "MockBatchProvider",
    "LiteLLMBatchProvider",
    "AnthropicBatchProvider",
    "make_fulfill_batch",
    "poll_outstanding_batches",
]

# OpenAI-style batch statuses we treat as terminal — once a batch reaches one,
# polling stops and we fetch (or surface failure for) its requests.
_BATCH_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})


@dataclass(frozen=True)
class BatchHandle:
    """A provider-side identifier for a submitted batch."""

    batch_id: str
    provider: str
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class BatchResult:
    """One completed (request_id, Response) pair from a batch."""

    request_id: str
    response: Response | None
    error: str | None = None


class BatchProvider(Protocol):
    """Provider-specific batch submission + polling."""

    provider_name: str

    @abstractmethod
    def submit(self, requests: list[LLMRequest]) -> BatchHandle: ...

    @abstractmethod
    def is_complete(self, batch_id: str) -> bool: ...

    @abstractmethod
    def fetch(self, batch_id: str) -> Iterable[BatchResult]: ...


class MockBatchProvider:
    """In-process batch provider used by tests and demos.

    ``submit`` synchronously calls ``LLMClient.complete`` per request and
    stashes the results. They become visible to ``fetch`` only after the
    caller has polled ``is_complete`` ``ready_after_polls`` times.
    """

    provider_name = "mock"

    def __init__(self, llm: LLMClient | None = None, ready_after_polls: int = 1):
        self.llm = llm or LLMClient()
        self.ready_after_polls = ready_after_polls
        self._batches: dict[str, dict] = {}

    def submit(self, requests: list[LLMRequest]) -> BatchHandle:
        batch_id = f"mock-batch-{len(self._batches) + 1}"
        results: list[BatchResult] = []
        for req in requests:
            try:
                resp = self.llm.complete(req)
                results.append(BatchResult(request_id=req.request_id, response=resp))
            except Exception as e:
                results.append(BatchResult(request_id=req.request_id, response=None, error=str(e)))
        self._batches[batch_id] = {"results": results, "polls": 0}
        return BatchHandle(
            batch_id=batch_id,
            provider=self.provider_name,
            request_ids=tuple(r.request_id for r in requests),
        )

    def is_complete(self, batch_id: str) -> bool:
        b = self._batches.get(batch_id)
        if b is None:
            return False
        b["polls"] += 1
        return b["polls"] >= self.ready_after_polls

    def fetch(self, batch_id: str) -> Iterable[BatchResult]:
        # Pop on fetch so long-running runs don't accumulate batch state forever.
        b = self._batches.pop(batch_id, None)
        return b["results"] if b else []


class LiteLLMBatchProvider:
    """Real batch provider over LiteLLM's OpenAI-style batch API.

    Covers any provider whose batch flow LiteLLM models as
    upload-file → create-batch → poll → download-results (OpenAI, Azure,
    Vertex, Bedrock, vLLM). Each :class:`LLMRequest` becomes one line of an
    OpenAI ``/v1/chat/completions`` batch input file, keyed by ``custom_id``
    so results map back to the originating request.

    Stateless: the configured ``provider`` is the LiteLLM ``custom_llm_provider``
    for every call, so a kill/resume mid-batch just re-polls by ``batch_id``. A
    run uses one batch provider (``dispatcher.batch_provider``).

    JSON-mode requests are submitted with plain ``{"type": "json_object"}``
    rather than a strict response schema — the structured-output path isn't
    wired through the batch file. The pipeline parses either shape.
    """

    provider_name = "litellm"

    def __init__(self, *, provider: str = "openai", completion_window: str = "24h"):
        self.provider = provider
        self.completion_window = completion_window

    def submit(self, requests: list[LLMRequest]) -> BatchHandle:
        litellm = _load_litellm()
        data = ("\n".join(_to_batch_line(r) for r in requests) + "\n").encode("utf-8")
        file_obj = litellm.create_file(
            file=("autosynth_batch.jsonl", io.BytesIO(data)),
            purpose="batch",
            custom_llm_provider=self.provider,
        )
        batch = litellm.create_batch(
            completion_window=self.completion_window,
            endpoint="/v1/chat/completions",
            input_file_id=file_obj.id,
            custom_llm_provider=self.provider,
        )
        return BatchHandle(
            batch_id=batch.id,
            provider=self.provider,
            request_ids=tuple(r.request_id for r in requests),
        )

    def is_complete(self, batch_id: str) -> bool:
        litellm = _load_litellm()
        batch = litellm.retrieve_batch(batch_id, custom_llm_provider=self.provider)
        return getattr(batch, "status", None) in _BATCH_TERMINAL

    def fetch(self, batch_id: str) -> Iterable[BatchResult]:
        litellm = _load_litellm()
        batch = litellm.retrieve_batch(batch_id, custom_llm_provider=self.provider)
        results: list[BatchResult] = []
        for file_id in (getattr(batch, "output_file_id", None), getattr(batch, "error_file_id", None)):
            if not file_id:
                continue
            for line in _read_jsonl(litellm.file_content(file_id, custom_llm_provider=self.provider)):
                results.append(_line_to_result(line))
        return results


def _load_litellm() -> Any:
    """Lazy import: litellm is a ~2s import, kept off the dispatcher's import path
    (loaded once on first batch call, then cached)."""
    import litellm

    return litellm


def _to_batch_line(req: LLMRequest) -> str:
    """Serialize one request as an OpenAI batch input line (keyed by request_id)."""
    body: dict[str, Any] = {"model": _model_name(req.model_key), "messages": req.messages}
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.json_mode:
        body["response_format"] = {"type": "json_object"}
    return json.dumps(
        {"custom_id": req.request_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
    )


def _model_name(model_key: str) -> str:
    """Strip the LiteLLM provider prefix for the provider-native batch body (``openai/x`` → ``x``)."""
    return model_key.split("/", 1)[1] if "/" in model_key else model_key


def _read_jsonl(content: Any) -> Iterable[dict]:
    """Decode a LiteLLM file-content object into parsed JSONL records."""
    raw = getattr(content, "content", None)
    if raw is None:
        raw = getattr(content, "text", "")
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _line_to_result(line: dict) -> BatchResult:
    """Map one OpenAI batch output/error record to a :class:`BatchResult`."""
    request_id = line.get("custom_id", "")
    err = line.get("error")
    resp = line.get("response")
    if err or resp is None:
        return BatchResult(
            request_id=request_id, response=None, error=json.dumps(err) if err else "no response"
        )
    if resp.get("status_code") not in (200, None):
        return BatchResult(request_id=request_id, response=None, error=json.dumps(resp.get("body")))
    body = resp.get("body") or {}
    choices = body.get("choices") or [{}]
    text = (choices[0].get("message") or {}).get("content") or ""
    usage = body.get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    model = body.get("model", "")
    return BatchResult(
        request_id=request_id,
        response=Response(
            request_id=request_id,
            model=model,
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=_completion_cost(model, pt, ct),
            duration_ms=0,
        ),
    )


def _completion_cost(model: str, pt: int | None, ct: int | None) -> float | None:
    """Best-effort per-request cost; the batch-API discount is not applied here."""
    try:
        litellm = _load_litellm()
        return float(litellm.completion_cost(model=model, prompt_tokens=pt or 0, completion_tokens=ct or 0))
    except Exception:
        return None


class AnthropicBatchProvider:
    """Batch provider for Anthropic's native Message Batches API.

    Anthropic's batch flow isn't OpenAI-file-shaped — requests are submitted
    inline (no upload) as Messages API params, polled via ``processing_status``,
    and results pulled from a ``results_url``. LiteLLM's unified ``create_batch``
    doesn't model that, so this talks to the REST endpoints directly (stdlib
    HTTP). ``_create_batch``/``_retrieve_batch``/``_download_results`` are the
    only network surface — stubbed in tests.

    ``ANTHROPIC_API_KEY`` is read from the environment unless ``api_key`` is
    passed. JSON-mode requests rely on the prompt (Anthropic has no
    ``response_format`` knob); cost is litellm's list price (no batch discount).
    """

    provider_name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str = "https://api.anthropic.com",
        anthropic_version: str = "2023-06-01",
        max_tokens_default: int = 4096,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.anthropic_version = anthropic_version
        self.max_tokens_default = max_tokens_default
        self.timeout = timeout

    def submit(self, requests: list[LLMRequest]) -> BatchHandle:
        payload = {"requests": [_to_anthropic_request(r, self.max_tokens_default) for r in requests]}
        batch = self._create_batch(payload)
        return BatchHandle(
            batch_id=batch["id"],
            provider=self.provider_name,
            request_ids=tuple(r.request_id for r in requests),
        )

    def is_complete(self, batch_id: str) -> bool:
        return self._retrieve_batch(batch_id).get("processing_status") == "ended"

    def fetch(self, batch_id: str) -> Iterable[BatchResult]:
        results_url = self._retrieve_batch(batch_id).get("results_url")
        if not results_url:
            return []  # ended without results (e.g. all cancelled) — the poller reconciles tagged requests
        results: list[BatchResult] = []
        for line in self._download_results(results_url).splitlines():
            line = line.strip()
            if line:
                results.append(_anthropic_result_to_result(json.loads(line)))
        return results

    # --- network seam (stubbed in tests) ---------------------------------

    def _create_batch(self, payload: dict) -> dict:
        return json.loads(self._http("POST", f"{self.api_base}/v1/messages/batches", payload))

    def _retrieve_batch(self, batch_id: str) -> dict:
        return json.loads(self._http("GET", f"{self.api_base}/v1/messages/batches/{batch_id}"))

    def _download_results(self, results_url: str) -> str:
        return self._http("GET", results_url).decode("utf-8")

    def _http(self, method: str, url: str, payload: dict | None = None) -> bytes:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; required for batch_provider: anthropic")
        headers = {
            "x-api-key": key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"anthropic batch {method} failed ({e.code}): {body}") from e


def _to_anthropic_request(req: LLMRequest, max_tokens_default: int) -> dict:
    """Map an LLMRequest to one Message Batches request (Messages API params)."""
    system = [m["content"] for m in req.messages if m.get("role") == "system"]
    params: dict[str, Any] = {
        "model": _model_name(req.model_key),
        "max_tokens": req.max_tokens or max_tokens_default,  # required by the Messages API
        "messages": [m for m in req.messages if m.get("role") != "system"],
    }
    if system:
        params["system"] = "\n\n".join(system)  # Anthropic carries system prompts out-of-band
    if req.temperature is not None:
        params["temperature"] = req.temperature
    return {"custom_id": req.request_id, "params": params}


def _anthropic_result_to_result(line: dict) -> BatchResult:
    """Map one Message Batches result record to a :class:`BatchResult`."""
    request_id = line.get("custom_id", "")
    result = line.get("result") or {}
    if result.get("type") != "succeeded":
        err = result.get("error") or result.get("type") or "unknown batch error"
        return BatchResult(
            request_id=request_id, response=None, error=err if isinstance(err, str) else json.dumps(err)
        )
    message = result.get("message") or {}
    text = "".join(b.get("text", "") for b in message.get("content", []) if b.get("type") == "text")
    usage = message.get("usage") or {}
    model = message.get("model", "")
    pt = usage.get("input_tokens")
    ct = usage.get("output_tokens")
    return BatchResult(
        request_id=request_id,
        response=Response(
            request_id=request_id,
            model=model,
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=_completion_cost(model, pt, ct),
            duration_ms=0,
        ),
    )


def make_fulfill_batch(provider: BatchProvider):
    """Build a ``fulfill`` callable bound to a specific :class:`BatchProvider`.

    Plug into ``Dispatcher(..., fulfill=make_fulfill_batch(provider))``.
    """

    def _fulfill(requests: list[RequestRow], dispatcher) -> None:
        if not requests:
            return
        # Different providers can't share a batch; group by provider prefix.
        groups: dict[str, list[RequestRow]] = defaultdict(list)
        for r in requests:
            groups[_provider_of(r.model_key)].append(r)

        for provider_prefix, group in groups.items():
            llm_requests = [row_to_llm_request(r) for r in group]
            try:
                handle = provider.submit(llm_requests)
            except Exception as e:
                logger.error("batch submit failed for {}: {}", provider_prefix, e)
                cap = dispatcher.cfg.dispatcher.max_request_failures
                for r in group:
                    dispatcher.store.mark_request_failed(
                        r.request_id,
                        f"batch_submit_error: {e}",
                        max_failures=cap,
                    )
                continue
            dispatcher.store.tag_batch(handle.batch_id, [r.request_id for r in group])
            logger.info("submitted batch {} ({} requests)", handle.batch_id, len(group))

    return _fulfill


def poll_outstanding_batches(provider: BatchProvider, dispatcher) -> int:
    """Poll any batches the dispatcher's store has tagged as in-flight.

    Returns the number of newly-completed requests.
    """
    store: Store = dispatcher.store
    completed = 0
    cap = dispatcher.cfg.dispatcher.max_request_failures
    for batch_id in store.outstanding_batch_ids(dispatcher.run_id):
        try:
            if not provider.is_complete(batch_id):
                continue
        except Exception as e:
            logger.warning("poll error for batch {}: {}", batch_id, e)
            continue
        # Snapshot what's tagged so we can reconcile anything the fetch omits.
        tagged = set(store.request_ids_for_batch(batch_id))
        try:
            results = list(provider.fetch(batch_id))
        except Exception as e:
            logger.warning("fetch error for batch {}: {}", batch_id, e)
            continue
        seen: set[str] = set()
        for br in results:
            # Safe to skip: in-flight requests are always tagged, so the sweep below catches any we drop.
            if br.request_id not in tagged:
                logger.warning("batch {} returned untagged request {!r}; skipping", batch_id, br.request_id)
                continue
            seen.add(br.request_id)
            if br.error is not None or br.response is None:
                store.mark_request_failed(
                    br.request_id,
                    br.error or "batch fetch missing response",
                    max_failures=cap,
                )
                continue
            store.insert_response(
                request_id=br.request_id,
                model=br.response.model,
                text=br.response.text,
                prompt_tokens=br.response.prompt_tokens,
                completion_tokens=br.response.completion_tokens,
                cost_usd=br.response.cost_usd,
                duration_ms=br.response.duration_ms,
            )
            completed += 1
        # A terminal batch that returned no result for a tagged request (wholesale
        # failure, expiry, partial output) would otherwise strand it IN_FLIGHT once
        # the tag is cleared. Fail it so the request retries or dead-letters.
        for request_id in tagged - seen:
            store.mark_request_failed(
                request_id,
                "batch completed without a result for this request",
                max_failures=cap,
            )
        store.clear_batch(batch_id)
    return completed


def _provider_of(model_key: str) -> str:
    """Return the provider prefix of a LiteLLM model string (``openai/x`` → ``openai``)."""
    return model_key.split("/", 1)[0]
