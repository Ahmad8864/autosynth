"""Batch dispatcher: submit pending requests in chunks to provider batch APIs.

Implements the ``fulfill`` strategy for ``Dispatcher`` that uses provider
batch endpoints (OpenAI, Anthropic) instead of streaming HTTP. The pipeline
state machine is unchanged across the batch SLA; resume works because every
request carries a deterministic ``request_id`` and the ``batch_id`` column
tags in-flight batch submissions.

The module defines a small :class:`BatchProvider` protocol and an in-process
:class:`MockBatchProvider` used by tests and demos. Real provider
implementations (OpenAI / Anthropic) live alongside their SDK adapters.
"""
from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from autodata.llm import LLMClient, LLMRequest, Response
from autodata.store import RequestRow, Store

__all__ = [
    "BatchProvider",
    "BatchHandle",
    "BatchResult",
    "MockBatchProvider",
    "make_fulfill_batch",
    "poll_outstanding_batches",
]


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
            llm_requests = [_row_to_request(r) for r in group]
            try:
                handle = provider.submit(llm_requests)
            except Exception as e:
                logger.error("batch submit failed for {}: {}", provider_prefix, e)
                for r in group:
                    dispatcher.store.mark_request_failed(
                        r.request_id, f"batch_submit_error: {e}",
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
    for batch_id in store.outstanding_batch_ids(dispatcher.run_id):
        try:
            if not provider.is_complete(batch_id):
                continue
        except Exception as e:
            logger.warning("poll error for batch {}: {}", batch_id, e)
            continue
        try:
            results = list(provider.fetch(batch_id))
        except Exception as e:
            logger.warning("fetch error for batch {}: {}", batch_id, e)
            continue
        for br in results:
            if br.error is not None or br.response is None:
                store.mark_request_failed(
                    br.request_id, br.error or "batch fetch missing response",
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
        store.clear_batch(batch_id)
    return completed


def _provider_of(model_key: str) -> str:
    """Return the provider prefix of a LiteLLM model string (``openai/x`` → ``openai``)."""
    return model_key.split("/", 1)[0]


def _row_to_request(r: RequestRow) -> LLMRequest:
    return LLMRequest(
        request_id=r.request_id, item_id=r.item_id, round_n=r.round_n,
        role=r.role, model_key=r.model_key, messages=r.messages,
        json_mode=r.json_mode, attempt=r.attempt,
        parent_response_id=r.parent_response_id,
    )
