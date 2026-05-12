"""Local (thread-pool) fulfillment strategy.

The dispatcher claims pending requests in chunks; this strategy submits each
to a persistent thread pool and inserts the response as the future completes.
"""
from __future__ import annotations

from concurrent.futures import as_completed
from typing import TYPE_CHECKING

from loguru import logger

from autodata.dispatcher.hydration import row_to_llm_request
from autodata.llm import Response
from autodata.store import RequestRow

if TYPE_CHECKING:
    from autodata.dispatcher.core import Dispatcher


def fulfill_local(requests: list[RequestRow], dispatcher: Dispatcher) -> None:
    """Thread-pool concurrent HTTP. Each request becomes a future; responses
    are inserted as they complete. Uses the dispatcher's persistent pool."""
    if not requests:
        return
    pool = dispatcher._executor()
    futures = [pool.submit(_one_request, r, dispatcher) for r in requests]
    for fut in as_completed(futures):
        try:
            fut.result()
        except Exception:
            logger.exception("dispatcher fulfill worker error")


def _one_request(req_row: RequestRow, dispatcher: Dispatcher) -> None:
    request = row_to_llm_request(req_row)
    try:
        resp: Response = dispatcher.llm.complete(request)
    except Exception as e:
        logger.warning("request {} failed: {}", req_row.request_id, e)
        dispatcher.store.mark_request_failed(req_row.request_id, str(e)[:500])
        return
    dispatcher.store.insert_response(
        request_id=req_row.request_id,
        model=resp.model,
        text=resp.text,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        cost_usd=resp.cost_usd,
        duration_ms=resp.duration_ms,
    )
