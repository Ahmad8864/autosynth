"""Local (thread-pool) fulfillment strategy.

Fire-and-forget: the dispatcher claims pending requests in chunks and this
strategy submits each one to a persistent thread pool, returning as soon as
the work is queued. Workers post responses (or failure rows) directly to
the store and ping ``dispatcher.notify()`` so the main loop wakes on
completion instead of waiting out the idle poll interval.

This mirrors the :mod:`autodata.dispatcher.batch` strategy's shape — the
main loop drives polling either way — and lets ``in_flight_count`` reflect
real transient state for the progress bar, budget checks, and the
concurrency cap in :meth:`Dispatcher._dispatch_pending`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from autodata.dispatcher.hydration import row_to_llm_request
from autodata.llm import Response
from autodata.store import RequestRow

if TYPE_CHECKING:
    from autodata.dispatcher.core import Dispatcher


def fulfill_local(requests: list[RequestRow], dispatcher: Dispatcher) -> None:
    """Submit each request to the dispatcher's thread pool and return.

    Workers run :func:`_one_request` independently; the main loop observes
    completions via ``in_flight_count`` and the ``notify`` wake-up.
    """
    if not requests:
        return
    pool = dispatcher._executor()
    for r in requests:
        pool.submit(_one_request, r, dispatcher)


def _one_request(req_row: RequestRow, dispatcher: Dispatcher) -> None:
    # Top-level guard: without an ``as_completed`` collector at the call
    # site, bugs in store writes or hydration would otherwise vanish.
    try:
        request = row_to_llm_request(req_row)
        try:
            resp: Response = dispatcher.llm.complete(request)
        except Exception as e:
            logger.warning("request {} failed: {}", req_row.request_id, e)
            dispatcher.store.mark_request_failed(
                req_row.request_id,
                str(e)[:500],
                max_failures=dispatcher.cfg.dispatcher.max_request_failures,
            )
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
    except Exception:
        logger.exception("dispatcher worker error for request {}", req_row.request_id)
    finally:
        dispatcher.notify()
