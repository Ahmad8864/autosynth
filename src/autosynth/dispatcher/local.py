"""Thread-pool fulfillment for local, immediate requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from autosynth.dispatcher.hydration import row_to_llm_request
from autosynth.llm import Response
from autosynth.store import RequestRow

if TYPE_CHECKING:
    from autosynth.dispatcher.core import Dispatcher


def fulfill_local(requests: list[RequestRow], dispatcher: Dispatcher) -> None:
    """Queue requests in the dispatcher's worker pool."""
    if not requests:
        return
    pool = dispatcher._executor()
    for r in requests:
        pool.submit(_one_request, r, dispatcher)


def _one_request(req_row: RequestRow, dispatcher: Dispatcher) -> None:
    # Surface worker errors even though no caller awaits the future.
    try:
        request = row_to_llm_request(req_row, dispatcher.domain)
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
