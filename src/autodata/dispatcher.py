"""Dispatcher: the only thing that touches the network.

The run loop, in pseudocode::

    while not stop:
        advance items with unconsumed responses (calls pure step())
        claim and fulfill pending requests
        poll outstanding batches (no-op for local)
        check budget; honor signals

``fulfill`` is a strategy callable — :func:`fulfill_local` (thread-pool) is
shipped here; the batch-API variant lives in :mod:`autodata.dispatcher_batch`.
:meth:`Dispatcher._load_item_state` reconstructs in-flight state from the
store before each call to :func:`pipeline.step`;
:meth:`Dispatcher._persist_step_result` writes the result back.
"""
from __future__ import annotations

import contextlib
import json
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from loguru import logger

from autodata.config import RunConfig
from autodata.domain import DomainAdapter, GroundingItem
from autodata.harness import DEFAULT_HARNESS, HarnessSpec
from autodata.llm import LLMClient, LLMRequest, Response
from autodata.pipeline import (
    TERMINAL_STATES,
    ItemState,
    State,
    StepResponse,
    StepResult,
    step,
)
from autodata.safety import SafetyFilter, load_filter
from autodata.schemas import Candidate, EvalReport, QualityCheck, Round
from autodata.store import (
    ITEM_ACCEPTED,
    ITEM_REJECTED,
    RUN_STATUS_ABORTED,
    RUN_STATUS_COMPLETED,
    RequestRow,
    Store,
)

_BUDGET_WARN_FRACTION = 0.8


@dataclass
class RunSummary:
    run_id: str
    accepted: int
    rejected: int
    state_counts: dict[str, int]
    cost_usd: float


Fulfill = Callable[[list[RequestRow], "Dispatcher"], None]


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
    request = _row_to_llm_request(req_row)
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


def _row_to_llm_request(r: RequestRow) -> LLMRequest:
    """Hydrate a stored RequestRow back into an LLMRequest for fulfillment."""
    return LLMRequest(
        request_id=r.request_id,
        item_id=r.item_id,
        round_n=r.round_n,
        role=r.role,
        model_key=r.model_key,
        messages=r.messages,
        json_mode=r.json_mode,
        attempt=r.attempt,
        parent_response_id=r.parent_response_id,
    )


class Dispatcher:
    def __init__(
        self,
        *,
        store: Store,
        llm: LLMClient,
        domain: DomainAdapter,
        cfg: RunConfig,
        run_id: str,
        harness: HarnessSpec | None = None,
        grounding: dict[str, GroundingItem] | None = None,
        fulfill: Fulfill = fulfill_local,
        poll_in_flight: Callable[[Dispatcher], int] | None = None,
    ):
        self.store = store
        self.llm = llm
        self.domain = domain
        self.cfg = cfg
        self.run_id = run_id
        self.harness = harness or DEFAULT_HARNESS
        self.grounding = grounding or {}
        self.fulfill = fulfill
        self.poll_in_flight = poll_in_flight
        self.safety_filter: SafetyFilter | None = (
            load_filter(cfg.safety.filter) if cfg.safety.enabled else None
        )
        self._stop = threading.Event()
        self._installed_handlers: list = []
        self._pool: ThreadPoolExecutor | None = None
        self._budget_warned = False

    def _executor(self) -> ThreadPoolExecutor:
        """Lazy persistent thread pool for fulfill_local — created once per run."""
        if self._pool is None:
            workers = max(1, self.cfg.dispatcher.concurrency)
            self._pool = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="autodata-dispatcher",
            )
        return self._pool

    def run(self) -> RunSummary:
        self._install_signal_handlers()
        self._normalize_for_resume()
        self._mark_unrecoverable_items()
        try:
            self._main_loop()
            if not self._stop.is_set() and not self.store.has_non_terminal_items(self.run_id):
                self.store.update_run_status(self.run_id, RUN_STATUS_COMPLETED)
        finally:
            self._uninstall_signal_handlers()
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
        return self._summarize()

    def _main_loop(self) -> None:
        while not self._stop.is_set():
            advanced = self._advance_ready_items()
            dispatched = self._dispatch_pending()
            polled = self.poll_in_flight(self) if self.poll_in_flight else 0
            self._mark_unrecoverable_items()
            in_flight = self.store.in_flight_count(self.run_id)
            if not advanced and not dispatched and not polled and in_flight == 0:
                if not self.store.has_non_terminal_items(self.run_id):
                    break
                if self.store.pending_count(self.run_id) == 0:
                    # Items are non-terminal but nothing is queued or in flight —
                    # break out instead of spinning on a stuck state.
                    logger.warning(
                        "dispatcher idle but {} items non-terminal; exiting",
                        len(self.store.items_terminal_counts(self.run_id)),
                    )
                    break
                time.sleep(self.cfg.dispatcher.poll_interval_s)
            if self._budget_exceeded():
                logger.warning("budget exceeded; aborting run")
                self.store.update_run_status(self.run_id, RUN_STATUS_ABORTED)
                break

    def _advance_ready_items(self) -> int:
        per_advance = self.cfg.dispatcher.items_per_advance
        rows = (
            self.store.items_pending_first_step(self.run_id, limit=per_advance)
            + self.store.items_ready_to_advance(self.run_id, limit=per_advance)
        )
        for row in rows:
            self._advance_one(row)
        return len(rows)

    def _advance_one(self, item_row) -> None:
        item_state = self._load_item_state(item_row)
        responses = self._hydrate_responses(item_row["item_id"], item_row["updated_at"])
        grounding = self.grounding.get(item_state.source_id)
        if grounding is None:
            logger.error(
                "no grounding item for source_id={}; marking REJECTED",
                item_state.source_id,
            )
            self.store.update_item(
                item_state.item_id,
                state=ITEM_REJECTED,
                rejection_reasons=["unrecoverable: missing grounding"],
            )
            return
        try:
            result = step(
                item_state, responses,
                cfg=self.cfg, harness=self.harness, domain=self.domain,
                grounding=grounding, safety_filter=self.safety_filter,
            )
        except Exception:
            logger.exception("pipeline step crashed for item {}", item_state.item_id)
            return
        self._persist_step_result(result)

    def _persist_step_result(self, result: StepResult) -> None:
        new_state = result.state
        completed = result.completed_round

        # Persist the current round's candidate + quality; if the same round
        # also completed (eval available), merge into one upsert below.
        if new_state.candidate is not None and (
            completed is None or completed.refinement_round != new_state.current_round
        ):
            self.store.upsert_round(
                item_id=new_state.item_id,
                round_n=new_state.current_round,
                candidate=new_state.candidate,
                quality=new_state.quality,
            )

        if completed is not None:
            self.store.upsert_round(
                item_id=new_state.item_id,
                round_n=completed.refinement_round,
                candidate=completed.candidate,
                quality=completed.quality,
                evaluation=completed.evaluation,
            )
            accepted = new_state.state == State.ACCEPTED
            self.store.finalize_round(
                new_state.item_id, completed.refinement_round, accepted=accepted,
            )
            if accepted:
                payload = self.domain.format_accepted(
                    completed.candidate, _accepted_extras(new_state, completed),
                )
                self.store.insert_accepted(
                    item_id=new_state.item_id,
                    round_n=completed.refinement_round,
                    payload=payload,
                )

        for sr in result.scores_to_persist:
            self.store.insert_score(
                item_id=new_state.item_id,
                round_n=new_state.current_round,
                score=sr.score,
                solver_response_id=sr.solver_response_id,
                judge_response_id=sr.judge_response_id,
            )

        # Stash last_feedback into the previous round's reflection column so
        # a resume can rehydrate it for the new challenger.
        if (
            new_state.state == State.NEED_CANDIDATE
            and new_state.current_round > 1
            and new_state.last_feedback
        ):
            self.store.upsert_round(
                item_id=new_state.item_id,
                round_n=new_state.current_round - 1,
                reflection=json.dumps(list(new_state.last_feedback)),
            )

        if result.new_requests:
            self.store.insert_requests([_request_to_row(r) for r in result.new_requests])

        final_round = (
            new_state.current_round if new_state.state in TERMINAL_STATES else None
        )
        self.store.update_item(
            new_state.item_id,
            state=new_state.state.value,
            current_round=new_state.current_round,
            final_round=final_round,
            rejection_reasons=(
                list(new_state.rejection_reasons) or None
                if new_state.state == State.REJECTED else None
            ),
        )

    def _dispatch_pending(self) -> int:
        claim_limit = max(1, self.cfg.dispatcher.concurrency)
        claimed = self.store.claim_pending(limit=claim_limit)
        if not claimed:
            return 0
        self.fulfill(claimed, self)
        return len(claimed)

    def _load_item_state(self, item_row) -> ItemState:
        item_id = item_row["item_id"]
        rounds_rows = self.store.rounds_for_item(item_id)
        current_round = int(item_row["current_round"])

        rounds_history: list[Round] = []
        current_round_blob: dict | None = None
        for r in rounds_rows:
            obj = _row_to_round(r)
            if r["round_n"] < current_round:
                rounds_history.append(obj)
            elif r["round_n"] == current_round:
                current_round_blob = dict(r)

        candidate = None
        quality = None
        if current_round_blob is not None:
            if current_round_blob["candidate_blob"]:
                candidate = Candidate.model_validate_json(current_round_blob["candidate_blob"])
            if current_round_blob["quality_blob"]:
                quality = QualityCheck.model_validate_json(current_round_blob["quality_blob"])

        scores = self.store.scores_for_round(item_id, current_round)
        weak_scores = tuple(s for s in scores if s.solver == "weak")
        strong_scores = tuple(s for s in scores if s.solver == "strong")

        # last_feedback from previous round's reflection column.
        last_feedback: tuple[str, ...] = ()
        if current_round > 1:
            prev = self.store.get_round(item_id, current_round - 1)
            if prev and prev["reflection"]:
                try:
                    last_feedback = tuple(json.loads(prev["reflection"]))
                except (json.JSONDecodeError, TypeError):
                    last_feedback = ()

        source_metadata = (json.loads(item_row["source_metadata"])
                           if item_row["source_metadata"] else {})
        rejection_reasons = tuple(
            json.loads(item_row["rejection_reasons"])
            if item_row["rejection_reasons"] else []
        )

        return ItemState(
            item_id=item_id,
            run_id=item_row["run_id"],
            source_id=item_row["source_id"],
            domain=item_row["domain"],
            state=State(item_row["state"]),
            current_round=current_round,
            rounds_history=tuple(rounds_history),
            candidate=candidate,
            quality=quality,
            weak_scores=weak_scores,
            strong_scores=strong_scores,
            last_feedback=last_feedback,
            source_metadata=source_metadata,
            rejection_reasons=rejection_reasons,
        )

    def _hydrate_responses(self, item_id: str, since_ts: str) -> list[StepResponse]:
        """Pull responses + matching request/parent fields in a single query."""
        out: list[StepResponse] = []
        for row in self.store.hydrate_responses(item_id, since_ts):
            is_judge = row["role"] == "judge" and row["parent_response_id"] is not None
            out.append(StepResponse(
                request_id=row["request_id"],
                role=row["role"],
                round_n=row["round_n"],
                attempt=row["attempt"],
                text=row["text"],
                parent_response_id=row["parent_response_id"],
                solver_response_text=(row["parent_text"] or "") if is_judge else None,
                solver_role=row["parent_role"] if is_judge else None,
            ))
        return out

    def _normalize_for_resume(self) -> None:
        if not self.cfg.resume:
            return
        counts = self.store.normalize_for_resume(
            run_id=self.run_id,
            max_request_failures=self.cfg.dispatcher.max_request_failures,
        )
        if any(counts.values()):
            logger.info("resume normalization: {}", counts)

    def _mark_unrecoverable_items(self) -> None:
        cap = self.cfg.dispatcher.max_request_failures
        for item_id in self.store.unrecoverable_items(self.run_id, cap):
            logger.warning("item {} hit failure cap; marking REJECTED", item_id)
            self.store.update_item(
                item_id,
                state=ITEM_REJECTED,
                rejection_reasons=[f"unrecoverable: request failed {cap} times"],
            )

    def _budget_exceeded(self) -> bool:
        cap = self.cfg.budget_usd
        if cap is None:
            return False
        spent = self.store.cost_so_far(self.run_id)
        if spent >= cap:
            return True
        if spent >= _BUDGET_WARN_FRACTION * cap and not self._budget_warned:
            logger.warning(
                "cost ${:.4f} reached {:.0%} of budget ${:.4f}",
                spent, _BUDGET_WARN_FRACTION, cap,
            )
            self._budget_warned = True
        return False

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            logger.warning("received signal {}; finishing in-flight then exiting", signum)
            self._stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.signal(sig, handler)
                self._installed_handlers.append((sig, prev))
            except (ValueError, OSError):
                # Off-main-thread: signal installation is restricted; ignore.
                pass

    def _uninstall_signal_handlers(self) -> None:
        for sig, prev in self._installed_handlers:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, prev)
        self._installed_handlers.clear()

    def _summarize(self) -> RunSummary:
        counts = self.store.items_terminal_counts(self.run_id)
        return RunSummary(
            run_id=self.run_id,
            accepted=counts.get(ITEM_ACCEPTED, 0),
            rejected=counts.get(ITEM_REJECTED, 0),
            state_counts=counts,
            cost_usd=self.store.cost_so_far(self.run_id),
        )


def _request_to_row(r: LLMRequest) -> dict:
    return {
        "request_id": r.request_id,
        "item_id": r.item_id,
        "round_n": r.round_n,
        "role": r.role,
        "model_key": r.model_key,
        "attempt": r.attempt,
        "messages": r.messages,
        "json_mode": r.json_mode,
        "parent_response_id": r.parent_response_id,
    }


def _row_to_round(row) -> Round:
    candidate = (Candidate.model_validate_json(row["candidate_blob"])
                 if row["candidate_blob"] else None)
    quality = (QualityCheck.model_validate_json(row["quality_blob"])
               if row["quality_blob"] else None)
    evaluation = (EvalReport.model_validate_json(row["eval_blob"])
                  if row["eval_blob"] else None)
    # A historical round (round_n < current_round) always has a persisted
    # candidate; the placeholders here only matter for partially-written rows
    # observed mid-write during resume.
    return Round(
        refinement_round=int(row["round_n"]),
        candidate=candidate or _PLACEHOLDER_CANDIDATE,
        quality=quality or QualityCheck(passed=False),
        evaluation=evaluation,
        reflection=row["reflection"],
    )


_PLACEHOLDER_CANDIDATE = Candidate(
    candidate_id="missing", domain="missing", source_id="missing",
    payload={}, rubric=[],
)


def _accepted_extras(item: ItemState, round_obj: Round) -> dict:
    ev = round_obj.evaluation
    return {
        "run_id": item.run_id,
        "item_id": item.item_id,
        "source_id": item.source_id,
        "refinement_round": round_obj.refinement_round,
        "weak_avg": ev.weak_avg if ev else None,
        "strong_avg": ev.strong_avg if ev else None,
        "gap": ev.gap if ev else None,
        "weak_scores": [s.model_dump() for s in (ev.weak_scores if ev else [])],
        "strong_scores": [s.model_dump() for s in (ev.strong_scores if ev else [])],
        "acceptance_rationale": ev.acceptance_rationale if ev else None,
    }
