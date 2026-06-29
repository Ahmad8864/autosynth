"""Dispatcher: the only thing that touches the network.

The run loop, in pseudocode::

    while not stop:
        advance items with unconsumed responses (calls pure step())
        claim and fulfill pending requests
        poll outstanding batches (no-op for local)
        check budget; honor signals

``fulfill`` is a strategy callable — :func:`fulfill_local` (thread-pool) is
the default; the batch-API variant lives in :mod:`autosynth.dispatcher.batch`.
DTO mapping (store rows ↔ frozen pipeline state) lives in
:mod:`autosynth.dispatcher.hydration` so the run loop stays focused on flow.
"""

from __future__ import annotations

import contextlib
import json
import signal
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from loguru import logger

from autosynth.acceptance import resolve_policy
from autosynth.config import RunConfig
from autosynth.dispatcher.hydration import (
    accepted_extras,
    hydrate_responses,
    load_item_state,
    request_to_row,
)
from autosynth.dispatcher.local import fulfill_local
from autosynth.dispatcher.progress import DispatcherProgress
from autosynth.domain import DomainAdapter, GroundingItem
from autosynth.harness import DEFAULT_HARNESS, HarnessSpec
from autosynth.llm import LLMClient
from autosynth.pipeline import TERMINAL_STATES, State, StepResult, step
from autosynth.safety import SafetyFilter, load_filter
from autosynth.store import (
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
        self.policy = resolve_policy(cfg, domain)
        self.fulfill = fulfill
        self.poll_in_flight = poll_in_flight
        self.safety_filter: SafetyFilter | None = (
            load_filter(cfg.safety.filter) if cfg.safety.enabled else None
        )
        self._stop = threading.Event()
        # Workers set this after posting a response so the main loop can
        # wake immediately instead of waiting out the poll interval.
        self._work_ready = threading.Event()
        self._installed_handlers: list = []
        self._pool: ThreadPoolExecutor | None = None
        self._budget_warned = False
        self._progress: DispatcherProgress | None = None

    def _executor(self) -> ThreadPoolExecutor:
        """Lazy persistent thread pool for fulfill_local — created once per run."""
        if self._pool is None:
            workers = max(1, self.cfg.dispatcher.concurrency)
            self._pool = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="autosynth-dispatcher",
            )
        return self._pool

    def run(self) -> RunSummary:
        self._install_signal_handlers()
        self._normalize_for_resume()
        self._mark_unrecoverable_items()
        try:
            with DispatcherProgress(total=len(self.grounding)) as progress:
                self._progress = progress
                self._refresh_progress(in_flight=self.store.in_flight_count(self.run_id))
                self._main_loop()
            if not self._stop.is_set() and not self.store.has_non_terminal_items(self.run_id):
                self.store.update_run_status(self.run_id, RUN_STATUS_COMPLETED)
        finally:
            self._progress = None
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
            self._refresh_progress(in_flight=in_flight)
            if not advanced and not dispatched and not polled:
                if in_flight == 0:
                    # A worker may have committed a response between our
                    # ``items_ready_to_advance`` query above and now. Since
                    # ``in_flight_count == 0`` proves no workers are still
                    # running, any such write is now visible — re-check
                    # before deciding to break.
                    if self.store.items_ready_to_advance(self.run_id, limit=1):
                        continue
                    if not self.store.has_non_terminal_items(self.run_id):
                        break
                    if self.store.pending_count(self.run_id) == 0:
                        # Non-terminal items remain but nothing is queued or in
                        # flight — break out instead of spinning on stuck state.
                        logger.warning(
                            "dispatcher idle but non-terminal items remain; states={}",
                            self.store.items_terminal_counts(self.run_id),
                        )
                        break
                # Idle this tick: block until a worker posts a response or
                # the poll interval elapses. Workers set _work_ready in
                # ``notify``; the signal handler sets it on stop too.
                self._work_ready.wait(self.cfg.dispatcher.poll_interval_s)
                self._work_ready.clear()
            if self._budget_exceeded():
                logger.warning("budget exceeded; aborting run")
                self.store.update_run_status(self.run_id, RUN_STATUS_ABORTED)
                break

    def _advance_ready_items(self) -> int:
        per_advance = self.cfg.dispatcher.items_per_advance
        rows = self.store.items_pending_first_step(
            self.run_id, limit=per_advance
        ) + self.store.items_ready_to_advance(self.run_id, limit=per_advance)
        for row in rows:
            self._advance_one(row)
        return len(rows)

    def _advance_one(self, item_row) -> None:
        item_state = load_item_state(self.store, item_row)
        responses, consumed_watermark = hydrate_responses(
            self.store, item_row["item_id"], item_row["consumed_seq"]
        )
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
                item_state,
                responses,
                cfg=self.cfg,
                harness=self.harness,
                domain=self.domain,
                grounding=grounding,
                safety_filter=self.safety_filter,
                policy=self.policy,
            )
        except Exception as e:
            # step() is pure, so a crash recurs every tick — retrying livelocks; dead-letter.
            logger.exception("pipeline step crashed for item {}; rejecting", item_state.item_id)
            self.store.update_item(
                item_state.item_id,
                state=ITEM_REJECTED,
                rejection_reasons=[f"unrecoverable: pipeline step crashed: {type(e).__name__}: {e}"],
            )
            return
        self._persist_step_result(result, consumed_watermark=consumed_watermark)

    def _persist_step_result(
        self,
        result: StepResult,
        *,
        consumed_watermark: int | None = None,
    ) -> None:
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
                new_state.item_id,
                completed.refinement_round,
                accepted=accepted,
            )
            if accepted:
                payload = self.domain.format_accepted(
                    completed.candidate,
                    accepted_extras(new_state, completed),
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
            self.store.insert_requests([request_to_row(r) for r in result.new_requests])

        final_round = new_state.current_round if new_state.state in TERMINAL_STATES else None
        self.store.update_item(
            new_state.item_id,
            state=new_state.state.value,
            current_round=new_state.current_round,
            final_round=final_round,
            rejection_reasons=(
                list(new_state.rejection_reasons) or None if new_state.state == State.REJECTED else None
            ),
            # Advance the watermark to the max rowid of consumed responses so
            # worker rows committed concurrently with this step (invisible to our
            # hydrate snapshot, but with strictly higher rowids) aren't masked.
            # ``None`` for transitions that consumed no responses (e.g. the
            # PENDING-first-step path) — consumed_seq stays put, which is correct
            # there since it's already 0 and the challenger response has rowid ≥ 1.
            consumed_seq=consumed_watermark,
        )

    def _dispatch_pending(self) -> int:
        # Cap claims by available headroom so the worker pool's queue can't
        # grow unbounded under async fulfillment. ``concurrency`` doubles as
        # the pool's max_workers, so this bounds true parallelism honestly.
        headroom = self.cfg.dispatcher.concurrency - self.store.in_flight_count(self.run_id)
        if headroom <= 0:
            return 0
        claimed = self.store.claim_pending(limit=headroom)
        if not claimed:
            return 0
        self.fulfill(claimed, self)
        return len(claimed)

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
        for item_id, last_error in self.store.unrecoverable_items(self.run_id, cap):
            reason = f"unrecoverable after {cap} attempts: {last_error or 'unknown error'}"
            logger.warning(
                "item {} hit failure cap; marking REJECTED ({})", item_id, last_error or "no error recorded"
            )
            self.store.update_item(
                item_id,
                state=ITEM_REJECTED,
                rejection_reasons=[reason],
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
                spent,
                _BUDGET_WARN_FRACTION,
                cap,
            )
            self._budget_warned = True
        return False

    def notify(self) -> None:
        """Wake the main loop. Workers call this after posting a response."""
        self._work_ready.set()

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            logger.warning("received signal {}; finishing in-flight then exiting", signum)
            self._stop.set()
            # Cut through any in-progress idle wait so we exit promptly.
            self._work_ready.set()

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

    def _refresh_progress(self, *, in_flight: int) -> None:
        if self._progress is None:
            return
        counts = self.store.items_terminal_counts(self.run_id)
        self._progress.update(
            accepted=counts.get(ITEM_ACCEPTED, 0),
            rejected=counts.get(ITEM_REJECTED, 0),
            in_flight=in_flight,
            cost_usd=self.store.cost_so_far(self.run_id),
        )
