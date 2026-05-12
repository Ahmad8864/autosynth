"""Dispatcher: the only thing that touches the network.

One ``Dispatcher`` class drives the run loop:

  while not stop:
      advance any items with unconsumed responses (calls pure step())
      claim and fulfill pending requests
      poll outstanding batches (no-op for local)
      check budget; honor signals

``fulfill`` is a strategy callable — `fulfill_local` for thread-pool
execution, `fulfill_batch` for provider batch APIs. The local strategy is
shipped here; batch lives in ``dispatcher_batch`` (commit 9).

State reconstruction from the store happens in :py:meth:`Dispatcher._load_item_state`.
Persistence of a ``StepResult`` happens in :py:meth:`Dispatcher._persist_step_result`.
Both are this module's responsibility — they bridge the pure pipeline with
the durable store.

See MIGRATION_PLAN.md §4.
"""
from __future__ import annotations

import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger

from autodata.config import RunConfig
from autodata.domain import DomainAdapter, GroundingItem
from autodata.harness import DEFAULT_HARNESS, HarnessSpec
from autodata.llm import LLMClient, LLMRequest, Response
from autodata.pipeline import (
    ItemState,
    ScoreRecord,
    State,
    StepResponse,
    StepResult,
    TERMINAL_STATES,
    step,
)
from autodata.safety import SafetyFilter, load_filter
from autodata.schemas import Candidate, EvalReport, QualityCheck, Round, SolverScore
from autodata.store import (
    REQ_DONE,
    REQ_FAILED,
    REQ_PENDING,
    RUN_STATUS_ABORTED,
    RUN_STATUS_COMPLETED,
    RequestRow,
    Store,
)


@dataclass
class RunSummary:
    run_id: str
    accepted: int
    rejected: int
    state_counts: dict[str, int]
    cost_usd: float


Fulfill = Callable[[list[RequestRow], "Dispatcher"], None]


# ---------------------------------------------------------------------------
# fulfill strategies
# ---------------------------------------------------------------------------

def fulfill_local(requests: list[RequestRow], dispatcher: "Dispatcher") -> None:
    """Thread-pool concurrent HTTP. Each request becomes a future; responses
    are inserted as they complete."""
    if not requests:
        return
    concurrency = max(1, dispatcher.cfg.dispatcher.concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_one_request, r, dispatcher): r for r in requests}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                logger.exception("dispatcher fulfill worker error")


def _one_request(req_row: RequestRow, dispatcher: "Dispatcher") -> None:
    request = LLMRequest(
        request_id=req_row.request_id,
        item_id=req_row.item_id,
        round_n=req_row.round_n,
        role=req_row.role,
        model_key=req_row.model_key,
        messages=req_row.messages,
        json_mode=req_row.json_mode,
        attempt=req_row.attempt,
        parent_response_id=req_row.parent_response_id,
    )
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


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

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
        poll_in_flight: Optional[Callable[["Dispatcher"], int]] = None,
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
        self.safety_filter: Optional[SafetyFilter] = (
            load_filter(cfg.safety.filter) if cfg.safety.enabled else None
        )
        self._stop = threading.Event()
        self._installed_handlers: list = []

    # ---- public entry --------------------------------------------------

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
        return self._summarize()

    # ---- main loop -----------------------------------------------------

    def _main_loop(self) -> None:
        while not self._stop.is_set():
            advanced = self._advance_ready_items()
            dispatched = self._dispatch_pending()
            polled = self.poll_in_flight(self) if self.poll_in_flight else 0
            self._mark_unrecoverable_items()    # any newly-capped items → REJECTED
            in_flight = self.store.in_flight_count(self.run_id)
            if not advanced and not dispatched and not polled and in_flight == 0:
                if not self.store.has_non_terminal_items(self.run_id):
                    break
                if self.store.pending_count(self.run_id) == 0:
                    # Nothing left to do but the items aren't terminal either.
                    # This indicates stuck state — exit to avoid spinning.
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

    # ---- advancement (pipeline → store) --------------------------------

    def _advance_ready_items(self) -> int:
        count = 0
        # First-step items (PENDING) advance with empty responses.
        pending_first = self.store.items_pending_first_step(
            self.run_id, limit=self.cfg.dispatcher.items_per_advance
        )
        for row in pending_first:
            self._advance_one(row)
            count += 1
        # Items with new responses since their last update.
        ready = self.store.items_ready_to_advance(
            self.run_id, limit=self.cfg.dispatcher.items_per_advance
        )
        for row in ready:
            self._advance_one(row)
            count += 1
        return count

    def _advance_one(self, item_row) -> None:
        item_state = self._load_item_state(item_row)
        responses = self._hydrate_responses(item_row["item_id"], item_row["updated_at"])
        grounding = self.grounding.get(item_state.source_id)
        if grounding is None:
            logger.error("no grounding item for source_id={}; marking REJECTED",
                         item_state.source_id)
            self.store.update_item(
                item_state.item_id, state=State.REJECTED.value,
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
        self._persist_step_result(item_state, result)

    def _persist_step_result(self, before: ItemState, result: StepResult) -> None:
        new_state = result.state
        # Upsert round row for the *current* round if a candidate exists.
        if new_state.candidate is not None:
            self.store.upsert_round(
                item_id=new_state.item_id,
                round_n=new_state.current_round,
                candidate=new_state.candidate,
                quality=new_state.quality,
            )
        # Completed-round bookkeeping (eval + finalize + accepted).
        if result.completed_round is not None:
            cr = result.completed_round
            self.store.upsert_round(
                item_id=new_state.item_id,
                round_n=cr.refinement_round,
                candidate=cr.candidate,
                quality=cr.quality,
                evaluation=cr.evaluation,
            )
            if new_state.state == State.ACCEPTED:
                self.store.finalize_round(new_state.item_id, cr.refinement_round, accepted=True)
                accepted_payload = self.domain.format_accepted(cr.candidate, _accepted_extras(new_state, cr))
                self.store.insert_accepted(
                    item_id=new_state.item_id, round_n=cr.refinement_round,
                    payload=accepted_payload,
                )
            else:
                self.store.finalize_round(new_state.item_id, cr.refinement_round, accepted=False)
        # Persist scores produced this step.
        for sr in result.scores_to_persist:
            self.store.insert_score(
                item_id=new_state.item_id, round_n=new_state.current_round,
                score=sr.score,
                solver_response_id=sr.solver_response_id,
                judge_response_id=sr.judge_response_id,
            )
        # Persist last_feedback into the previous round's reflection column
        # so resume can rehydrate it.
        if (new_state.state == State.NEED_CANDIDATE and new_state.current_round > 1
                and new_state.last_feedback):
            self.store.upsert_round(
                item_id=new_state.item_id,
                round_n=new_state.current_round - 1,
                reflection=json.dumps(list(new_state.last_feedback)),
            )
        # New requests.
        if result.new_requests:
            self.store.insert_requests([_request_to_row(r) for r in result.new_requests])
        # Item-level updates.
        final_round = (new_state.current_round
                       if new_state.state in TERMINAL_STATES else None)
        self.store.update_item(
            new_state.item_id,
            state=new_state.state.value,
            current_round=new_state.current_round,
            final_round=final_round,
            rejection_reasons=(list(new_state.rejection_reasons) or None
                               if new_state.state == State.REJECTED else None),
        )

    # ---- dispatching pending requests ----------------------------------

    def _dispatch_pending(self) -> int:
        claim_limit = max(1, self.cfg.dispatcher.concurrency)
        claimed = self.store.claim_pending(limit=claim_limit)
        if not claimed:
            return 0
        self.fulfill(claimed, self)
        return len(claimed)

    # ---- state reconstruction ------------------------------------------

    def _load_item_state(self, item_row) -> ItemState:
        item_id = item_row["item_id"]
        rounds_rows = self.store.rounds_for_item(item_id)
        current_round = int(item_row["current_round"])

        rounds_history: list[Round] = []
        current_round_blob: Optional[dict] = None
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
        rows = self.store.responses_since(item_id, since_ts)
        out: list[StepResponse] = []
        for rr in rows:
            req = self.store.get_request(rr.request_id)
            if req is None:
                continue
            sr = StepResponse(
                request_id=rr.request_id,
                role=req.role,
                round_n=req.round_n,
                attempt=req.attempt,
                text=rr.text,
                parent_response_id=req.parent_response_id,
            )
            if req.role == "judge" and req.parent_response_id:
                parent_req = self.store.get_request(req.parent_response_id)
                parent_resp = self.store.get_response(req.parent_response_id)
                if parent_req is not None:
                    sr = StepResponse(
                        request_id=rr.request_id,
                        role=req.role,
                        round_n=req.round_n,
                        attempt=req.attempt,
                        text=rr.text,
                        parent_response_id=req.parent_response_id,
                        solver_response_text=parent_resp.text if parent_resp else "",
                        solver_role=parent_req.role,
                    )
            out.append(sr)
        return out

    # ---- resume / unrecoverable ---------------------------------------

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
                item_id, state=State.REJECTED.value,
                rejection_reasons=[f"unrecoverable: request failed {cap} times"],
            )

    # ---- budget / signals ----------------------------------------------

    def _budget_exceeded(self) -> bool:
        cap = self.cfg.budget_usd
        if cap is None:
            return False
        spent = self.store.cost_so_far(self.run_id)
        if spent >= cap:
            return True
        if spent >= 0.8 * cap and not getattr(self, "_budget_warned", False):
            logger.warning("cost ${:.4f} reached 80% of budget ${:.4f}", spent, cap)
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
                # off-main-thread: signals can't be installed here. fine.
                pass

    def _uninstall_signal_handlers(self) -> None:
        for sig, prev in self._installed_handlers:
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._installed_handlers.clear()

    # ---- summary -------------------------------------------------------

    def _summarize(self) -> RunSummary:
        counts = self.store.items_terminal_counts(self.run_id)
        return RunSummary(
            run_id=self.run_id,
            accepted=counts.get(State.ACCEPTED.value, 0),
            rejected=counts.get(State.REJECTED.value, 0),
            state_counts=counts,
            cost_usd=self.store.cost_so_far(self.run_id),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
    cand = (Candidate.model_validate_json(row["candidate_blob"])
            if row["candidate_blob"] else None)
    quality = (QualityCheck.model_validate_json(row["quality_blob"])
               if row["quality_blob"] else None)
    ev = (EvalReport.model_validate_json(row["eval_blob"])
          if row["eval_blob"] else None)
    return Round(
        refinement_round=int(row["round_n"]),
        candidate=cand or Candidate(candidate_id="x", domain="x", source_id="x",
                                    payload={}, rubric=[]),
        quality=quality or QualityCheck(passed=False),
        evaluation=ev,
        reflection=row["reflection"],
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
