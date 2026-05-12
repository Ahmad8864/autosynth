"""The Autodata orchestrator.

Implements the generate → verify → evaluate → reflect → refine loop for each
source item, with configurable acceptance criteria, retries, and resume.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from autodata.agents import ChallengerAgent, Reflector, SolverAgent, VerifierJudge
from autodata.config import RunConfig
from autodata.domain import DomainAdapter, GroundingItem, build_domain
from autodata.evaluator import evaluate
from autodata.harness import DEFAULT_HARNESS, HarnessSpec
from autodata.models import LLMClient
from autodata.safety import SafetyFilter, load_filter
from autodata.schemas import EvalReport, QualityCheck, Round, Trajectory
from autodata.utils import make_run_id, stable_id, utcnow
from autodata.writer import RunWriter, build_accepted_record


class Orchestrator:
    def __init__(
        self,
        cfg: RunConfig,
        *,
        run_id: str | None = None,
        harness: HarnessSpec | None = None,
        grounding_filter: set[str] | None = None,
    ):
        self.cfg = cfg
        self.run_id = run_id or cfg.run_id or make_run_id("run", cfg.model_dump_json())
        self.harness = harness or DEFAULT_HARNESS
        self.grounding_filter = grounding_filter  # restrict to these source_ids if set
        self.domain: DomainAdapter = build_domain(cfg.domain.name, cfg.domain.path, cfg.domain.params)

        if cfg.max_concurrency > 1:
            logger.warning(
                "max_concurrency={} requested but orchestrator is currently sequential; "
                "ignoring (see README 'Limitations')",
                cfg.max_concurrency,
            )

        self.challenger = ChallengerAgent(
            self._client(cfg.challenger, "challenger"),
            self.domain,
            rubric_max_weight=cfg.acceptance.rubric_max_weight,
            harness=self.harness,
        )
        self.verifier = VerifierJudge(
            self._client(cfg.judge, "judge"),
            self.domain,
            harness=self.harness,
        )
        self.weak = SolverAgent(
            self._client(cfg.weak_solver, "weak"),
            self.domain,
            "weak",
            harness=self.harness,
        )
        self.strong = SolverAgent(
            self._client(cfg.strong_solver, "strong"),
            self.domain,
            "strong",
            harness=self.harness,
        )
        self.reflector = Reflector(
            self._client(cfg.orchestrator, "reflector"),
            acceptance=cfg.acceptance,
            harness=self.harness,
        )

        self.safety: SafetyFilter | None = load_filter(cfg.safety.filter) if cfg.safety.enabled else None

        self.writer = RunWriter(cfg, self.run_id)
        self.writer.snapshot_config()

    def _client(self, model_cfg, role: str) -> LLMClient:
        return LLMClient(
            model_cfg,
            role=role,
            timeout_s=self.cfg.request_timeout_s,
            max_retries=self.cfg.max_retries,
        )

    # -----------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        items: Iterable[GroundingItem] = self.domain.load_grounding()
        item_list = list(items)
        if self.grounding_filter is not None:
            item_list = [i for i in item_list if i.source_id in self.grounding_filter]
        if not item_list:
            logger.warning("domain {} produced no grounding items", self.domain.name)
            return self.writer.summary

        item_list = item_list[: self.cfg.max_examples]
        logger.info("run_id={} starting with {} source items", self.run_id, len(item_list))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            t = progress.add_task("autodata", total=len(item_list))
            for item in item_list:
                try:
                    self._process_item(item)
                except KeyboardInterrupt:
                    logger.warning("interrupted; flushing summary")
                    break
                except Exception as e:
                    # Record the failure so summary counters stay consistent
                    # with the number of source items processed.
                    logger.exception("unhandled error on source {}", item.source_id)
                    self.writer.write_rejected(
                        {
                            "source_id": item.source_id,
                            "domain": self.domain.name,
                            "reasons": [f"unhandled_error: {type(e).__name__}: {e}"],
                        }
                    )
                    self.writer.bump("errors")
                progress.advance(t)

        logger.info("run complete: {}", self.writer.summary)
        return self.writer.summary

    # -----------------------------------------------------------------------

    def _process_item(self, item: GroundingItem) -> None:
        existing = self.writer.load_trajectory(item.source_id) if self.cfg.resume else None
        if existing and existing.final_accepted_round is not None:
            logger.info("[{}] already accepted in prior run; skip", item.source_id)
            return

        trajectory = existing or Trajectory(
            trajectory_id=stable_id(self.run_id, item.source_id),
            run_id=self.run_id,
            domain=self.domain.name,
            source_id=item.source_id,
            source_metadata=item.metadata,
        )
        start_round = (trajectory.latest_round().refinement_round + 1) if trajectory.rounds else 1

        for round_n in range(start_round, self.cfg.loop.max_rounds + 1):
            r = self._one_round(item, trajectory, round_n)
            trajectory.rounds.append(r)
            trajectory.total_rounds = len(trajectory.rounds)

            if r.evaluation and r.evaluation.accepted:
                trajectory.final_accepted_round = round_n
                self.writer.write_trajectory(trajectory)
                record = build_accepted_record(
                    domain=self.domain,
                    trajectory=trajectory,
                    extra={"source_metadata": item.metadata},
                )
                self.writer.write_accepted(record)
                logger.info("[{}] ACCEPTED at round {}", item.source_id, round_n)
                if self.cfg.loop.stop_on_first_accept:
                    return
            else:
                self.writer.write_trajectory(trajectory)

        # all rounds exhausted without acceptance
        last = trajectory.latest_round()
        reasons = last.evaluation.rejection_reasons if last and last.evaluation else ["no_evaluation"]
        self.writer.write_rejected(
            {
                "trajectory_id": trajectory.trajectory_id,
                "source_id": item.source_id,
                "domain": self.domain.name,
                "reasons": reasons,
                "total_rounds": trajectory.total_rounds,
            }
        )
        logger.info("[{}] exhausted without acceptance: {}", item.source_id, reasons)

    # -----------------------------------------------------------------------

    def _one_round(self, item: GroundingItem, trajectory: Trajectory, round_n: int) -> Round:
        started_at = utcnow()
        feedback: list[str] = []
        prior_payloads: list[dict[str, Any]] = []
        if trajectory.rounds:
            reflection = self.reflector.reflect(
                trajectory.rounds, self.domain.name, self.domain.leakage_rules()
            )
            feedback = reflection.get("feedback", [])
            if reflection.get("new_angle"):
                feedback.append(f"NEW_ANGLE: {reflection['new_angle']}")
            prior_payloads = [r.candidate.payload for r in trajectory.rounds]

        candidate = self.challenger.generate(item, round_n, feedback, prior_payloads)

        # Domain-level structural validation
        struct_errs = self.domain.validate_candidate(candidate)
        if struct_errs:
            return _build_round(round_n, candidate, _failed_quality(struct_errs), feedback, started_at)

        # Safety filter (optional)
        if self.safety:
            verdict = self.safety(_payload_text(candidate.payload))
            if not verdict.allowed:
                quality = _failed_quality([f"safety:{r}" for r in verdict.reasons])
                return _build_round(round_n, candidate, quality, feedback, started_at)

        # Quality verifier
        quality = self.verifier.quality_check(candidate)
        if not quality.passed:
            return _build_round(round_n, candidate, quality, feedback, started_at)

        # Solver sweeps
        weak_scores = []
        for k in range(self.cfg.loop.weak_samples):
            resp = self.weak.attempt(candidate)
            weak_scores.append(self.verifier.score(candidate, resp, "weak", attempt=k))

        strong_scores = []
        for k in range(self.cfg.loop.strong_samples):
            resp = self.strong.attempt(candidate)
            strong_scores.append(self.verifier.score(candidate, resp, "strong", attempt=k))

        ev: EvalReport = evaluate(weak_scores, strong_scores, quality, self.cfg.acceptance)
        return _build_round(round_n, candidate, quality, feedback, started_at, evaluation=ev)


def _build_round(
    round_n: int,
    candidate,
    quality: QualityCheck,
    feedback: list[str],
    started_at,
    *,
    evaluation: EvalReport | None = None,
) -> Round:
    return Round(
        refinement_round=round_n,
        candidate=candidate,
        quality=quality,
        evaluation=evaluation,
        reflection=" | ".join(feedback) if feedback else None,
        started_at=started_at,
        ended_at=utcnow(),
    )


def _failed_quality(errs: list[str]) -> QualityCheck:
    return QualityCheck(passed=False, failures=errs, notes="structural/safety validation failed")


def _payload_text(payload: dict[str, Any]) -> str:
    """Flatten a payload into a single string for the safety filter.

    Uses ``json.dumps`` so nested lists/dicts (common in domain payloads, e.g.
    ``reasoning_skills: [...]``) are also scanned — a scalar-only join would
    let PII inside a list escape the check.
    """
    return json.dumps(payload, default=str)
