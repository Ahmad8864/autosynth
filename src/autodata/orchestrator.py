"""The Autodata orchestrator.

Implements the generate → verify → evaluate → reflect → refine loop for each
source item, with configurable acceptance criteria, retries, and resume.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

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
from autodata.harness import DEFAULT_HARNESS, HarnessSpec
from autodata.models import LLMClient
from autodata.safety import SafetyFilter, load_filter
from autodata.schemas import EvalReport, Round, Trajectory
from autodata.evaluator import evaluate
from autodata.utils import stable_id
from autodata.writer import RunWriter, build_accepted_record


class Orchestrator:
    def __init__(
        self,
        cfg: RunConfig,
        *,
        run_id: Optional[str] = None,
        harness: Optional[HarnessSpec] = None,
        grounding_filter: Optional[set[str]] = None,
    ):
        self.cfg = cfg
        self.run_id = run_id or cfg.run_id or _make_run_id(cfg)
        self.harness = harness or DEFAULT_HARNESS
        self.grounding_filter = grounding_filter  # restrict to these source_ids if set
        self.domain: DomainAdapter = build_domain(cfg.domain.name, cfg.domain.path, cfg.domain.params)

        self.challenger = ChallengerAgent(
            LLMClient(cfg.challenger, role="challenger", timeout_s=cfg.request_timeout_s, max_retries=cfg.max_retries),
            self.domain,
            rubric_max_weight=cfg.acceptance.rubric_max_weight,
            harness=self.harness,
        )
        self.verifier = VerifierJudge(
            LLMClient(cfg.judge, role="judge", timeout_s=cfg.request_timeout_s, max_retries=cfg.max_retries),
            self.domain,
            harness=self.harness,
        )
        self.weak = SolverAgent(
            LLMClient(cfg.weak_solver, role="weak", timeout_s=cfg.request_timeout_s, max_retries=cfg.max_retries),
            self.domain,
            "weak",
            harness=self.harness,
        )
        self.strong = SolverAgent(
            LLMClient(cfg.strong_solver, role="strong", timeout_s=cfg.request_timeout_s, max_retries=cfg.max_retries),
            self.domain,
            "strong",
            harness=self.harness,
        )
        self.reflector = Reflector(
            LLMClient(cfg.orchestrator, role="reflector", timeout_s=cfg.request_timeout_s, max_retries=cfg.max_retries),
            harness=self.harness,
        )

        self.safety: Optional[SafetyFilter] = load_filter(cfg.safety.filter) if cfg.safety.enabled else None

        self.writer = RunWriter(cfg, self.run_id)
        self.writer.snapshot_config()

    # -----------------------------------------------------------------------

    def run(self) -> dict[str, int]:
        items: Iterable[GroundingItem] = self.domain.load_grounding()
        item_list = list(items)
        if self.grounding_filter is not None:
            item_list = [i for i in item_list if i.source_id in self.grounding_filter]
        if not item_list:
            logger.warning("domain {} produced no grounding items", self.domain.name)
            return self.writer._summary

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
                except Exception:
                    logger.exception("unhandled error on source {}", item.source_id)
                progress.advance(t)

        logger.info("run complete: {}", self.writer._summary)
        return self.writer._summary

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
            self.writer.write_trajectory(trajectory)

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

        # all rounds exhausted without acceptance
        last = trajectory.latest_round()
        reasons = (last.evaluation.rejection_reasons if last and last.evaluation else ["no_evaluation"])
        self.writer.write_rejected({
            "trajectory_id": trajectory.trajectory_id,
            "source_id": item.source_id,
            "domain": self.domain.name,
            "reasons": reasons,
            "total_rounds": trajectory.total_rounds,
        })
        logger.info("[{}] exhausted without acceptance: {}", item.source_id, reasons)

    # -----------------------------------------------------------------------

    def _one_round(self, item: GroundingItem, trajectory: Trajectory, round_n: int) -> Round:
        feedback: list[str] = []
        prior_payloads: list[dict] = []
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
            quality = self._struct_failed_quality(struct_errs)
            return Round(
                refinement_round=round_n,
                candidate=candidate,
                quality=quality,
                evaluation=None,
                reflection=" | ".join(feedback) if feedback else None,
                ended_at=datetime.now(timezone.utc),
            )

        # Safety filter (optional)
        if self.safety:
            verdict = self.safety(_concat_payload(candidate.payload))
            if not verdict.allowed:
                quality = self._struct_failed_quality([f"safety:{r}" for r in verdict.reasons])
                return Round(
                    refinement_round=round_n,
                    candidate=candidate,
                    quality=quality,
                    evaluation=None,
                    reflection=" | ".join(feedback) if feedback else None,
                    ended_at=datetime.now(timezone.utc),
                )

        # Quality verifier
        quality = self.verifier.quality_check(candidate)
        if not quality.passed:
            return Round(
                refinement_round=round_n,
                candidate=candidate,
                quality=quality,
                evaluation=None,
                reflection=" | ".join(feedback) if feedback else None,
                ended_at=datetime.now(timezone.utc),
            )

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

        return Round(
            refinement_round=round_n,
            candidate=candidate,
            quality=quality,
            evaluation=ev,
            reflection=" | ".join(feedback) if feedback else None,
            ended_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _struct_failed_quality(errs: list[str]):
        from autodata.schemas import QualityCheck

        return QualityCheck(passed=False, failures=errs, notes="structural/safety validation failed")


def _make_run_id(cfg: RunConfig) -> str:
    return f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{stable_id(cfg.model_dump_json(), length=6)}"


def _concat_payload(payload: dict) -> str:
    return " ".join(str(v) for v in payload.values() if isinstance(v, (str, int, float)))
