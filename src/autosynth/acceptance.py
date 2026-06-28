"""Acceptance policies: turn solver scores into an accept/reject EvalReport.

Two regimes from the paper:

- :class:`ThresholdPolicy` (§3.1/3.2) — rubric-gap acceptance. An LLM judge
  scores each rollout; accept on weak/strong thresholds + gap.
- :class:`VerifiablePolicy` (§3.3) — count gate over programmatic correctness.
  Each rollout is scored by the domain's ``verify()``; accept when the weak
  solver mostly fails and the strong solver mostly succeeds.

``requires_judge`` tells the pipeline how per-attempt scores are produced:
``True`` dispatches a judge LLM request per attempt; ``False`` scores each
attempt in-process via ``domain.verify()``. ``resolve_policy`` picks the policy
from ``cfg.acceptance.mode`` (falling back to the domain's default).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from statistics import mean

from loguru import logger

from autosynth.config import AcceptanceConfig, RunConfig
from autosynth.domain import DomainAdapter
from autosynth.schemas import EvalReport, QualityCheck, SolverScore


def _base_report(weak: Sequence[SolverScore], strong: Sequence[SolverScore]) -> EvalReport:
    """EvalReport with the weak/strong stats populated, no decision yet."""
    weak_vals = [s.total for s in weak] or [0.0]
    strong_vals = [s.total for s in strong] or [0.0]
    report = EvalReport(
        weak_scores=list(weak),
        strong_scores=list(strong),
        weak_avg=mean(weak_vals),
        weak_max=max(weak_vals),
        weak_min=min(weak_vals),
        strong_avg=mean(strong_vals),
        strong_max=max(strong_vals),
        strong_min=min(strong_vals),
    )
    report.gap = report.strong_avg - report.weak_avg
    return report


def evaluate(
    weak_scores: Sequence[SolverScore],
    strong_scores: Sequence[SolverScore],
    quality: QualityCheck,
    criteria: AcceptanceConfig,
) -> EvalReport:
    """Rubric-gap acceptance: weak/strong thresholds + gap (paper §3.1/3.2)."""
    report = _base_report(weak_scores, strong_scores)

    reasons: list[str] = []
    if criteria.require_quality_passed and not quality.passed:
        reasons.append(f"quality_failed: {', '.join(quality.failures) or 'unspecified'}")
    if report.weak_avg > criteria.weak_avg_max:
        reasons.append(f"weak_avg {report.weak_avg:.3f} > {criteria.weak_avg_max}")
    if report.weak_max > criteria.weak_max_max:
        reasons.append(f"weak_max {report.weak_max:.3f} > {criteria.weak_max_max}")
    if criteria.forbid_weak_zero and any(s.total == 0.0 for s in weak_scores):
        reasons.append("forbid_weak_zero: at least one weak score was 0.0")
    if report.strong_avg < criteria.strong_avg_min:
        reasons.append(f"strong_avg {report.strong_avg:.3f} < {criteria.strong_avg_min}")
    if report.strong_avg >= criteria.strong_avg_max:
        reasons.append(f"strong_avg {report.strong_avg:.3f} >= ceiling {criteria.strong_avg_max}")
    if report.gap < criteria.min_gap:
        reasons.append(f"gap {report.gap:.3f} < min {criteria.min_gap}")

    report.accepted = not reasons
    report.rejection_reasons = reasons
    if report.accepted:
        report.acceptance_rationale = (
            f"weak_avg={report.weak_avg:.3f}, strong_avg={report.strong_avg:.3f}, "
            f"gap={report.gap:.3f}; quality passed; thresholds met."
        )
    return report


class AcceptancePolicy(ABC):
    """Strategy that scores-and-decides a round's acceptance.

    ``requires_judge``: True -> pipeline dispatches a judge per attempt; False
    -> pipeline scores each attempt via ``domain.verify()``.
    ``weak_ceiling`` / ``strong_floor``: rates the reflector uses to bucket
    prior rounds (weak above the ceiling = "too easy"; strong below the floor
    = "failed_strong").
    """

    requires_judge: bool = True
    weak_ceiling: float
    strong_floor: float

    @abstractmethod
    def evaluate(
        self,
        weak_scores: Sequence[SolverScore],
        strong_scores: Sequence[SolverScore],
        quality: QualityCheck,
    ) -> EvalReport: ...


class ThresholdPolicy(AcceptancePolicy):
    """Rubric-gap acceptance (mode="rubric")."""

    requires_judge = True

    def __init__(self, criteria: AcceptanceConfig):
        self.criteria = criteria
        self.weak_ceiling = criteria.weak_avg_max
        self.strong_floor = criteria.strong_avg_min

    def evaluate(self, weak_scores, strong_scores, quality) -> EvalReport:
        return evaluate(weak_scores, strong_scores, quality, self.criteria)


class VerifiablePolicy(AcceptancePolicy):
    """Count gate over per-attempt correctness (mode="verifiable", paper §3.3).

    Accept when at most ``verifiable_weak_max_correct`` weak rollouts and at
    least ``verifiable_strong_min_correct`` strong rollouts are correct. A
    rollout is correct when its (binary) ``total`` is 1.0; the gate reads
    ``total`` rather than ``correct`` so it is robust when ``correct`` is absent.
    """

    requires_judge = False

    def __init__(self, criteria: AcceptanceConfig, weak_samples: int, strong_samples: int):
        self.criteria = criteria
        self.weak_ceiling = criteria.verifiable_weak_max_correct / max(weak_samples, 1)
        self.strong_floor = criteria.verifiable_strong_min_correct / max(strong_samples, 1)

    def evaluate(self, weak_scores, strong_scores, quality) -> EvalReport:
        report = _base_report(weak_scores, strong_scores)
        weak_correct = sum(1 for s in weak_scores if s.total >= 1.0)
        strong_correct = sum(1 for s in strong_scores if s.total >= 1.0)

        reasons: list[str] = []
        if self.criteria.require_quality_passed and not quality.passed:
            reasons.append(f"quality_failed: {', '.join(quality.failures) or 'unspecified'}")
        if weak_correct > self.criteria.verifiable_weak_max_correct:
            reasons.append(
                f"weak_correct {weak_correct} > {self.criteria.verifiable_weak_max_correct} (too easy)"
            )
        if strong_correct < self.criteria.verifiable_strong_min_correct:
            reasons.append(
                f"strong_correct {strong_correct} < {self.criteria.verifiable_strong_min_correct} (too hard)"
            )

        report.accepted = not reasons
        report.rejection_reasons = reasons
        if report.accepted:
            report.acceptance_rationale = (
                f"weak_correct={weak_correct}/{len(weak_scores)}, "
                f"strong_correct={strong_correct}/{len(strong_scores)}; quality passed; verifiable gate met."
            )
        return report


def resolve_policy(cfg: RunConfig, domain: DomainAdapter) -> AcceptancePolicy:
    """Pick the acceptance policy from config + domain.

    ``cfg.acceptance.mode`` wins; otherwise the domain's
    ``default_acceptance_mode``. Verifiable mode requires the domain to
    implement ``verify()`` and a well-formed count gate.
    """
    mode = cfg.acceptance.mode or getattr(domain, "default_acceptance_mode", "rubric")
    if mode == "rubric":
        return ThresholdPolicy(cfg.acceptance)
    if mode == "verifiable":
        if type(domain).verify is DomainAdapter.verify:
            raise ValueError(
                f"domain {domain.name!r} does not implement verify(); verifiable acceptance "
                "mode requires a programmatic verifier (override DomainAdapter.verify)"
            )
        _validate_verifiable(cfg)
        return VerifiablePolicy(cfg.acceptance, cfg.loop.weak_samples, cfg.loop.strong_samples)
    raise ValueError(f"unknown acceptance mode {mode!r}")


def _validate_verifiable(cfg: RunConfig) -> None:
    """Reject unsatisfiable / vacuous count gates; warn on forced perfection."""
    a, loop = cfg.acceptance, cfg.loop
    if a.verifiable_strong_min_correct > loop.strong_samples:
        raise ValueError(
            f"verifiable_strong_min_correct ({a.verifiable_strong_min_correct}) > "
            f"strong_samples ({loop.strong_samples}): gate is unsatisfiable"
        )
    if a.verifiable_weak_max_correct >= loop.weak_samples:
        raise ValueError(
            f"verifiable_weak_max_correct ({a.verifiable_weak_max_correct}) >= "
            f"weak_samples ({loop.weak_samples}): weak gate always passes"
        )
    if a.verifiable_weak_max_correct >= a.verifiable_strong_min_correct:
        raise ValueError(
            f"verifiable_weak_max_correct ({a.verifiable_weak_max_correct}) >= "
            f"verifiable_strong_min_correct ({a.verifiable_strong_min_correct}): no separation band"
        )
    if a.verifiable_strong_min_correct == loop.strong_samples:
        logger.warning(
            "verifiable_strong_min_correct == strong_samples ({}): strong must solve ALL rollouts; "
            "the default gate is calibrated for the paper's 4 samples",
            loop.strong_samples,
        )
