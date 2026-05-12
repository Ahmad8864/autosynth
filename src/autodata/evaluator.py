"""Acceptance criteria evaluation.

Pure function: given a list of solver scores and the configured thresholds,
populate an EvalReport with weak/strong stats, gap, accept/reject, and
human-readable reasons.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from autodata.config import AcceptanceConfig
from autodata.schemas import EvalReport, QualityCheck, SolverScore


def evaluate(
    weak_scores: Sequence[SolverScore],
    strong_scores: Sequence[SolverScore],
    quality: QualityCheck,
    criteria: AcceptanceConfig,
) -> EvalReport:
    weak_vals = [s.total for s in weak_scores] or [0.0]
    strong_vals = [s.total for s in strong_scores] or [0.0]

    report = EvalReport(
        weak_scores=list(weak_scores),
        strong_scores=list(strong_scores),
        weak_avg=mean(weak_vals),
        weak_max=max(weak_vals),
        weak_min=min(weak_vals),
        strong_avg=mean(strong_vals),
        strong_max=max(strong_vals),
        strong_min=min(strong_vals),
    )
    report.gap = report.strong_avg - report.weak_avg

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
