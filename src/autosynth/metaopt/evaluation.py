"""Evaluate a HarnessSpec by running it on a slice of grounding items.

``evaluate_harness`` drives an inner run via :class:`Runner` and returns the
acceptance rate; ``aggregate_failures_from_db`` aggregates rejection reasons
from the inner run's ``run.db`` so the mutator can address concrete failure
modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autosynth.config import RunConfig
from autosynth.harness import HarnessSpec
from autosynth.runner import Runner
from autosynth.store import Store


def evaluate_harness(
    run_cfg: RunConfig,
    harness: HarnessSpec,
    source_ids: list[str],
    *,
    run_id: str,
    output_dir: Path,
) -> tuple[float, Path]:
    """Run with this harness on the given source items via Runner.

    Returns (acceptance_rate, run_dir). acceptance_rate = accepted / N.
    """
    cfg = run_cfg.model_copy(deep=True)
    cfg.output_dir = str(output_dir)
    cfg.run_id = run_id
    cfg.resume = False
    cfg.max_examples = max(len(source_ids), 1)
    cfg.loop.max_rounds = min(cfg.loop.max_rounds, cfg.metaopt.inner_max_rounds)

    runner = Runner(cfg, run_id=run_id, harness=harness, source_id_filter=set(source_ids))
    summary = runner.run()
    rate = summary.accepted / max(len(source_ids), 1)
    return rate, runner.run_dir


def aggregate_failures_from_db(run_db_path: Path, *, sample_size: int = 3) -> dict[str, Any]:
    """Compute the failure histogram for an inner run from its run.db."""
    empty = {"total_rounds": 0, "rejected_rounds": 0, "reason_counts": {}, "samples": []}
    if not run_db_path.exists():
        return empty
    store = Store(run_db_path)
    try:
        run_row = store.first_run()
        if run_row is None:
            return empty
        reason_counts: dict[str, int] = {}
        samples: list[dict[str, Any]] = []
        total = 0
        rejected = 0
        for row in store.failure_rounds(run_row["run_id"]):
            total += 1
            quality = json.loads(row["quality_blob"]) if row["quality_blob"] else None
            ev = json.loads(row["eval_blob"]) if row["eval_blob"] else None
            if quality and not quality.get("passed", False):
                rejected += 1
                for f in quality.get("failures") or []:
                    key = f"quality:{str(f).split(':')[0]}"
                    reason_counts[key] = reason_counts.get(key, 0) + 1
                if len(samples) < sample_size:
                    samples.append({"reason": "quality_failed", "failures": quality.get("failures") or []})
                continue
            if ev is None or ev.get("accepted"):
                continue
            rejected += 1
            for r in ev.get("rejection_reasons") or []:
                key = str(r).split(" ")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
            if len(samples) < sample_size:
                samples.append(
                    {
                        "reason": "; ".join(ev.get("rejection_reasons") or []),
                        "weak_avg": ev.get("weak_avg"),
                        "strong_avg": ev.get("strong_avg"),
                        "gap": ev.get("gap"),
                    }
                )
        return {
            "total_rounds": total,
            "rejected_rounds": rejected,
            "reason_counts": reason_counts,
            "samples": samples,
        }
    finally:
        store.close()
