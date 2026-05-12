"""Dataset and trajectory writers.

Layout under `output_dir/<run_id>/`:
  config.snapshot.yaml         # frozen run config
  accepted.jsonl               # final dataset
  rejected.jsonl               # rejected candidates with reasons
  trajectories/<source_id>.json  # full per-source-item history
  summary.json                 # run-level metrics (updated incrementally)
  hf_export/                   # optional Hugging Face datasets dir
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from autodata.config import RunConfig
from autodata.domain import DomainAdapter
from autodata.schemas import Trajectory
from autodata.utils import append_jsonl, read_jsonl, write_pydantic


class RunWriter:
    def __init__(self, cfg: RunConfig, run_id: str):
        self.cfg = cfg
        self.run_id = run_id
        self.root = Path(cfg.output_dir) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.accepted_path = self.root / "accepted.jsonl"
        self.rejected_path = self.root / "rejected.jsonl"
        self.summary_path = self.root / "summary.json"
        self.trajectories_dir = self.root / "trajectories"
        self.trajectories_dir.mkdir(exist_ok=True)
        self._summary: dict[str, Any] = self._load_summary()

    @property
    def summary(self) -> dict[str, Any]:
        """Snapshot of run-level counters. Returns a copy."""
        return dict(self._summary)

    def snapshot_config(self) -> None:
        path = self.root / "config.snapshot.yaml"
        path.write_text(yaml.safe_dump(self.cfg.model_dump(mode="json"), sort_keys=False))

    def trajectory_path(self, source_id: str) -> Path:
        return self.trajectories_dir / f"{source_id}.json"

    def load_trajectory(self, source_id: str) -> Trajectory | None:
        path = self.trajectory_path(source_id)
        try:
            text = path.read_text()
        except FileNotFoundError:
            return None
        try:
            return Trajectory.model_validate_json(text)
        except Exception as e:
            logger.warning("could not parse existing trajectory {}: {}", path, e)
            return None

    def write_trajectory(self, trajectory: Trajectory) -> None:
        write_pydantic(self.trajectory_path(trajectory.source_id), trajectory)

    def write_accepted(self, record: dict[str, Any]) -> None:
        append_jsonl(self.accepted_path, record)
        self._bump("accepted")

    def write_rejected(self, record: dict[str, Any]) -> None:
        append_jsonl(self.rejected_path, record)
        self._bump("rejected")

    def bump(self, key: str) -> None:
        """Increment a named counter on the summary (e.g., 'errors')."""
        self._bump(key)

    def _bump(self, key: str) -> None:
        self._summary[key] = int(self._summary.get(key, 0)) + 1
        self._flush_summary()

    def update_summary(self, **kv: Any) -> None:
        self._summary.update(kv)
        self._flush_summary()

    def _flush_summary(self) -> None:
        self.summary_path.write_text(json.dumps(self._summary, indent=2, default=str))

    def _load_summary(self) -> dict[str, Any]:
        try:
            text = self.summary_path.read_text()
        except FileNotFoundError:
            return {"run_id": self.run_id, "accepted": 0, "rejected": 0}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"run_id": self.run_id, "accepted": 0, "rejected": 0}

    # ---- HF export ----------------------------------------------------------

    def export_hf(self) -> Path | None:
        try:
            from datasets import Dataset  # type: ignore
        except ImportError:
            logger.warning("`datasets` not installed; skip HF export. `pip install autodata[hf]`")
            return None
        if not self.accepted_path.exists():
            logger.warning("no accepted records to export")
            return None
        records = list(read_jsonl(self.accepted_path))
        if not records:
            logger.warning("no accepted records to export")
            return None
        out = self.root / "hf_export"
        ds = Dataset.from_list(records)
        ds.save_to_disk(str(out))
        return out


def build_accepted_record(
    *,
    domain: DomainAdapter,
    trajectory: Trajectory,
    extra: dict[str, Any],
) -> dict[str, Any]:
    r = trajectory.accepted_round()
    if r is None:
        raise ValueError(
            f"trajectory {trajectory.trajectory_id} has no accepted round; "
            "build_accepted_record must only be called after acceptance."
        )
    ev = r.evaluation
    return domain.format_accepted(
        r.candidate,
        extra={
            "trajectory_id": trajectory.trajectory_id,
            "run_id": trajectory.run_id,
            "refinement_round": r.refinement_round,
            "weak_avg": ev.weak_avg if ev else None,
            "strong_avg": ev.strong_avg if ev else None,
            "gap": ev.gap if ev else None,
            "weak_scores": [s.model_dump() for s in (ev.weak_scores if ev else [])],
            "strong_scores": [s.model_dump() for s in (ev.strong_scores if ev else [])],
            "acceptance_rationale": ev.acceptance_rationale if ev else None,
            **(extra or {}),
        },
    )
