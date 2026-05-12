"""Runner: thin driver that wires Store + Dispatcher + Pipeline.

This replaces the legacy ``Orchestrator``. It:
  - opens or creates a ``run.db`` under ``output_dir/<run_id>/``
  - seeds PENDING items from the domain (unless resuming)
  - builds the LLMClient + Dispatcher
  - calls ``dispatcher.run()`` and returns the summary

Each Runner instance is one logical run. Resume is just "open the same
run.db" (cfg.resume=True).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from autodata.config import RunConfig
from autodata.dispatcher import Dispatcher, RunSummary
from autodata.domain import DomainAdapter, GroundingItem, build_domain
from autodata.harness import DEFAULT_HARNESS, HarnessSpec
from autodata.llm import LLMClient, LLMConfig, RateLimitSpec
from autodata.pipeline import State
from autodata.store import Store
from autodata.utils import stable_id


def _make_run_id(cfg: RunConfig) -> str:
    return (f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            f"-{stable_id(cfg.model_dump_json(), length=6)}")


class Runner:
    def __init__(
        self,
        cfg: RunConfig,
        *,
        run_id: Optional[str] = None,
        harness: Optional[HarnessSpec] = None,
        source_id_filter: Optional[set[str]] = None,
    ):
        self.cfg = cfg
        self.run_id = run_id or cfg.run_id or _make_run_id(cfg)
        self.harness = harness or DEFAULT_HARNESS
        self.source_id_filter = source_id_filter

        self.run_dir = Path(cfg.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.run_dir / "run.db")

        self.domain: DomainAdapter = build_domain(
            cfg.domain.name, cfg.domain.path, cfg.domain.params
        )

        # Build the LLMClient from cfg. We don't expose a top-level LLMConfig in
        # RunConfig yet; build a sensible default and let users override later.
        llm_cfg = _build_llm_config(cfg)
        self.llm = LLMClient(llm_cfg)

    def run(self) -> RunSummary:
        self._snapshot_config()
        self._ensure_run_row()
        grounding = self._seed_items()
        if not grounding:
            logger.warning("no grounding items; run is empty")
            return RunSummary(run_id=self.run_id, accepted=0, rejected=0,
                              state_counts={}, cost_usd=0.0)

        dispatcher = Dispatcher(
            store=self.store, llm=self.llm, domain=self.domain,
            cfg=self.cfg, run_id=self.run_id, harness=self.harness,
            grounding=grounding,
        )
        return dispatcher.run()

    # ------------------------------------------------------------------

    def _snapshot_config(self) -> None:
        path = self.run_dir / "config.snapshot.yaml"
        path.write_text(yaml.safe_dump(self.cfg.model_dump(mode="json"), sort_keys=False))

    def _ensure_run_row(self) -> None:
        if self.store.get_run(self.run_id) is None:
            self.store.create_run(
                self.run_id, config=self.cfg.model_dump(mode="json"),
                harness=self.harness.model_dump(),
                cost_usd_cap=self.cfg.budget_usd,
            )

    def _seed_items(self) -> dict[str, GroundingItem]:
        grounding: dict[str, GroundingItem] = {}
        for item in self.domain.load_grounding():
            if (self.source_id_filter is not None
                    and item.source_id not in self.source_id_filter):
                continue
            if len(grounding) >= self.cfg.max_examples:
                break
            grounding[item.source_id] = item
            self.store.insert_item(
                run_id=self.run_id, source_id=item.source_id,
                domain=self.domain.name, state=State.PENDING.value,
                source_metadata=item.metadata,
            )
        return grounding


def _build_llm_config(cfg: RunConfig) -> LLMConfig:
    """Derive an LLMConfig from the existing RunConfig fields.

    Until we add a first-class top-level ``llm:`` YAML block, we infer
    sensible defaults from per-role ModelConfigs and the existing global
    knobs (request_timeout_s, max_retries).
    """
    return LLMConfig(
        max_retries=cfg.max_retries,
        request_timeout_s=cfg.request_timeout_s,
        rate_limits={
            # Conservative defaults — users override in their config later.
            "mock/*": RateLimitSpec(rpm=None),
        },
        default_temperature=cfg.challenger.temperature,
        default_max_tokens=cfg.challenger.max_tokens,
    )
