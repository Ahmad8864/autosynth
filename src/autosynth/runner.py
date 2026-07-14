"""Wire configuration, storage, dispatch, and pipeline state into a run."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from autosynth.config import ModelConfig, RunConfig
from autosynth.dispatcher import (
    AnthropicBatchProvider,
    BatchProvider,
    Dispatcher,
    LiteLLMBatchProvider,
    MockBatchProvider,
    RunSummary,
    fulfill_local,
    make_fulfill_batch,
    poll_outstanding_batches,
)
from autosynth.domain import DomainAdapter, GroundingItem, build_domain
from autosynth.harness import DEFAULT_HARNESS, HarnessSpec
from autosynth.llm import LLMClient, LLMConfig, RateLimitSpec
from autosynth.store import ITEM_PENDING, Store
from autosynth.utils import make_run_id, write_yaml_snapshot


class Runner:
    def __init__(
        self,
        cfg: RunConfig,
        *,
        run_id: str | None = None,
        harness: HarnessSpec | None = None,
        source_id_filter: set[str] | None = None,
    ):
        self.cfg = cfg
        self.harness = harness or DEFAULT_HARNESS
        self.source_id_filter = source_id_filter
        self._cfg_json = cfg.model_dump(mode="json")
        self.run_id = run_id or cfg.run_id or make_run_id("run", cfg.model_dump_json())

        self.run_dir = Path(cfg.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.run_dir / "run.db")

        self.domain: DomainAdapter = build_domain(cfg.domain.name, cfg.domain.path, cfg.domain.params)
        self.llm = LLMClient(_build_llm_config(cfg))

    def run(self) -> RunSummary:
        # Meta-optimization creates many runners, so each run must close its DB.
        try:
            write_yaml_snapshot(self.run_dir / "config.snapshot.yaml", self.cfg)
            self._ensure_run_row()
            grounding = self._seed_items()
            if not grounding:
                logger.warning("no grounding items; run is empty")
                return RunSummary(
                    run_id=self.run_id,
                    accepted=0,
                    rejected=0,
                    state_counts={},
                    cost_usd=0.0,
                )

            fulfill, poll_in_flight = self._fulfillment_strategy()
            dispatcher = Dispatcher(
                store=self.store,
                llm=self.llm,
                domain=self.domain,
                cfg=self.cfg,
                run_id=self.run_id,
                harness=self.harness,
                grounding=grounding,
                fulfill=fulfill,
                poll_in_flight=poll_in_flight,
            )
            return dispatcher.run()
        finally:
            self.store.close()

    def _fulfillment_strategy(self):
        """Select local or batch fulfillment hooks."""
        if self.cfg.dispatcher.mode != "batch":
            return fulfill_local, None
        provider = self._batch_provider()

        def poll_in_flight(dispatcher):
            return poll_outstanding_batches(provider, dispatcher)

        return make_fulfill_batch(provider), poll_in_flight

    def _batch_provider(self) -> BatchProvider:
        """Build the configured batch provider."""
        name = self.cfg.dispatcher.batch_provider
        if name == "mock":
            return MockBatchProvider(self.llm)
        if name == "anthropic":
            return AnthropicBatchProvider()
        return LiteLLMBatchProvider(
            provider=name,
            completion_window=self.cfg.dispatcher.batch_completion_window,
        )

    def close(self) -> None:
        """Close the store connection. Safe to call more than once."""
        self.store.close()

    def _ensure_run_row(self) -> None:
        if self.store.get_run(self.run_id) is None:
            self.store.create_run(
                self.run_id,
                config=self._cfg_json,
                harness=self.harness.model_dump(),
                cost_usd_cap=self.cfg.budget_usd,
            )

    def _seed_items(self) -> dict[str, GroundingItem]:
        grounding: dict[str, GroundingItem] = {}
        for item in self.domain.load_grounding():
            if self.source_id_filter is not None and item.source_id not in self.source_id_filter:
                continue
            if len(grounding) >= self.cfg.max_examples:
                break
            grounding[item.source_id] = item
            self.store.insert_item(
                run_id=self.run_id,
                source_id=item.source_id,
                domain=self.domain.name,
                state=ITEM_PENDING,
                source_metadata=item.metadata,
            )
        return grounding


def _build_llm_config(cfg: RunConfig) -> LLMConfig:
    """Build client-level settings from the run configuration."""
    return LLMConfig(
        max_retries=cfg.max_retries,
        request_timeout_s=cfg.request_timeout_s,
        rate_limits={"mock/*": RateLimitSpec(rpm=None)},
        model_extras=_collect_model_extras(cfg),
    )


def _collect_model_extras(cfg: RunConfig) -> dict[str, dict]:
    """Group per-role provider options by model name."""
    roles: list[ModelConfig | None] = [
        cfg.orchestrator,
        cfg.challenger,
        cfg.weak_solver,
        cfg.strong_solver,
        cfg.judge,
        cfg.metaopt.mutator,
    ]
    out: dict[str, dict] = {}
    for role_cfg in roles:
        if role_cfg is None or not role_cfg.extra:
            continue
        bucket = out.setdefault(role_cfg.provider_model, {})
        bucket.update(role_cfg.extra)
    return out
