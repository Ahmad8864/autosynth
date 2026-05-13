"""MetaOptimizer: the evolutionary loop over HarnessSpecs.

For each iteration: pick a parent by Boltzmann sampling on train score, ask
the mutator LLM for a diff, evaluate the child on train items, and accept
only if validation also improves on the parent.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from loguru import logger

from autosynth.config import RunConfig
from autosynth.domain import build_domain
from autosynth.harness import DEFAULT_HARNESS, HarnessSpec
from autosynth.llm import LLMClient, LLMConfig
from autosynth.metaopt.evaluation import aggregate_failures_from_db, evaluate_harness
from autosynth.metaopt.mutation import Mutator, apply_mutation
from autosynth.metaopt.records import HarnessRecord, MetaIteration
from autosynth.metaopt.selection import boltzmann_select
from autosynth.utils import make_run_id, write_json, write_pydantic, write_yaml_snapshot


class MetaOptimizer:
    def __init__(self, run_cfg: RunConfig, *, seed_harness: HarnessSpec | None = None, rng_seed: int = 0):
        self.cfg = run_cfg
        self.meta = run_cfg.metaopt
        if not self.meta.enabled:
            raise ValueError("RunConfig.metaopt.enabled must be true")

        self.rng = random.Random(rng_seed)
        self.run_id = make_run_id("metaopt", run_cfg.model_dump_json())
        self.root = Path(run_cfg.output_dir) / self.meta.output_subdir / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)

        write_yaml_snapshot(self.root / "config.snapshot.yaml", self.cfg)

        self.seed_harness = seed_harness or self._load_seed()
        # Mutator client; falls back to the orchestrator's model. The mutator
        # should ideally be a stronger reasoning model than the orchestrator
        # (the paper relies on this), so warn loudly when the fallback kicks in.
        if self.meta.mutator is None:
            logger.warning(
                "metaopt.mutator not set; falling back to orchestrator model — "
                "consider configuring a stronger reasoning model for the mutator"
            )
        mutator_cfg = self.meta.mutator or run_cfg.orchestrator
        self.mutator = Mutator(
            LLMClient(
                LLMConfig(
                    max_retries=run_cfg.max_retries,
                    request_timeout_s=run_cfg.request_timeout_s,
                )
            ),
            model_key=mutator_cfg.provider_model,
            temperature=mutator_cfg.temperature,
            max_tokens=mutator_cfg.max_tokens,
        )

        # Compute deterministic train/val split.
        self.train_ids, self.val_ids = self._split_grounding()

        self.population: list[HarnessRecord] = []
        self.iterations: list[MetaIteration] = []
        # Where the most recent training run for each harness lives (used for
        # failure aggregation in the next iteration).
        self._last_train_run_dir: dict[str, Path] = {}

    # ---- main loop ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        # 1. Evaluate the seed
        seed_train_rate, _ = self._eval(self.seed_harness, "train", iter_n=0)
        seed_val_rate, _ = self._eval(self.seed_harness, "val", iter_n=0)
        self.seed_harness.train_score = seed_train_rate
        self.seed_harness.val_score = seed_val_rate
        self.population.append(
            HarnessRecord(
                spec=self.seed_harness,
                train_score=seed_train_rate,
                val_score=seed_val_rate,
                accepted=True,
            )
        )
        self._save_state(iter_n=0)
        logger.info("[meta] seed harness: train={:.3f} val={:.3f}", seed_train_rate, seed_val_rate)

        # 2. Iterate
        for it in range(1, self.meta.max_iterations + 1):
            parent = boltzmann_select(self.population, self.meta.boltzmann_temp, self.rng)
            iter_dir = self.root / "iterations" / f"iter_{it:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            write_pydantic(iter_dir / "parent_harness.json", parent.spec)

            # Collect failures from the parent's most recent training run.
            failure_summary = self._collect_failures(parent.spec)
            write_json(iter_dir / "failure_summary.json", failure_summary)

            mutation = self.mutator.propose(parent.spec, failure_summary)
            write_json(iter_dir / "mutation.json", mutation)

            child = apply_mutation(parent.spec, mutation, iteration=it)
            write_pydantic(iter_dir / "proposed_harness.json", child)

            if child.fingerprint() == parent.spec.fingerprint():
                logger.info("[meta] iter {}: mutation produced identical harness; skipping", it)
                self._record_iter(
                    it,
                    parent,
                    child,
                    mutation,
                    train_score=parent.train_score,
                    val_score=None,
                    accepted=False,
                    reasons=["no_op_mutation"],
                )
                continue

            train_rate, _ = self._eval(child, "train", iter_n=it)
            child.train_score = train_rate

            decision_reasons: list[str] = []
            val_rate: float | None = None
            accepted = False
            if train_rate <= parent.train_score:
                decision_reasons.append(
                    f"train_did_not_improve: child {train_rate:.3f} <= parent {parent.train_score:.3f}"
                )
            else:
                val_rate, _ = self._eval(child, "val", iter_n=it)
                child.val_score = val_rate
                if val_rate is None or val_rate <= (parent.val_score or 0.0):
                    decision_reasons.append(
                        f"val_did_not_improve: child {val_rate} <= parent {parent.val_score}"
                    )
                else:
                    accepted = True
                    decision_reasons.append(
                        f"accepted: train {parent.train_score:.3f}->{train_rate:.3f}, "
                        f"val {parent.val_score:.3f}->{val_rate:.3f}"
                    )

            self._record_iter(it, parent, child, mutation, train_rate, val_rate, accepted, decision_reasons)

            if accepted:
                self.population.append(
                    HarnessRecord(
                        spec=child,
                        train_score=train_rate,
                        val_score=val_rate,
                        accepted=True,
                        parent_accepted_id=parent.spec.harness_id,
                    )
                )
                logger.info(
                    "[meta] iter {}: ACCEPTED child {} (train {:.3f}, val {:.3f})",
                    it,
                    child.harness_id,
                    train_rate,
                    val_rate or 0.0,
                )
            else:
                self.population.append(
                    HarnessRecord(
                        spec=child,
                        train_score=train_rate,
                        val_score=val_rate,
                        accepted=False,
                        parent_accepted_id=parent.spec.harness_id,
                    )
                )
                logger.info("[meta] iter {}: rejected child ({})", it, "; ".join(decision_reasons))

            self._save_state(iter_n=it)

        # Final state
        best = self._best()
        write_pydantic(self.root / "best_harness.json", best.spec)
        return {
            "run_id": self.run_id,
            "iterations": len(self.iterations),
            "population_size": len(self.population),
            "accepted_mutations": sum(1 for r in self.population if r.accepted) - 1,  # -1 for seed
            "best_harness_id": best.spec.harness_id,
            "best_train": best.train_score,
            "best_val": best.val_score,
            "output_dir": str(self.root),
        }

    # ---- helpers -----------------------------------------------------------

    def _load_seed(self) -> HarnessSpec:
        if self.meta.seed_harness_path:
            return HarnessSpec.model_validate_json(Path(self.meta.seed_harness_path).read_text())
        return copy.deepcopy(DEFAULT_HARNESS)

    def _split_grounding(self) -> tuple[list[str], list[str]]:
        domain = build_domain(self.cfg.domain.name, self.cfg.domain.path, self.cfg.domain.params)
        items = list(domain.load_grounding())
        # Deterministic shuffle.
        rng = random.Random(self.meta.val_seed)
        ids = [i.source_id for i in items]
        rng.shuffle(ids)
        train = ids[: self.meta.train_size]
        val = ids[self.meta.train_size : self.meta.train_size + self.meta.val_size]
        if not train or not val:
            raise ValueError(
                f"need at least train_size+val_size={self.meta.train_size + self.meta.val_size} "
                f"grounding items, got {len(ids)}"
            )
        return train, val

    def _eval(self, harness: HarnessSpec, split: str, *, iter_n: int) -> tuple[float, Path]:
        ids = self.train_ids if split == "train" else self.val_ids
        run_id = f"iter_{iter_n:03d}-{split}-{harness.harness_id}"
        split_root = self.root / "iterations" / f"iter_{iter_n:03d}" / split
        rate, run_root = evaluate_harness(self.cfg, harness, ids, run_id=run_id, output_dir=split_root)
        if split == "train":
            self._last_train_run_dir[harness.harness_id] = run_root
        return rate, run_root

    def _collect_failures(self, parent: HarnessSpec) -> dict[str, Any]:
        run_dir = self._last_train_run_dir.get(parent.harness_id)
        if run_dir is None:
            return {
                "note": "no prior training run",
                "total_rounds": 0,
                "rejected_rounds": 0,
                "reason_counts": {},
                "samples": [],
            }
        return aggregate_failures_from_db(run_dir / "run.db")

    def _record_iter(
        self,
        it: int,
        parent: HarnessRecord,
        child: HarnessSpec,
        mutation: dict[str, Any],
        train_score: float,
        val_score: float | None,
        accepted: bool,
        reasons: list[str],
    ) -> None:
        rec = MetaIteration(
            iteration=it,
            parent_id=parent.spec.harness_id,
            child_id=child.harness_id,
            mutation=mutation,
            train_score=train_score,
            val_score=val_score,
            accepted=accepted,
            reasons=reasons,
        )
        self.iterations.append(rec)
        write_pydantic(self.root / "iterations" / f"iter_{it:03d}" / "decision.json", rec)

    def _save_state(self, iter_n: int) -> None:
        write_pydantic(self.root / "population.json", self.population)
        write_pydantic(self.root / "iterations_log.json", self.iterations)

    def _best(self) -> HarnessRecord:
        accepted = [r for r in self.population if r.accepted]
        return max(accepted, key=lambda r: (r.val_score or 0.0, r.train_score))
