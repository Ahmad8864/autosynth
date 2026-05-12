"""Meta-optimization loop.

Implements the paper's secondary loop: evolve the orchestrator's instruction
set (the `HarnessSpec`) by selecting a parent via Boltzmann sampling, asking
a mutator LLM to propose an edit informed by failure trajectories, evaluating
the child on training items, and accepting the mutation only if the
validation score on held-out items exceeds the parent's.

Mutations operate on `HarnessSpec` (text rules + a few numeric knobs), not on
Python source. The expressive scope for prompt edits is preserved; the safety
profile is dramatically smaller than letting an LLM rewrite the repo.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from autodata.config import RunConfig
from autodata.domain import build_domain
from autodata.harness import DEFAULT_HARNESS, HarnessSpec, make_harness
from autodata.llm import LLMClient, LLMConfig, LLMRequest
from autodata.runner import Runner
from autodata.store import Store
from autodata.utils import (
    make_run_id,
    stable_id,
    write_json,
    write_pydantic,
    write_yaml_snapshot,
)

# ---------------------------------------------------------------------------
# Population / record types
# ---------------------------------------------------------------------------


class HarnessRecord(BaseModel):
    spec: HarnessSpec
    train_score: float = 0.0
    val_score: float | None = None
    accepted: bool = True  # seed is always accepted
    parent_accepted_id: str | None = None


class MetaIteration(BaseModel):
    iteration: int
    parent_id: str
    child_id: str
    mutation: dict[str, Any] = Field(default_factory=dict)
    train_score: float
    val_score: float | None = None
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Boltzmann selection
# ---------------------------------------------------------------------------


def boltzmann_select(records: list[HarnessRecord], temperature: float, rng: random.Random) -> HarnessRecord:
    """Select a parent record proportional to exp(train_score / T)."""
    accepted = [r for r in records if r.accepted]
    if not accepted:
        accepted = records
    if len(accepted) == 1:
        return accepted[0]
    T = max(temperature, 1e-6)
    scores = [r.train_score for r in accepted]
    max_s = max(scores)
    # Subtract max for numerical stability; same softmax up to constant.
    weights = [math.exp((s - max_s) / T) for s in scores]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    r = rng.random()
    cum = 0.0
    for rec, w in zip(accepted, weights, strict=True):
        cum += w
        if r <= cum:
            return rec
    return accepted[-1]


# ---------------------------------------------------------------------------
# Mutator: LLM proposes an edit to the harness
# ---------------------------------------------------------------------------


class Mutator:
    SYSTEM = (
        "ROLE:META_MUTATOR. You optimize an agentic synthetic-data harness by editing its rule set.\n"
        "Given the current HarnessSpec (as JSON) and a failure summary from the previous evaluation, propose ONE\n"
        "small mutation that addresses the most common failure mode. Mutations are structured edits to rule lists\n"
        "and structural knobs — you do NOT emit code.\n"
        "Return STRICT JSON with these keys (every key optional except `rationale`):\n"
        "{\n"
        '  "rationale": "one or two sentences",\n'
        '  "challenger_rules_add": [strings],\n'
        '  "challenger_rules_remove_indices": [ints],\n'
        '  "quality_rules_add": [strings],\n'
        '  "quality_rules_remove_indices": [ints],\n'
        '  "judge_rules_add": [strings],\n'
        '  "judge_rules_remove_indices": [ints],\n'
        '  "weak_solver_rules_add": [strings],\n'
        '  "weak_solver_rules_remove_indices": [ints],\n'
        '  "strong_solver_rules_add": [strings],\n'
        '  "strong_solver_rules_remove_indices": [ints],\n'
        '  "reflector_rules_add": [strings],\n'
        '  "reflector_rules_remove_indices": [ints],\n'
        '  "rubric_max_weight": int 1..10,\n'
        '  "require_self_test": bool\n'
        "}\n"
        "Add NEW rules, not paraphrases of existing ones. Keep rule strings imperative and specific."
    )

    def __init__(self, llm: LLMClient, model_key: str):
        self.llm = llm
        self.model_key = model_key

    def propose(self, parent: HarnessSpec, failure_summary: dict[str, Any]) -> dict[str, Any]:
        user = json.dumps(
            {
                "current_harness": parent.model_dump(
                    exclude={"harness_id", "parent_id", "iteration", "train_score", "val_score"}
                ),
                "failure_summary": failure_summary,
            },
            indent=2,
        )
        request_id = stable_id("metaopt-mutator", parent.harness_id,
                               parent.iteration, json.dumps(failure_summary)[:200])
        request = LLMRequest(
            request_id=request_id,
            item_id="metaopt",
            round_n=parent.iteration,
            role="meta_mutator",
            model_key=self.model_key,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user},
            ],
            json_mode=True,
        )
        try:
            resp = self.llm.complete(request)
            return resp.parse_json()
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("mutator parse failure: {}", e)
            return {"rationale": "mutator parse error; no mutation applied"}
        except Exception as e:
            logger.warning("mutator call failed: {}", e)
            return {"rationale": f"mutator error: {e}; no mutation applied"}


# ---------------------------------------------------------------------------
# Applying a mutation
# ---------------------------------------------------------------------------

_RULE_ROLES = ("challenger", "quality", "judge", "weak_solver", "strong_solver", "reflector")


def apply_mutation(parent: HarnessSpec, mutation: dict[str, Any], *, iteration: int) -> HarnessSpec:
    """Apply an add/remove diff and produce a child HarnessSpec.

    Out-of-range indices, wrong types, and unknown keys are silently dropped —
    we never want a malformed mutator response to crash the loop.
    """
    fields = parent.model_dump()

    for role in _RULE_ROLES:
        key = f"{role}_rules"
        rules: list[str] = list(fields.get(key, []) or [])

        rm = mutation.get(f"{role}_rules_remove_indices") or []
        if isinstance(rm, list):
            rm_set = {int(i) for i in rm if isinstance(i, int) and 0 <= int(i) < len(rules)}
            rules = [r for i, r in enumerate(rules) if i not in rm_set]

        add = mutation.get(f"{role}_rules_add") or []
        if isinstance(add, list):
            for s in add:
                if isinstance(s, str) and s.strip():
                    rules.append(s.strip())

        fields[key] = rules

    rmw = mutation.get("rubric_max_weight")
    if isinstance(rmw, int) and 1 <= rmw <= 10:
        fields["rubric_max_weight"] = rmw

    rst = mutation.get("require_self_test")
    if isinstance(rst, bool):
        fields["require_self_test"] = rst

    rationale = str(mutation.get("rationale") or "").strip()

    # Strip non-mutable / inherited fields before re-constructing.
    for k in ("harness_id", "parent_id", "iteration", "rationale", "train_score", "val_score"):
        fields.pop(k, None)

    return make_harness(
        parent_id=parent.harness_id,
        iteration=iteration,
        rationale=rationale or "no rationale",
        **fields,
    )


# ---------------------------------------------------------------------------
# Evaluation: run the orchestrator with a harness and compute a score
# ---------------------------------------------------------------------------


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
                    samples.append(
                        {"reason": "quality_failed", "failures": quality.get("failures") or []}
                    )
                continue
            if ev is None or ev.get("accepted"):
                continue
            rejected += 1
            for r in ev.get("rejection_reasons") or []:
                key = str(r).split(" ")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
            if len(samples) < sample_size:
                samples.append({
                    "reason": "; ".join(ev.get("rejection_reasons") or []),
                    "weak_avg": ev.get("weak_avg"),
                    "strong_avg": ev.get("strong_avg"),
                    "gap": ev.get("gap"),
                })
        return {
            "total_rounds": total,
            "rejected_rounds": rejected,
            "reason_counts": reason_counts,
            "samples": samples,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The MetaOptimizer
# ---------------------------------------------------------------------------


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
            LLMClient(LLMConfig(
                max_retries=run_cfg.max_retries,
                request_timeout_s=run_cfg.request_timeout_s,
            )),
            model_key=mutator_cfg.provider_model,
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
