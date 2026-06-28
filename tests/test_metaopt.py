"""Tests for the meta-optimization loop."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from autosynth.config import (
    AcceptanceConfig,
    DomainConfig,
    LoopConfig,
    MetaOptConfig,
    ModelConfig,
    RunConfig,
)
from autosynth.harness import make_harness
from autosynth.metaopt import (
    HarnessRecord,
    MetaOptimizer,
    apply_mutation,
    boltzmann_select,
)

# ---------------------------------------------------------------------------
# Boltzmann selection
# ---------------------------------------------------------------------------


def _record(score: float, hid: str = "h") -> HarnessRecord:
    spec = make_harness(challenger_rules=[hid])
    spec.harness_id = hid
    return HarnessRecord(spec=spec, train_score=score, val_scores=[score], accepted=True)


def test_val_mean_averages_scores_else_falls_back():
    assert _record(0.0).model_copy(update={"val_scores": [0.2, 0.4]}).val_mean == pytest.approx(0.3)
    assert HarnessRecord(spec=make_harness(), val_scores=[], val_score=0.7).val_mean == 0.7
    assert HarnessRecord(spec=make_harness(), val_scores=[]).val_mean is None


def test_boltzmann_selects_higher_at_low_temp():
    pop = [_record(0.1, "low"), _record(0.9, "high")]
    rng = random.Random(0)
    counts = {"low": 0, "high": 0}
    for _ in range(200):
        winner = boltzmann_select(pop, temperature=0.05, rng=rng)
        counts[winner.spec.harness_id] += 1
    assert counts["high"] > counts["low"] * 5  # large preference at low T


def test_boltzmann_single_record_returns_it():
    pop = [_record(0.5, "only")]
    rng = random.Random(0)
    assert boltzmann_select(pop, 0.1, rng).spec.harness_id == "only"


def test_boltzmann_skips_unaccepted_when_possible():
    pop = [
        HarnessRecord(spec=make_harness(challenger_rules=["a"]), train_score=0.9, accepted=False),
        HarnessRecord(spec=make_harness(challenger_rules=["b"]), train_score=0.1, accepted=True),
    ]
    rng = random.Random(0)
    for _ in range(20):
        # only the accepted one is eligible
        winner = boltzmann_select(pop, 0.1, rng)
        assert winner.spec.challenger_rules == ["b"]


# ---------------------------------------------------------------------------
# apply_mutation
# ---------------------------------------------------------------------------


def test_apply_mutation_adds_and_removes_rules():
    parent = make_harness(challenger_rules=["keep-this", "drop-this"], quality_rules=["q1"])
    parent_id = parent.harness_id
    mutation = {
        "rationale": "test",
        "challenger_rules_add": ["new-rule"],
        "challenger_rules_remove_indices": [1],
        "quality_rules_add": ["q2"],
        "rubric_max_weight": 5,
        "require_self_test": True,
    }
    child = apply_mutation(parent, mutation, iteration=3)
    assert child.parent_id == parent_id
    assert child.iteration == 3
    assert child.challenger_rules == ["keep-this", "new-rule"]
    assert child.quality_rules == ["q1", "q2"]
    assert child.rubric_max_weight == 5
    assert child.require_self_test is True
    assert child.rationale == "test"


def test_apply_mutation_ignores_garbage_safely():
    parent = make_harness(challenger_rules=["a"])
    # All malformed; the loop must NOT crash.
    mutation = {
        "challenger_rules_add": ["", None, "ok"],
        "challenger_rules_remove_indices": ["not-int", 99, -1],  # out of range / wrong type
        "rubric_max_weight": "huge",
        "require_self_test": "yes",  # not a bool
    }
    child = apply_mutation(parent, mutation, iteration=1)
    assert "a" in child.challenger_rules
    assert "ok" in child.challenger_rules
    assert child.rubric_max_weight == 7  # unchanged
    assert child.require_self_test is False  # unchanged


def test_apply_mutation_noop_yields_same_fingerprint():
    parent = make_harness(challenger_rules=["a"])
    child = apply_mutation(parent, {"rationale": ""}, iteration=1)
    assert child.fingerprint() == parent.fingerprint()


# ---------------------------------------------------------------------------
# End-to-end mocked meta-opt loop
# ---------------------------------------------------------------------------


def _make_run_cfg(sample_docs_dir: Path, output_dir: Path, scenario: str = "metaopt") -> RunConfig:
    return RunConfig(
        output_dir=str(output_dir),
        max_examples=4,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(sample_docs_dir)}),
        loop=LoopConfig(max_rounds=2, weak_samples=1, strong_samples=1),
        acceptance=AcceptanceConfig(forbid_weak_zero=False),
        orchestrator=ModelConfig(provider_model=f"mock/{scenario}"),
        challenger=ModelConfig(provider_model=f"mock/{scenario}"),
        weak_solver=ModelConfig(provider_model=f"mock/{scenario}"),
        strong_solver=ModelConfig(provider_model=f"mock/{scenario}"),
        judge=ModelConfig(provider_model=f"mock/{scenario}"),
        metaopt=MetaOptConfig(
            enabled=True,
            max_iterations=2,
            boltzmann_temp=0.1,
            train_size=1,
            val_size=1,
            inner_max_rounds=2,
            mutator=ModelConfig(provider_model=f"mock/{scenario}"),
        ),
    )


def test_metaopt_loop_runs_end_to_end(tmp_path: Path):
    # Need at least train+val grounding items; create 2 docs.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\nDoc A talks about topic A in depth, with specific quantitative claims.")
    (docs / "b.md").write_text("# B\nDoc B covers topic B with detailed numeric examples and design choices.")

    cfg = _make_run_cfg(docs, tmp_path / "out", scenario="metaopt")
    opt = MetaOptimizer(cfg, rng_seed=1)
    summary = opt.run()

    assert summary["iterations"] == 2
    assert summary["population_size"] >= 1
    # Best harness must be accepted (always true since seed is accepted).
    assert summary["best_harness_id"]
    # Output layout
    root = Path(summary["output_dir"])
    assert (root / "config.snapshot.yaml").exists()
    assert (root / "population.json").exists()
    assert (root / "best_harness.json").exists()
    assert (root / "iterations" / "iter_001" / "decision.json").exists()
    assert (root / "iterations" / "iter_001" / "mutation.json").exists()
    pop = json.loads((root / "population.json").read_text())
    assert len(pop) >= 1


def test_metaopt_accepts_improving_mutation(tmp_path: Path):
    """With the 'metaopt' mock, adding the marker rule lifts the strong score,
    so the child's val rate exceeds the seed's val mean and the mutation is
    accepted on the val-only gate."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\nDoc A.")
    (docs / "b.md").write_text("# B\nDoc B.")
    (docs / "c.md").write_text("# C\nDoc C.")

    cfg = _make_run_cfg(docs, tmp_path / "out", scenario="metaopt")
    cfg.metaopt.max_iterations = 1
    opt = MetaOptimizer(cfg, rng_seed=1)
    summary = opt.run()

    # Best should be the child (the marker rule was added).
    pop = [
        HarnessRecord.model_validate(r)
        for r in json.loads((Path(summary["output_dir"]) / "population.json").read_text())
    ]
    accepted_children = [r for r in pop if r.accepted and r.spec.parent_id is not None]
    assert accepted_children, f"no mutation was accepted; pop={[r.model_dump() for r in pop]}"
    # The accepted child must contain the marker rule the mock mutator proposed.
    assert any("Target a quantitative" in r for r in accepted_children[0].spec.challenger_rules)
    # The seed (parent) was re-evaluated on val this iteration, so it accumulated
    # a second val sample beyond its initial one.
    seed = next(r for r in pop if r.spec.parent_id is None)
    assert len(seed.val_scores) >= 2


def test_metaopt_requires_enabled_flag(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("a")
    cfg = _make_run_cfg(docs, tmp_path / "out")
    cfg.metaopt.enabled = False
    with pytest.raises(ValueError):
        MetaOptimizer(cfg)


def test_metaopt_rejects_too_few_grounding_items(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "only.md").write_text("only one doc")
    cfg = _make_run_cfg(docs, tmp_path / "out")
    cfg.metaopt.train_size = 1
    cfg.metaopt.val_size = 1
    with pytest.raises(ValueError):
        MetaOptimizer(cfg)
