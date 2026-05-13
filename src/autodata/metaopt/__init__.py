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

from autodata.metaopt.evaluation import aggregate_failures_from_db, evaluate_harness
from autodata.metaopt.mutation import Mutator, apply_mutation
from autodata.metaopt.optimizer import MetaOptimizer
from autodata.metaopt.records import HarnessRecord, MetaIteration
from autodata.metaopt.selection import boltzmann_select

__all__ = [
    "MetaOptimizer",
    "Mutator",
    "HarnessRecord",
    "MetaIteration",
    "apply_mutation",
    "boltzmann_select",
    "evaluate_harness",
    "aggregate_failures_from_db",
]
