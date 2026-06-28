"""Boltzmann (softmax) parent selection over the harness population."""

from __future__ import annotations

import math
import random

from autosynth.metaopt.records import HarnessRecord


def boltzmann_select(records: list[HarnessRecord], temperature: float, rng: random.Random) -> HarnessRecord:
    """Select a parent record proportional to exp(val_mean / T)."""
    accepted = [r for r in records if r.accepted]
    if not accepted:
        accepted = records
    if len(accepted) == 1:
        return accepted[0]
    T = max(temperature, 1e-6)
    scores = [r.val_mean if r.val_mean is not None else 0.0 for r in accepted]
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
