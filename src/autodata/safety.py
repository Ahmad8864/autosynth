"""Optional safety / PII filter hook.

The default filter is permissive (no-op). Users can override the hook by
providing a Python path in config (`safety.filter: my_pkg.my_module:my_filter`).
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Callable

# Very rough PII heuristics — intentionally conservative; meant as a starting
# point, not a compliance solution. Override with a real DLP tool in production.
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "credit-card-like"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "email"),
]


@dataclass
class SafetyVerdict:
    allowed: bool
    reasons: list[str]


SafetyFilter = Callable[[str], SafetyVerdict]


def default_filter(text: str) -> SafetyVerdict:
    reasons: list[str] = []
    for pat, label in _PII_PATTERNS:
        if pat.search(text or ""):
            reasons.append(f"pii:{label}")
    return SafetyVerdict(allowed=not reasons, reasons=reasons)


def load_filter(spec: str | None) -> SafetyFilter:
    if not spec:
        return default_filter
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"safety filter must be 'module:attr', got {spec!r}")
    mod = importlib.import_module(module)
    return getattr(mod, attr)
