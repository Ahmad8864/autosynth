"""Optional safety and PII filtering hooks."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

# These heuristics are a starting point, not a compliance check.
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
    """Load a safety filter from ``module:attr`` or ``path/to/file.py:attr``."""
    if not spec:
        return default_filter
    target, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"safety filter must be 'module:attr' or 'path.py:attr', got {spec!r}")

    p = Path(target)
    if p.exists() and p.suffix == ".py":
        mod_name = f"autosynth_user_safety_{p.stem}"
        spec_obj = importlib.util.spec_from_file_location(mod_name, p)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"cannot import safety filter from {target}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    fn = getattr(module, attr, None)
    if fn is None:
        raise AttributeError(f"safety filter {attr!r} not found in {target!r}")
    if not callable(fn):
        raise TypeError(f"safety filter {spec!r} is not callable")
    try:
        inspect.signature(fn).bind("")
    except TypeError as e:
        raise TypeError(f"safety filter {spec!r} must accept one positional str argument: {e}") from e
    # Signature accepts (str); return value still trusted at first call.
    return cast(SafetyFilter, fn)
