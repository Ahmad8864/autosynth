"""Domain plugin interface and registry.

A domain plugin defines *only* the parts of the pipeline that depend on the
target task. Everything else (the loop, acceptance criteria, writers, agents)
is provided by the framework.

There are two ways to register a domain:

1. In-process decorator (for domains shipped with this package or with the
   user's own code that is already imported):

       @register_domain("my_qa")
       class MyQA(DomainAdapter): ...

2. By path, referenced in YAML config:

       domain:
         path: /abs/path/to/my_domain.py:MyDomain
         params: {seeds_path: ./seeds.jsonl}

   The framework dynamically imports the file and instantiates the class.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autodata.schemas import Candidate, RubricCriterion


@dataclass
class GroundingItem:
    """Source material for one trajectory.

    `body` is whatever raw content the challenger should read (text, JSON,
    serialized table). `metadata` propagates into the trajectory.
    """

    source_id: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DomainAdapter(ABC):
    """Subclass this to add a new domain. Six methods, all required.

    See `autodata/domains/qa_from_documents.py` for a worked example.
    """

    name: str = "unnamed"
    description: str = ""

    def __init__(self, **params: Any):
        self.params = params

    # ---- 1. grounding source ------------------------------------------------
    @abstractmethod
    def load_grounding(self) -> Iterable[GroundingItem]:
        """Yield source items. May be a generator; framework iterates once."""

    # ---- 2. challenger prompt ----------------------------------------------
    @abstractmethod
    def generation_prompt(
        self,
        item: GroundingItem,
        feedback: list[str],
        round_n: int,
        prior_payloads: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Return chat messages instructing the challenger to emit a candidate.

        Must instruct the model to return JSON with at least:
          {"payload": {...}, "reference_output": "...", "rubric": [...]}
        """

    # ---- 3. candidate validation -------------------------------------------
    @abstractmethod
    def validate_candidate(self, candidate: Candidate) -> list[str]:
        """Return a list of failure reasons; empty list means valid."""

    # ---- 4. solver prompt --------------------------------------------------
    @abstractmethod
    def solver_prompt(self, candidate: Candidate, solver_role: str) -> list[dict[str, str]]:
        """Build the prompt that asks a solver to attempt the task."""

    # ---- 5. verifier (quality) prompt --------------------------------------
    @abstractmethod
    def quality_prompt(self, candidate: Candidate) -> list[dict[str, str]]:
        """Prompt for the quality verifier (leakage, coverage, formatting...)."""

    # ---- 6. judge prompt ---------------------------------------------------
    @abstractmethod
    def judge_prompt(
        self,
        candidate: Candidate,
        solver_response: str,
        solver_role: str,
    ) -> list[dict[str, str]]:
        """Prompt the judge to score solver_response against the rubric."""

    # ---- optional hooks -----------------------------------------------------

    def format_accepted(self, candidate: Candidate, extra: dict[str, Any]) -> dict[str, Any]:
        """Final shape written to the dataset. Override to customize."""
        return {
            "input": candidate.payload,
            "reference_output": candidate.reference_output,
            "rubric": [c.model_dump() for c in candidate.rubric],
            "domain": candidate.domain,
            "source_id": candidate.source_id,
            "metadata": candidate.metadata,
            **extra,
        }

    def leakage_rules(self) -> list[str]:
        """Optional: domain-specific leakage rules surfaced to the verifier."""
        return []


# ---------------------------------------------------------------------------
# Registry + loader
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[DomainAdapter]] = {}


def register_domain(name: str):
    def deco(cls: type[DomainAdapter]) -> type[DomainAdapter]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_domain_class(name: str) -> type[DomainAdapter]:
    if name not in _REGISTRY:
        # Lazy-import built-ins so registration runs.
        with contextlib.suppress(ImportError):
            importlib.import_module("autodata.domains")
    if name not in _REGISTRY:
        raise KeyError(f"unknown domain {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def load_domain_from_path(spec: str) -> type[DomainAdapter]:
    """Load `path/to/file.py:ClassName` or `module.submod:ClassName`."""
    target, _, cls_name = spec.partition(":")
    if not cls_name:
        raise ValueError(f"domain path must be 'file_or_module:Class', got {spec!r}")

    p = Path(target)
    if p.exists() and p.suffix == ".py":
        mod_name = f"autodata_user_domain_{p.stem}"
        spec_obj = importlib.util.spec_from_file_location(mod_name, p)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"cannot import {target}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    cls = getattr(module, cls_name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, DomainAdapter):
        raise TypeError(f"{spec} is not a DomainAdapter subclass")
    return cls


def build_domain(name: str | None, path: str | None, params: dict[str, Any]) -> DomainAdapter:
    if path:
        cls = load_domain_from_path(path)
    elif name:
        cls = get_domain_class(name)
    else:
        raise ValueError("domain requires `name` or `path`")
    return cls(**(params or {}))


# Convenience for rubric construction in domain plugins
def rubric(*items: tuple[str, str, int]) -> list[RubricCriterion]:
    return [RubricCriterion(id=i, description=d, weight=w) for i, d, w in items]


def bullet_list(
    items: list[str] | list[dict[str, Any]],
    *,
    key: str | None = None,
    limit: int = 0,
    empty: str = "(none)",
) -> str:
    """Render a bulleted list for inclusion in a prompt.

    - Pass a list of strings to bullet them directly.
    - Pass a list of dicts plus ``key`` to extract that field from each dict.
    - ``limit > 0`` truncates each rendered item to that many characters.
    - Returns ``empty`` when the input is empty (default ``"(none)"``).
    """
    if not items:
        return empty
    if key is None:
        lines = [str(x) for x in items]
    else:
        lines = [str(d.get(key, "") or "") for d in items]  # type: ignore[union-attr]
    if limit > 0:
        lines = [s[:limit] for s in lines]
    return "\n".join(f"- {s}" for s in lines) or empty
