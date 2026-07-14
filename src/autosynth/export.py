"""Build SFT, DPO, and GRPO datasets from accepted records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Any

from autosynth import __version__
from autosynth.acceptance import resolve_mode
from autosynth.agents.solver import solver_messages
from autosynth.config import RunConfig, load_snapshot
from autosynth.domain import DomainAdapter, build_domain
from autosynth.harness import DEFAULT_HARNESS, HarnessSpec
from autosynth.schemas import Candidate
from autosynth.store import Store
from autosynth.utils import write_hf_dataset, write_jsonl

TRAINER_FORMATS = ("sft", "dpo", "grpo")


@dataclass(frozen=True)
class ExportContext:
    """Settings shared by the export transforms."""

    cfg: RunConfig
    domain: DomainAdapter
    harness: HarnessSpec
    completion: str = "reference"
    min_score: float = 0.75
    min_chosen: float = 0.75
    min_margin: float = 0.30
    include_meta: bool = True
    truncated: frozenset[tuple[str, int, str, int]] = frozenset()

    @property
    def mode(self) -> str:
        return resolve_mode(self.cfg, self.domain)


@dataclass(frozen=True)
class ExportResult:
    path: Path
    written: int
    skipped: int


def run_export(
    store: Store,
    run_row: sqlite3.Row,
    run_dir: Path,
    fmt: str,
    *,
    out: Path | None = None,
    to: str = "jsonl",
    completion: str = "reference",
    min_score: float = 0.75,
    min_chosen: float = 0.75,
    min_margin: float = 0.30,
    include_meta: bool = True,
    card: bool = True,
) -> ExportResult:
    """Export a run's accepted records as a training dataset."""
    snap = run_dir / "config.snapshot.yaml"
    if not snap.exists():
        raise FileNotFoundError(f"no config snapshot at {snap} (needed to re-render solver prompts)")
    cfg = load_snapshot(snap)
    run_id = run_row["run_id"]
    needs_truncated = fmt == "dpo" or (fmt == "sft" and completion == "best-strong")
    ctx = ExportContext(
        cfg=cfg,
        domain=build_domain(cfg.domain.name, cfg.domain.path, cfg.domain.params),
        harness=(
            HarnessSpec.model_validate_json(run_row["harness_blob"])
            if run_row["harness_blob"]
            else DEFAULT_HARNESS
        ),
        completion=completion,
        min_score=min_score,
        min_chosen=min_chosen,
        min_margin=min_margin,
        include_meta=include_meta,
        truncated=frozenset(store.truncated_attempts(run_id)) if needs_truncated else frozenset(),
    )
    rows, skipped = build_rows(store.accepted_with_candidates(run_id), fmt, ctx)
    card_text = dataset_card(fmt, ctx, run_id, len(rows)) if card else None
    if to == "hf":
        target = out or (run_dir / "exports" / f"{fmt}_hf")
        if write_hf_dataset(target, rows) is None:
            raise RuntimeError("hf export failed (no rows, or install autosynth[hf])")
        if card_text:
            (target / "README.md").write_text(card_text, encoding="utf-8")
    else:
        target = out or (run_dir / "exports" / f"{fmt}.jsonl")
        write_jsonl(target, rows)
        if card_text:
            target.with_suffix(".README.md").write_text(card_text, encoding="utf-8")
    return ExportResult(path=target, written=len(rows), skipped=skipped)


def build_rows(
    records: Iterable[tuple[dict, Candidate | None]], fmt: str, ctx: ExportContext
) -> tuple[list[dict], int]:
    """Build rows, deduplicated by prompt and sorted by item ID."""
    transform = {"sft": _sft_row, "dpo": _dpo_row, "grpo": _grpo_row}[fmt]
    static_meta = _static_meta(ctx) if ctx.include_meta else None
    keyed: list[tuple[str, dict]] = []
    seen: set[str] = set()
    skipped = 0
    for rec, candidate in records:
        if candidate is None:
            skipped += 1
            continue
        prompt = solver_messages(candidate, ctx.domain, ctx.harness)
        row = transform(rec, candidate, prompt, ctx)
        if row is None:
            skipped += 1
            continue
        key = json.dumps(prompt, sort_keys=True)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        if static_meta is not None:
            row["meta"] = {**static_meta, **_record_meta(rec), **(row.pop("meta", None) or {})}
        else:
            row.pop("meta", None)
        keyed.append((rec["item_id"], row))
    keyed.sort(key=lambda t: t[0])
    return [r for _, r in keyed], skipped


def _usable(rec: dict, score: dict, ctx: ExportContext) -> bool:
    """Return whether an attempt is non-empty and not truncated."""
    if not score["raw_response"].strip():
        return False
    key = (rec["item_id"], rec["refinement_round"], score["solver"], score["attempt"])
    return key not in ctx.truncated


def _sft_row(rec: dict, candidate: Candidate, prompt: list[dict], ctx: ExportContext) -> dict | None:
    completion = candidate.reference_output
    if ctx.completion == "best-strong":
        best = _best_strong(rec, ctx)
        if best is not None:
            completion = best["raw_response"]
    if not completion:
        return None
    return {"messages": [*prompt, {"role": "assistant", "content": completion}]}


def _best_strong(rec: dict, ctx: ExportContext) -> dict | None:
    pool = [s for s in rec["strong_scores"] if _usable(rec, s, ctx)]
    if ctx.mode == "verifiable":
        pool = [s for s in pool if s["correct"] is True]
    else:
        pool = [s for s in pool if s["total"] >= ctx.min_score]
    return max(pool, key=lambda s: (s["total"], -s["attempt"]), default=None)


def _dpo_row(rec: dict, candidate: Candidate, prompt: list[dict], ctx: ExportContext) -> dict | None:
    """Solver-blind pair mining: rank all rollouts by judge score, pair the extremes."""
    pool = sorted(
        (s for s in [*rec["weak_scores"], *rec["strong_scores"]] if _usable(rec, s, ctx)),
        key=lambda s: -s["total"],
    )
    if ctx.mode == "verifiable":
        correct = [s for s in pool if s["correct"] is True]
        wrong = [s for s in pool if s["correct"] is False]
        if not correct or not wrong:
            return None
        chosen, rejected = correct[0], wrong[-1]
    else:
        if len(pool) < 2:
            return None
        chosen, rejected = pool[0], pool[-1]
        if chosen["total"] < ctx.min_chosen or chosen["total"] - rejected["total"] < ctx.min_margin:
            return None
    if chosen["raw_response"].strip() == rejected["raw_response"].strip():
        return None
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen["raw_response"]}],
        "rejected": [{"role": "assistant", "content": rejected["raw_response"]}],
        "meta": {
            "chosen_score": chosen["total"],
            "rejected_score": rejected["total"],
            "chosen_solver": chosen["solver"],
            "rejected_solver": rejected["solver"],
        },
    }


def _grpo_row(rec: dict, candidate: Candidate, prompt: list[dict], ctx: ExportContext) -> dict:
    row: dict[str, Any] = {
        "prompt": prompt,
        "task": rec.get("domain", ctx.domain.name),
        "reward_kind": "verifiable" if ctx.mode == "verifiable" else "rubric_judge",
    }
    if ctx.mode == "verifiable":
        # Keep the reference under `solution` for correctness-based rewards.
        row["solution"] = candidate.reference_output
    else:
        row["rubric"] = [c.model_dump() for c in candidate.rubric]
        row["reference_output"] = candidate.reference_output
    row["stats"] = {
        "weak_avg": rec["weak_avg"],
        "weak_std": round(pstdev([s["total"] for s in rec["weak_scores"]] or [0.0]), 4),
        "strong_avg": rec["strong_avg"],
        "strong_std": round(pstdev([s["total"] for s in rec["strong_scores"]] or [0.0]), 4),
        "gap": rec["gap"],
    }
    return row


def _static_meta(ctx: ExportContext) -> dict:
    """Build the metadata shared by every row in an export."""
    cfg = ctx.cfg
    roles = {
        "challenger": cfg.challenger,
        "weak_solver": cfg.weak_solver,
        "strong_solver": cfg.strong_solver,
        "judge": cfg.judge,
    }
    if cfg.audit.enabled:
        roles["auditor"] = cfg.auditor or cfg.judge
    return {
        "acceptance_mode": ctx.mode,
        "models": {k: m.provider_model for k, m in roles.items()},
        "sampling": {k: {"temperature": m.temperature, "max_tokens": m.max_tokens} for k, m in roles.items()},
        "harness_fingerprint": ctx.harness.fingerprint(),
        "autosynth_version": __version__,
    }


def _record_meta(rec: dict) -> dict:
    audit = rec.get("audit")  # Older records may not have an audit field.
    return {
        "run_id": rec["run_id"],
        "item_id": rec["item_id"],
        "source_id": rec["source_id"],
        "domain": rec.get("domain"),
        "refinement_round": rec["refinement_round"],
        "acceptance": {
            "weak_avg": rec["weak_avg"],
            "strong_avg": rec["strong_avg"],
            "gap": rec["gap"],
            "rationale": rec["acceptance_rationale"],
            "audit_passed": audit["passed"] if audit else None,
        },
        "source_metadata": rec.get("metadata") or {},
    }


_GRPO_REWARD_SNIPPET = """\
# Example reward for rubric datasets. Dataset columns arrive as keyword
# arguments. Implement judge_criterion() to return a score from 0 to 1.
def rubric_reward(prompts, completions, rubric, reference_output, **kwargs):
    rewards = []
    for completion, criteria, reference in zip(completions, rubric, reference_output):
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        total = sum(c["weight"] * judge_criterion(text, c, reference) for c in criteria)
        rewards.append(total / (sum(c["weight"] for c in criteria) or 1))
    return rewards"""


def dataset_card(fmt: str, ctx: ExportContext, run_id: str, n: int) -> str:
    lines = [
        f"# autosynth {fmt} export",
        "",
        f"- run: {run_id}",
        f"- records: {n}",
        f"- domain: {ctx.domain.name} · acceptance mode: {ctx.mode}",
        f"- autosynth version: {__version__}",
        "",
    ]
    if fmt == "sft":
        lines.append("Conversational SFT records. Apply the model's chat template before tokenization.")
    elif fmt == "dpo":
        lines.append(
            "Conversational preference records with explicit prompts. Pairs are selected without regard "
            "to solver role: attempts are ranked by score, then the extremes are paired and margin-gated."
        )
    elif ctx.mode == "verifiable":
        lines.append(
            "Prompt records for reward-based training. The `solution` column contains the reference answer "
            "for correctness rewards."
        )
    else:
        lines += [
            "Prompt records for reward-based training. The `rubric`, `reference_output`, and `stats` "
            "columns are available to reward functions:",
            "",
            "```python",
            _GRPO_REWARD_SNIPPET,
            "```",
        ]
    return "\n".join(lines) + "\n"
