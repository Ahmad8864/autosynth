"""Training export transforms, truncation tracking, and CLI integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autosynth.agents.solver import build_request
from autosynth.cli import app
from autosynth.config import (
    AcceptanceConfig,
    DispatcherConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autosynth.domains.qa_from_documents import QAFromDocuments
from autosynth.export import ExportContext, build_rows
from autosynth.harness import DEFAULT_HARNESS
from autosynth.llm import register_mock
from autosynth.runner import Runner
from autosynth.schemas import Candidate, RubricCriterion, SolverScore
from autosynth.store import Store
from autosynth.utils import stable_id

cli = CliRunner()

# Fixtures


@pytest.fixture
def domain(tmp_path: Path):
    (tmp_path / "doc.md").write_text("body of doc " * 30)
    return QAFromDocuments(source_dir=str(tmp_path))


def _cfg(**acceptance) -> RunConfig:
    return RunConfig(
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": "/tmp"}),
        acceptance=AcceptanceConfig(**acceptance),
        judge=ModelConfig(provider_model="mock/x"),
    )


def _ctx(domain, **kw) -> ExportContext:
    acceptance = kw.pop("acceptance", {})
    return ExportContext(cfg=_cfg(**acceptance), domain=domain, harness=DEFAULT_HARNESS, **kw)


def _score(solver: str, attempt: int, total: float, *, text: str | None = None, correct=None) -> dict:
    return SolverScore(
        solver=solver,
        attempt=attempt,
        raw_response=text if text is not None else f"{solver}-{attempt}",
        total=total,
        correct=correct,
    ).model_dump()


def _record(**overrides) -> dict:
    rec = {
        "input": {"question": "What does the doc say?", "context": "ctx"},
        "reference_output": "the reference answer",
        "rubric": [{"id": "c1", "description": "names detail", "weight": 5}],
        "domain": "qa_from_documents",
        "source_id": "s1",
        "metadata": {},
        "run_id": "r1",
        "item_id": "i1",
        "refinement_round": 1,
        "weak_avg": 0.25,
        "strong_avg": 0.85,
        "gap": 0.6,
        "weak_scores": [_score("weak", 0, 0.2), _score("weak", 1, 0.3)],
        "strong_scores": [_score("strong", 0, 0.9), _score("strong", 1, 0.8)],
        "acceptance_rationale": "ok",
        "audit": None,
    }
    rec.update(overrides)
    return rec


def _candidate_of(rec: dict) -> Candidate:
    return Candidate(
        candidate_id=rec["item_id"],
        domain=rec["domain"],
        source_id=rec["source_id"],
        payload=rec["input"],
        rubric=[RubricCriterion(**c) for c in rec["rubric"]],
        reference_output=rec["reference_output"],
    )


def _pairs(*recs: dict) -> list[tuple[dict, Candidate]]:
    return [(r, _candidate_of(r)) for r in recs]


# SFT


def test_sft_reference_default(domain):
    rows, skipped = build_rows(_pairs(_record()), "sft", _ctx(domain))
    assert skipped == 0
    (row,) = rows
    assert row["messages"][-1] == {"role": "assistant", "content": "the reference answer"}
    assert row["meta"]["item_id"] == "i1"


def test_sft_best_strong_with_floor_fallback(domain):
    rows, _ = build_rows(_pairs(_record()), "sft", _ctx(domain, completion="best-strong"))
    assert rows[0]["messages"][-1]["content"] == "strong-0"
    low = _record(strong_scores=[_score("strong", 0, 0.5), _score("strong", 1, 0.4)])
    rows, _ = build_rows(_pairs(low), "sft", _ctx(domain, completion="best-strong"))
    assert rows[0]["messages"][-1]["content"] == "the reference answer"


def test_sft_prompt_matches_solver_request(domain):
    rec = _record()
    rows, _ = build_rows(_pairs(rec), "sft", _ctx(domain))
    req = build_request(
        item_id="i1",
        round_n=1,
        attempt=0,
        model_key="mock/x",
        candidate=_candidate_of(rec),
        role="weak",
        domain=domain,
        harness=DEFAULT_HARNESS,
    )
    assert rows[0]["messages"][:-1] == req.messages


def test_missing_candidate_blob_is_skipped(domain):
    rows, skipped = build_rows([(_record(), None)], "grpo", _ctx(domain))
    assert rows == [] and skipped == 1


# DPO


def test_dpo_pairs_extremes_solver_blind(domain):
    rows, _ = build_rows(_pairs(_record()), "dpo", _ctx(domain))
    (row,) = rows
    assert row["chosen"] == [{"role": "assistant", "content": "strong-0"}]
    assert row["rejected"] == [{"role": "assistant", "content": "weak-0"}]
    assert row["meta"]["chosen_score"] == 0.9 and row["meta"]["rejected_score"] == 0.2

    # A weak rollout that outscores every strong one is chosen: mining is solver-blind.
    upset = _record(weak_scores=[_score("weak", 0, 0.95), _score("weak", 1, 0.1)])
    rows, _ = build_rows(_pairs(upset), "dpo", _ctx(domain))
    assert rows[0]["meta"]["chosen_solver"] == "weak"


def test_dpo_margin_and_chosen_floors(domain):
    rows, skipped = build_rows(_pairs(_record()), "dpo", _ctx(domain, min_margin=0.9))
    assert rows == [] and skipped == 1
    low = _record(strong_scores=[_score("strong", 0, 0.6)], weak_scores=[_score("weak", 0, 0.1)])
    rows, skipped = build_rows(_pairs(low), "dpo", _ctx(domain))
    assert rows == [] and skipped == 1


def test_dpo_skips_truncated_rollout(domain):
    truncated = frozenset({("i1", 1, "weak", 0)})
    rows, _ = build_rows(_pairs(_record()), "dpo", _ctx(domain, truncated=truncated))
    assert rows[0]["rejected"] == [{"role": "assistant", "content": "weak-1"}]


def test_dpo_verifiable_pools_by_correctness(domain):
    rec = _record(
        weak_scores=[_score("weak", 0, 0.0, correct=False), _score("weak", 1, 1.0, correct=True)],
        strong_scores=[_score("strong", 0, 1.0, correct=True), _score("strong", 1, 0.0, correct=False)],
    )
    rows, _ = build_rows(_pairs(rec), "dpo", _ctx(domain, acceptance={"mode": "verifiable"}))
    (row,) = rows
    assert row["meta"]["chosen_score"] == 1.0 and row["meta"]["rejected_score"] == 0.0


# GRPO


def test_grpo_rubric_columns_and_stats(domain):
    rows, _ = build_rows(_pairs(_record()), "grpo", _ctx(domain))
    (row,) = rows
    assert row["reward_kind"] == "rubric_judge"
    assert row["rubric"] == _record()["rubric"]
    assert "solution" not in row
    assert row["stats"] == {
        "weak_avg": 0.25,
        "weak_std": 0.05,
        "strong_avg": 0.85,
        "strong_std": 0.05,
        "gap": 0.6,
    }


def test_grpo_verifiable_solution_column(domain):
    rows, _ = build_rows(_pairs(_record()), "grpo", _ctx(domain, acceptance={"mode": "verifiable"}))
    (row,) = rows
    assert row["reward_kind"] == "verifiable"
    assert row["solution"] == "the reference answer"
    assert "rubric" not in row


# Meta + dedup


def test_meta_contents_and_opt_out(domain):
    rows, _ = build_rows(
        _pairs(_record(audit={"passed": True, "failures": [], "notes": None})), "grpo", _ctx(domain)
    )
    meta = rows[0]["meta"]
    assert meta["models"]["judge"] == "mock/x"
    assert meta["acceptance"] == {
        "weak_avg": 0.25,
        "strong_avg": 0.85,
        "gap": 0.6,
        "rationale": "ok",
        "audit_passed": True,
    }
    assert meta["harness_fingerprint"] == DEFAULT_HARNESS.fingerprint()
    rows, _ = build_rows(_pairs(_record()), "grpo", _ctx(domain, include_meta=False))
    assert "meta" not in rows[0]


def test_duplicate_prompts_deduped(domain):
    a, b = _record(item_id="i1"), _record(item_id="i2")
    rows, skipped = build_rows(_pairs(a, b), "sft", _ctx(domain))
    assert len(rows) == 1 and skipped == 1


# Store truncation query


def test_truncated_attempts(tmp_path):
    store = Store(tmp_path / "run.db")
    store.create_run("r1", config={}, harness=None)
    iid = store.insert_item(run_id="r1", source_id="s1", domain="d", state="NEED_SCORES")
    store.upsert_round(item_id=iid, round_n=1)
    for attempt, completion_tokens in ((0, 100), (1, 60)):
        rid = stable_id(iid, 1, "weak", attempt)
        store.insert_requests(
            [
                {
                    "request_id": rid,
                    "item_id": iid,
                    "round_n": 1,
                    "role": "weak",
                    "model_key": "mock/x",
                    "attempt": attempt,
                    "messages": [],
                    "json_mode": False,
                    "max_tokens": 100,
                }
            ]
        )
        store.insert_response(request_id=rid, model="mock/x", text="t", completion_tokens=completion_tokens)
        store.insert_score(
            item_id=iid,
            round_n=1,
            score=SolverScore(solver="weak", attempt=attempt, raw_response="t", total=0.0),
            solver_response_id=rid,
            judge_response_id=rid,
        )
    assert store.truncated_attempts("r1") == {(iid, 1, "weak", 0)}
    store.close()


# CLI end-to-end (mock provider)


def _export_happy(role: str, messages) -> str:
    """Return a distinct mock candidate for each source document."""
    text = " ".join(m.get("content", "") for m in messages)
    if role == "challenger" or "ROLE:CHALLENGER" in text:
        tag = "A" if "topic A" in text else "B"
        return json.dumps(
            {
                "payload": {"question": f"What is topic {tag} about?", "context": f"ctx {tag}"},
                "reference_output": f"topic {tag} reference",
                "rubric": [{"id": "c1", "description": "names detail", "weight": 5}],
            }
        )
    if "ROLE:QUALITY" in text:
        return json.dumps({"passed": True, "failures": [], "notes": "ok"})
    if role == "judge":
        if "vague answer" in text:
            return json.dumps({"per_criterion": {"c1": 0.2}, "total": 0.2})
        return json.dumps({"per_criterion": {"c1": 0.9}, "total": 0.9})
    if role == "weak":
        return "vague answer"
    if role == "strong":
        return "specific, source-grounded answer"
    return "{}"


register_mock("export-happy", _export_happy)


def _run_cfg(docs_dir: Path, output_dir: Path) -> RunConfig:
    return RunConfig(
        run_id="export-e2e",
        output_dir=str(output_dir),
        max_examples=2,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(docs_dir)}),
        loop=LoopConfig(max_rounds=2, weak_samples=2, strong_samples=2),
        acceptance=AcceptanceConfig(forbid_weak_zero=False),
        orchestrator=ModelConfig(provider_model="mock/export-happy"),
        challenger=ModelConfig(provider_model="mock/export-happy"),
        weak_solver=ModelConfig(provider_model="mock/export-happy"),
        strong_solver=ModelConfig(provider_model="mock/export-happy"),
        judge=ModelConfig(provider_model="mock/export-happy"),
        dispatcher=DispatcherConfig(concurrency=4, items_per_advance=10, poll_interval_s=0.0),
    )


@pytest.fixture
def finished_run(sample_docs: Path, output_dir: Path) -> Path:
    runner = Runner(_run_cfg(sample_docs, output_dir))
    summary = runner.run()
    assert summary.accepted == 2
    return runner.run_dir


def test_cli_export_grpo(finished_run: Path):
    result = cli.invoke(app, ["export", "--run", str(finished_run), "--format", "grpo"])
    assert result.exit_code == 0, result.output
    target = finished_run / "exports" / "grpo.jsonl"
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["prompt"][0]["role"] in ("system", "user")
    assert rows[0]["reward_kind"] == "rubric_judge"
    assert rows[0]["meta"]["run_id"] == "export-e2e"
    assert (finished_run / "exports" / "grpo.README.md").exists()


def test_cli_export_dpo_and_sft(finished_run: Path):
    result = cli.invoke(app, ["export", "--run", str(finished_run), "--format", "dpo"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in (finished_run / "exports" / "dpo.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["chosen"][0]["role"] == "assistant"
    assert rows[0]["meta"]["chosen_score"] > rows[0]["meta"]["rejected_score"]

    result = cli.invoke(
        app,
        ["export", "--run", str(finished_run), "--format", "sft", "--completion", "best-strong", "--no-card"],
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in (finished_run / "exports" / "sft.jsonl").read_text().splitlines()]
    assert rows[0]["messages"][-1]["role"] == "assistant"
    assert not (finished_run / "exports" / "sft.README.md").exists()


def test_cli_export_rejects_bad_flags(finished_run: Path):
    assert cli.invoke(app, ["export", "--run", str(finished_run), "--format", "nope"]).exit_code == 2
    assert (
        cli.invoke(
            app, ["export", "--run", str(finished_run), "--format", "sft", "--completion", "nope"]
        ).exit_code
        == 2
    )
