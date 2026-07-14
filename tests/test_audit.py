"""Final-audit transitions, configuration, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autosynth.acceptance import JudgePolicy
from autosynth.agents import auditor
from autosynth.config import (
    AcceptanceConfig,
    AuditConfig,
    DispatcherConfig,
    DomainConfig,
    LoopConfig,
    ModelConfig,
    RunConfig,
)
from autosynth.dispatcher.hydration import accepted_extras, load_item_state
from autosynth.domain import GroundingItem
from autosynth.domains.qa_from_documents import QAFromDocuments
from autosynth.harness import DEFAULT_HARNESS, make_harness
from autosynth.llm import register_mock
from autosynth.pipeline import ItemState, State, StepResponse, StepResult, model_key_for, step
from autosynth.runner import Runner
from autosynth.schemas import (
    Candidate,
    EvalReport,
    QualityCheck,
    Round,
    RubricCriterion,
    SolverScore,
)
from autosynth.store import Store
from autosynth.utils import stable_id

# Fixtures


@pytest.fixture
def domain(tmp_path: Path):
    (tmp_path / "doc.md").write_text("body of doc " * 50)
    return QAFromDocuments(source_dir=str(tmp_path))


@pytest.fixture
def grounding(domain) -> GroundingItem:
    return next(iter(domain.load_grounding()))


def _cfg(*, max_rounds: int = 3, audit_enabled: bool = True, **audit_kw) -> RunConfig:
    return RunConfig(
        run_id="r1",
        output_dir="/tmp/out",
        max_examples=1,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": "/tmp"}),
        loop=LoopConfig(max_rounds=max_rounds, weak_samples=2, strong_samples=2, short_circuit_strong=False),
        acceptance=AcceptanceConfig(forbid_weak_zero=False),
        audit=AuditConfig(enabled=audit_enabled, **audit_kw),
        challenger=ModelConfig(provider_model="mock/x"),
        weak_solver=ModelConfig(provider_model="mock/x"),
        strong_solver=ModelConfig(provider_model="mock/x"),
        judge=ModelConfig(provider_model="mock/x"),
    )


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="cand-1",
        domain="qa_from_documents",
        source_id="s1",
        payload={"question": "q", "context": "c"},
        rubric=[
            RubricCriterion(id="c1", description="names contribution", weight=5),
            RubricCriterion(id="c2", description="cites detail", weight=3),
        ],
        reference_output="reference answer",
    )


def _score(role: str, attempt: int, total: float) -> SolverScore:
    return SolverScore(solver=role, attempt=attempt, raw_response="x", total=total)


def _report(accepted: bool = True) -> EvalReport:
    return EvalReport(
        weak_scores=[_score("weak", 0, 0.2), _score("weak", 1, 0.2)],
        strong_scores=[_score("strong", 0, 0.9), _score("strong", 1, 0.9)],
        weak_avg=0.2,
        strong_avg=0.9,
        gap=0.7,
        accepted=accepted,
        acceptance_rationale="thresholds met" if accepted else None,
    )


def _seed_item(grounding: GroundingItem, *, state: State, **overrides) -> ItemState:
    base = ItemState(
        item_id="i1",
        run_id="r1",
        source_id=grounding.source_id,
        domain="qa_from_documents",
        state=state,
        current_round=1,
        source_metadata=grounding.metadata,
    )
    return type(base)(**{**base.__dict__, **overrides})


def _audit_item(grounding: GroundingItem, **overrides) -> ItemState:
    defaults = dict(
        candidate=_candidate(),
        quality=QualityCheck(passed=True),
        weak_scores=(_score("weak", 0, 0.2), _score("weak", 1, 0.2)),
        strong_scores=(_score("strong", 0, 0.9), _score("strong", 1, 0.9)),
        pending_report=_report(),
    )
    return _seed_item(grounding, state=State.NEED_AUDIT, **{**defaults, **overrides})


def _judge_resp(*, solver_role: str, attempt: int, total: float) -> StepResponse:
    parent_id = stable_id("i1", 1, solver_role, attempt)
    return StepResponse(
        request_id=stable_id("i1", 1, "judge", solver_role, parent_id),
        role="judge",
        round_n=1,
        attempt=attempt,
        text=json.dumps({"per_criterion": {"c1": total, "c2": total}, "total": total}),
        parent_response_id=parent_id,
        solver_response_text="x",
        solver_role=solver_role,
    )


def _audit_resp(body: str) -> StepResponse:
    return StepResponse(
        request_id=stable_id("i1", 1, "audit", 0),
        role="audit",
        round_n=1,
        attempt=0,
        text=body,
    )


def _step(item, responses, *, cfg, domain, grounding, policy=None):
    return step(
        item,
        responses,
        cfg=cfg,
        harness=DEFAULT_HARNESS,
        domain=domain,
        grounding=grounding,
        policy=policy,
    )


def _evaluation(res: StepResult) -> EvalReport:
    assert res.completed_round is not None and res.completed_round.evaluation is not None
    return res.completed_round.evaluation


# Accept decision -> NEED_AUDIT


def test_accepted_decision_enters_audit(domain, grounding):
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_candidate(),
        quality=QualityCheck(passed=True),
    )
    judges = [
        _judge_resp(solver_role="weak", attempt=0, total=0.2),
        _judge_resp(solver_role="weak", attempt=1, total=0.2),
        _judge_resp(solver_role="strong", attempt=0, total=0.9),
        _judge_resp(solver_role="strong", attempt=1, total=0.9),
    ]
    res = _step(item, judges, cfg=_cfg(), domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_AUDIT
    assert res.state.pending_report is not None and res.state.pending_report.accepted
    assert res.completed_round is None
    assert len(res.scores_to_persist) == 4
    (req,) = res.new_requests
    assert req.role == "audit"
    assert req.request_id == stable_id("i1", 1, "audit", 0)
    assert req.json_mode


def test_audit_disabled_accepts_directly(domain, grounding):
    item = _seed_item(
        grounding,
        state=State.NEED_SCORES,
        candidate=_candidate(),
        quality=QualityCheck(passed=True),
    )
    judges = [
        _judge_resp(solver_role="weak", attempt=0, total=0.2),
        _judge_resp(solver_role="weak", attempt=1, total=0.2),
        _judge_resp(solver_role="strong", attempt=0, total=0.9),
        _judge_resp(solver_role="strong", attempt=1, total=0.9),
    ]
    res = _step(item, judges, cfg=_cfg(audit_enabled=False), domain=domain, grounding=grounding)
    assert res.state.state == State.ACCEPTED
    assert not any(r.role == "audit" for r in res.new_requests)


def test_judge_policy_accept_routes_through_audit(domain, grounding):
    cfg = _cfg()
    item = _seed_item(
        grounding,
        state=State.NEED_DECISION,
        candidate=_candidate(),
        quality=QualityCheck(passed=True),
        weak_scores=(_score("weak", 0, 0.2), _score("weak", 1, 0.2)),
        strong_scores=(_score("strong", 0, 0.9), _score("strong", 1, 0.9)),
    )
    verdict = StepResponse(
        request_id=stable_id("i1", 1, "loop_judge", 0),
        role="loop_judge",
        round_n=1,
        attempt=0,
        text=json.dumps({"verdict": "accept", "grpo_suitability": "high", "reason": "ok", "suggestion": ""}),
    )
    res = _step(
        item, [verdict], cfg=cfg, domain=domain, grounding=grounding, policy=JudgePolicy(cfg.acceptance)
    )
    assert res.state.state == State.NEED_AUDIT
    assert res.new_requests[-1].role == "audit"


# NEED_AUDIT -> verdict


def test_audit_pass_accepts(domain, grounding):
    item = _audit_item(grounding)
    resp = _audit_resp(json.dumps({"passed": True, "failures": [], "notes": "clean"}))
    res = _step(item, [resp], cfg=_cfg(), domain=domain, grounding=grounding)
    assert res.state.state == State.ACCEPTED
    assert res.state.audit is not None and res.state.audit.passed
    assert res.state.pending_report is None
    assert _evaluation(res).accepted
    assert res.new_requests == ()


def test_audit_fail_bumps_round_with_feedback(domain, grounding):
    item = _audit_item(grounding)
    resp = _audit_resp(json.dumps({"passed": False, "failures": ["LEAKAGE: context names the answer"]}))
    res = _step(item, [resp], cfg=_cfg(), domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_CANDIDATE
    assert res.state.current_round == 2
    assert res.state.audit is None
    ev = _evaluation(res)
    assert ev.accepted is False
    assert ev.rejection_reasons == ["audit:LEAKAGE: context names the answer"]
    assert res.state.last_feedback == ("final audit failed: LEAKAGE: context names the answer",)
    assert [r.role for r in res.new_requests] == ["challenger"]


def test_audit_fail_at_max_rounds_rejects(domain, grounding):
    item = _audit_item(grounding)
    resp = _audit_resp(json.dumps({"passed": False, "failures": ["RUBRIC: redundant criteria"]}))
    res = _step(item, [resp], cfg=_cfg(max_rounds=1), domain=domain, grounding=grounding)
    assert res.state.state == State.REJECTED
    assert res.state.rejection_reasons == ("audit:RUBRIC: redundant criteria",)


def test_audit_parse_error_fails_closed(domain, grounding):
    item = _audit_item(grounding)
    res = _step(item, [_audit_resp("not json at all")], cfg=_cfg(), domain=domain, grounding=grounding)
    assert res.state.state == State.NEED_CANDIDATE
    assert "audit_parse_error" in _evaluation(res).rejection_reasons[0]


def test_audit_ignores_unrelated_responses(domain, grounding):
    item = _audit_item(grounding)
    unrelated = _judge_resp(solver_role="weak", attempt=0, total=0.2)
    res = _step(item, [unrelated], cfg=_cfg(), domain=domain, grounding=grounding)
    assert res.state == item
    assert res.new_requests == ()


# Auditor agent


def test_build_request_default_prompt(domain, grounding):
    req = auditor.build_request(
        item_id="i1",
        round_n=1,
        model_key="mock/x",
        candidate=_candidate(),
        grounding=GroundingItem(source_id="s1", body="Z" * 100),
        weak_scores=[_score("weak", 0, 0.2)],
        strong_scores=[_score("strong", 0, 0.9)],
        domain=domain,
        audit_cfg=AuditConfig(enabled=True, grounding_chars=10),
    )
    assert req.role == "audit" and req.json_mode
    user = json.loads(req.messages[-1]["content"])
    assert user["source_excerpt"] == "Z" * 10
    assert user["rollout_scores"]["gap"] == 0.7
    assert "ROLE:FINAL_AUDITOR" in req.messages[0]["content"]


def test_build_request_without_evidence_or_source(domain):
    req = auditor.build_request(
        item_id="i1",
        round_n=1,
        model_key="mock/x",
        candidate=_candidate(),
        grounding=GroundingItem(source_id="s1", body="Z" * 100),
        weak_scores=[],
        strong_scores=[],
        domain=domain,
        audit_cfg=AuditConfig(enabled=True, include_evidence=False, grounding_chars=0),
    )
    user = json.loads(req.messages[-1]["content"])
    assert user["source_excerpt"] is None
    assert user["rollout_scores"] is None


def test_build_request_applies_harness_audit_rules(domain, grounding):
    harness = make_harness(audit_rules=["Reject questions about publication metadata."])
    req = auditor.build_request(
        item_id="i1",
        round_n=1,
        model_key="mock/x",
        candidate=_candidate(),
        grounding=grounding,
        weak_scores=[],
        strong_scores=[],
        domain=domain,
        audit_cfg=AuditConfig(enabled=True),
        harness=harness,
    )
    assert "publication metadata" in req.messages[0]["content"]


def test_domain_audit_prompt_override(domain, monkeypatch):
    custom = [{"role": "system", "content": "CUSTOM AUDIT"}, {"role": "user", "content": "x"}]
    monkeypatch.setattr(domain, "audit_prompt", lambda candidate, grounding, evidence: custom)
    req = auditor.build_request(
        item_id="i1",
        round_n=1,
        model_key="mock/x",
        candidate=_candidate(),
        grounding=None,
        weak_scores=[],
        strong_scores=[],
        domain=domain,
        audit_cfg=AuditConfig(enabled=True),
    )
    assert req.messages[0]["content"] == "CUSTOM AUDIT"


def test_parse_audit_pass_and_fail_closed():
    ok = auditor.parse_audit(json.dumps({"passed": True, "failures": [], "notes": "n"}))
    assert ok.passed and ok.notes == "n"
    bad = auditor.parse_audit("{{nope")
    assert not bad.passed
    assert bad.failures and bad.failures[0].startswith("audit_parse_error")


# Config


def test_auditor_falls_back_to_judge():
    cfg = _cfg()
    assert model_key_for(cfg, "audit") == cfg.judge.provider_model
    cfg = _cfg().model_copy(update={"auditor": ModelConfig(provider_model="mock/other")})
    assert model_key_for(cfg, "audit") == "mock/other"


def test_batch_mode_rejects_cross_provider_auditor():
    with pytest.raises(ValidationError, match="batch_provider"):
        RunConfig(
            domain=DomainConfig(name="qa_from_documents"),
            audit=AuditConfig(enabled=True),
            auditor=ModelConfig(provider_model="anthropic/claude-x"),
            dispatcher=DispatcherConfig(mode="batch", batch_provider="openai"),
        )
    RunConfig(
        domain=DomainConfig(name="qa_from_documents"),
        audit=AuditConfig(enabled=True),
        auditor=ModelConfig(provider_model="openai/gpt-x"),
        dispatcher=DispatcherConfig(mode="batch", batch_provider="openai"),
    )
    RunConfig(
        domain=DomainConfig(name="qa_from_documents"),
        audit=AuditConfig(enabled=True),
        auditor=ModelConfig(provider_model="anthropic/claude-x"),
    )


# Persistence round-trip


def test_load_item_state_rehydrates_pending_report(tmp_path):
    store = Store(tmp_path / "run.db")
    store.create_run("r1", config={}, harness=None)
    iid = store.insert_item(run_id="r1", source_id="s1", domain="d", state="NEED_AUDIT")
    store.upsert_round(
        item_id=iid,
        round_n=1,
        candidate=_candidate(),
        quality=QualityCheck(passed=True),
        evaluation=_report(),
    )
    row = store.conn.execute("SELECT * FROM items WHERE item_id=?", (iid,)).fetchone()
    state = load_item_state(store, row)
    assert state.state == State.NEED_AUDIT
    assert state.pending_report is not None
    assert state.pending_report.accepted and state.pending_report.gap == 0.7
    store.close()


def test_accepted_extras_carries_audit_verdict(grounding):
    item = _audit_item(grounding, audit=QualityCheck(passed=True, notes="clean"), pending_report=None)
    round_obj = Round(
        refinement_round=1, candidate=_candidate(), quality=QualityCheck(passed=True), evaluation=_report()
    )
    extras = accepted_extras(item, round_obj)
    assert extras["audit"] == {"passed": True, "failures": [], "notes": "clean"}


# End-to-end through the Runner (mock provider)


def _e2e_handler(audit_body: str):
    def handler(role: str, messages):
        text = " ".join(m.get("content", "") for m in messages)
        if role == "audit" or "ROLE:FINAL_AUDITOR" in text:
            return audit_body
        if role == "challenger" or "ROLE:CHALLENGER" in text:
            return json.dumps(
                {
                    "payload": {"question": "What is the contribution?", "context": "ctx"},
                    "reference_output": "the contribution",
                    "rubric": [
                        {"id": "c1", "description": "Names contribution", "weight": 5},
                        {"id": "c2", "description": "Cites detail", "weight": 3},
                    ],
                }
            )
        if role == "reflector" or "ROLE:REFLECTION" in text:
            return json.dumps({"feedback": ["try harder"], "new_angle": "x"})
        if "ROLE:QUALITY" in text:
            return json.dumps({"passed": True, "failures": [], "notes": "ok"})
        if role == "judge":
            if "vague answer" in text:
                return json.dumps({"per_criterion": {"c1": 0.2, "c2": 0.1}, "total": 0.16})
            return json.dumps({"per_criterion": {"c1": 0.9, "c2": 0.85}, "total": 0.88})
        if role == "weak":
            return "vague answer"
        if role == "strong":
            return "specific, source-grounded answer"
        return "{}"

    return handler


def _e2e_cfg(docs_dir: Path, output_dir: Path, scenario: str) -> RunConfig:
    return RunConfig(
        run_id="audit-e2e",
        output_dir=str(output_dir),
        max_examples=2,
        domain=DomainConfig(name="qa_from_documents", params={"source_dir": str(docs_dir)}),
        loop=LoopConfig(max_rounds=2, weak_samples=2, strong_samples=2),
        acceptance=AcceptanceConfig(forbid_weak_zero=False),
        audit=AuditConfig(enabled=True),
        orchestrator=ModelConfig(provider_model=f"mock/{scenario}"),
        challenger=ModelConfig(provider_model=f"mock/{scenario}"),
        weak_solver=ModelConfig(provider_model=f"mock/{scenario}"),
        strong_solver=ModelConfig(provider_model=f"mock/{scenario}"),
        judge=ModelConfig(provider_model=f"mock/{scenario}"),
        dispatcher=DispatcherConfig(concurrency=4, items_per_advance=10, poll_interval_s=0.0),
    )


def test_audit_pass_end_to_end(sample_docs: Path, output_dir: Path):
    register_mock("audit-pass", _e2e_handler(json.dumps({"passed": True, "failures": [], "notes": "clean"})))
    runner = Runner(_e2e_cfg(sample_docs, output_dir, "audit-pass"))
    summary = runner.run()
    assert summary.accepted == 2
    assert summary.rejected == 0

    store = Store(runner.run_dir / "run.db")
    records = list(store.accepted_records("audit-e2e"))
    assert len(records) == 2
    assert all(r["audit"] == {"passed": True, "failures": [], "notes": "clean"} for r in records)
    n_audits = store.conn.execute("SELECT COUNT(*) FROM requests WHERE role='audit'").fetchone()[0]
    assert n_audits == 2
    store.close()


def test_audit_fail_end_to_end_rejects_after_max_rounds(sample_docs: Path, output_dir: Path):
    register_mock("audit-fail", _e2e_handler(json.dumps({"passed": False, "failures": ["LEAKAGE: leaks"]})))
    runner = Runner(_e2e_cfg(sample_docs, output_dir, "audit-fail"))
    summary = runner.run()
    assert summary.accepted == 0
    assert summary.rejected == 2

    store = Store(runner.run_dir / "run.db")
    reasons = store.conn.execute("SELECT rejection_reasons FROM items").fetchall()
    assert all("audit:LEAKAGE" in (r[0] or "") for r in reasons)
    # Two items exhaust two audited rounds each.
    n_audits = store.conn.execute("SELECT COUNT(*) FROM requests WHERE role='audit'").fetchone()[0]
    assert n_audits == 4
    store.close()
