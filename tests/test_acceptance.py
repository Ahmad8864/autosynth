import pytest
from loguru import logger

from autosynth.acceptance import (
    JudgePolicy,
    ThresholdPolicy,
    VerifiablePolicy,
    evaluate,
    resolve_policy,
    weak_gate_report,
)
from autosynth.agents.loop_judge import LoopJudgeVerdict, parse_verdict
from autosynth.config import AcceptanceConfig, DomainConfig, LoopConfig, RunConfig
from autosynth.domain import DomainAdapter
from autosynth.domains.math_word_problems import MathWordProblems
from autosynth.domains.qa_from_documents import QAFromDocuments
from autosynth.schemas import QualityCheck, SolverScore


def _score(solver: str, vals: list[float], correct: list[bool | None] | None = None) -> list[SolverScore]:
    out = []
    for i, v in enumerate(vals):
        c = correct[i] if correct is not None else None
        out.append(SolverScore(solver=solver, attempt=i, raw_response="r", total=v, correct=c))
    return out


# Rubric-gap acceptance


def test_accept_paper_defaults():
    rep = evaluate(
        weak_scores=_score("weak", [0.2, 0.3, 0.4]),
        strong_scores=_score("strong", [0.7, 0.8, 0.9]),
        quality=QualityCheck(passed=True),
        criteria=AcceptanceConfig(),
    )
    assert rep.accepted, rep.rejection_reasons
    assert rep.gap > 0.2


def test_reject_weak_too_high():
    rep = evaluate(
        weak_scores=_score("weak", [0.7, 0.8]),
        strong_scores=_score("strong", [0.9, 0.95]),
        quality=QualityCheck(passed=True),
        criteria=AcceptanceConfig(),
    )
    assert not rep.accepted
    assert any("weak_avg" in r for r in rep.rejection_reasons)


def test_reject_strong_too_low():
    rep = evaluate(
        weak_scores=_score("weak", [0.1, 0.2]),
        strong_scores=_score("strong", [0.3, 0.4]),
        quality=QualityCheck(passed=True),
        criteria=AcceptanceConfig(),
    )
    assert not rep.accepted
    assert any("strong_avg" in r for r in rep.rejection_reasons)


def test_reject_too_easy_ceiling():
    rep = evaluate(
        weak_scores=_score("weak", [0.1, 0.2]),
        strong_scores=_score("strong", [0.98, 0.99]),
        quality=QualityCheck(passed=True),
        criteria=AcceptanceConfig(strong_avg_max=0.95),
    )
    assert not rep.accepted
    assert any("ceiling" in r for r in rep.rejection_reasons)


def test_reject_quality_failed():
    rep = evaluate(
        weak_scores=_score("weak", [0.2]),
        strong_scores=_score("strong", [0.8]),
        quality=QualityCheck(passed=False, failures=["leak"]),
        criteria=AcceptanceConfig(),
    )
    assert not rep.accepted
    assert any("quality_failed" in r for r in rep.rejection_reasons)


def test_reject_weak_zero():
    rep = evaluate(
        weak_scores=_score("weak", [0.0, 0.3]),
        strong_scores=_score("strong", [0.8, 0.85]),
        quality=QualityCheck(passed=True),
        criteria=AcceptanceConfig(forbid_weak_zero=True),
    )
    assert not rep.accepted
    assert any("forbid_weak_zero" in r for r in rep.rejection_reasons)


def test_reject_gap_too_small():
    rep = evaluate(
        weak_scores=_score("weak", [0.5]),
        strong_scores=_score("strong", [0.65]),
        quality=QualityCheck(passed=True),
        criteria=AcceptanceConfig(min_gap=0.3),
    )
    assert not rep.accepted
    assert any("gap" in r for r in rep.rejection_reasons)


def test_threshold_policy_matches_evaluate():
    crit = AcceptanceConfig()
    weak, strong, q = _score("weak", [0.2]), _score("strong", [0.8]), QualityCheck(passed=True)
    pol = ThresholdPolicy(crit)
    assert pol.requires_judge is True
    assert pol.evaluate(weak, strong, q).accepted == evaluate(weak, strong, q, crit).accepted
    assert pol.weak_ceiling == crit.weak_avg_max
    assert pol.strong_floor == crit.strong_avg_min


def test_report_includes_rollout_std():
    rep = evaluate(
        weak_scores=_score("weak", [0.2, 0.4]),
        strong_scores=_score("strong", [0.7, 0.9]),
        quality=QualityCheck(passed=True),
        criteria=AcceptanceConfig(),
    )
    assert rep.weak_std == pytest.approx(0.1)
    assert rep.strong_std == pytest.approx(0.1)


def test_report_std_zero_for_single_rollout():
    rep = evaluate(
        weak_scores=_score("weak", [0.3]),
        strong_scores=_score("strong", [0.8]),
        quality=QualityCheck(passed=True),
        criteria=AcceptanceConfig(),
    )
    assert rep.weak_std == 0.0 and rep.strong_std == 0.0


# Verifiable acceptance


def _verifiable_policy(**kw) -> VerifiablePolicy:
    return VerifiablePolicy(AcceptanceConfig(**kw), weak_samples=4, strong_samples=4)


def test_verifiable_accepts_weak_fail_strong_succeed():
    rep = _verifiable_policy().evaluate(
        _score("weak", [1.0, 0.0, 0.0, 0.0]),
        _score("strong", [1.0, 1.0, 1.0, 0.0]),
        QualityCheck(passed=True),
    )
    assert rep.accepted, rep.rejection_reasons
    assert rep.weak_avg == 0.25 and rep.strong_avg == 0.75


def test_verifiable_rejects_weak_too_capable():
    rep = _verifiable_policy().evaluate(
        _score("weak", [1.0, 1.0, 0.0, 0.0]),
        _score("strong", [1.0, 1.0, 1.0, 1.0]),
        QualityCheck(passed=True),
    )
    assert not rep.accepted
    assert any("weak_correct" in r for r in rep.rejection_reasons)


def test_verifiable_rejects_strong_too_weak():
    rep = _verifiable_policy().evaluate(
        _score("weak", [0.0, 0.0, 0.0, 0.0]),
        _score("strong", [1.0, 1.0, 0.0, 0.0]),
        QualityCheck(passed=True),
    )
    assert not rep.accepted
    assert any("strong_correct" in r for r in rep.rejection_reasons)


def test_verifiable_rejects_on_quality():
    rep = _verifiable_policy().evaluate(
        _score("weak", [0.0, 0.0, 0.0, 0.0]),
        _score("strong", [1.0, 1.0, 1.0, 1.0]),
        QualityCheck(passed=False, failures=["bad"]),
    )
    assert not rep.accepted
    assert any("quality_failed" in r for r in rep.rejection_reasons)


def test_verifiable_gate_reads_total_not_correct():
    # An unverifiable attempt counts as incorrect.
    rep = _verifiable_policy().evaluate(
        _score("weak", [0.0, 0.0, 0.0, 0.0], correct=[None, None, False, False]),
        _score("strong", [1.0, 1.0, 1.0, 0.0], correct=[True, True, True, None]),
        QualityCheck(passed=True),
    )
    assert rep.accepted, rep.rejection_reasons


# Weak gate for short-circuit evaluation


def test_threshold_weak_gate_passes_when_weak_appropriately_fails():
    pol = ThresholdPolicy(AcceptanceConfig(forbid_weak_zero=False))
    assert pol.weak_gate_passes(_score("weak", [0.2, 0.3])) is True
    assert pol.weak_gate_passes(_score("weak", [0.7, 0.8])) is False


def test_verifiable_weak_gate_passes_on_count():
    pol = _verifiable_policy()
    assert pol.weak_gate_passes(_score("weak", [1.0, 0.0, 0.0, 0.0])) is True
    assert pol.weak_gate_passes(_score("weak", [1.0, 1.0, 0.0, 0.0])) is False


def test_base_weak_gate_defaults_true():
    assert JudgePolicy(AcceptanceConfig()).weak_gate_passes([]) is True


def test_weak_gate_report_is_explicit_and_strongless():
    rep = weak_gate_report(_score("weak", [0.7]))
    assert rep.accepted is False
    assert any("short_circuit" in r for r in rep.rejection_reasons)
    # Missing strong attempts should not look like a strong-solver failure.
    assert rep.strong_scores == []


# Judge-driven acceptance


def test_judge_policy_flags():
    pol = JudgePolicy(AcceptanceConfig())
    assert pol.decides_async is True and pol.requires_judge is True


def test_judge_policy_decide_accept():
    v = LoopJudgeVerdict(accept=True, grpo_suitability="high", reason="clean separation", suggestion="")
    d = JudgePolicy(AcceptanceConfig()).decide(
        v, _score("weak", [0.2]), _score("strong", [0.8]), QualityCheck(passed=True)
    )
    assert d.report.accepted is True and d.feedback == ()
    assert "grpo_suitability=high" in (d.report.acceptance_rationale or "")


def test_judge_policy_decide_improve_carries_suggestion():
    v = LoopJudgeVerdict(accept=False, grpo_suitability="low", reason="too easy", suggestion="make it harder")
    d = JudgePolicy(AcceptanceConfig()).decide(
        v, _score("weak", [0.7]), _score("strong", [0.7]), QualityCheck(passed=True)
    )
    assert d.report.accepted is False and d.feedback == ("make it harder",)


def test_judge_policy_improve_feedback_never_empty():
    # Keep retries actionable even when the judge omits its suggestion.
    v = LoopJudgeVerdict(accept=False, grpo_suitability="low", reason="", suggestion="")
    d = JudgePolicy(AcceptanceConfig()).decide(v, [], [], QualityCheck(passed=True))
    assert d.report.accepted is False and d.feedback and d.feedback[0]


def test_parse_verdict_improve_accept_and_fallback():
    v = parse_verdict(
        '{"verdict": "improve", "grpo_suitability": "medium", "reason": "r", "suggestion": "s"}'
    )
    assert v.accept is False and v.suggestion == "s"
    assert parse_verdict('{"verdict": "accept"}').accept is True
    assert parse_verdict("not json").accept is False


# Policy resolution


def _cfg(domain_name: str, *, mode=None, weak=4, strong=4, **acc) -> RunConfig:
    return RunConfig(
        domain=DomainConfig(name=domain_name),
        loop=LoopConfig(weak_samples=weak, strong_samples=strong),
        acceptance=AcceptanceConfig(mode=mode, **acc),
    )


def test_resolve_policy_uses_domain_default():
    assert isinstance(
        resolve_policy(_cfg("qa_from_documents"), QAFromDocuments(source_dir=".")), ThresholdPolicy
    )
    assert isinstance(resolve_policy(_cfg("math_word_problems"), MathWordProblems()), VerifiablePolicy)


def test_resolve_policy_config_overrides_domain():
    pol = resolve_policy(_cfg("math_word_problems", mode="rubric"), MathWordProblems())
    assert isinstance(pol, ThresholdPolicy)


def test_resolve_policy_judge_mode():
    pol = resolve_policy(_cfg("qa_from_documents", mode="judge"), QAFromDocuments(source_dir="."))
    assert isinstance(pol, JudgePolicy)


def test_resolve_policy_verifiable_requires_verify():
    class NoVerify(DomainAdapter):
        name = "no_verify"
        default_acceptance_mode = "verifiable"

        def load_grounding(self):
            raise NotImplementedError

        def generation_prompt(self, item, feedback, round_n, prior_payloads):
            raise NotImplementedError

        def validate_candidate(self, candidate):
            raise NotImplementedError

        def solver_prompt(self, candidate):
            raise NotImplementedError

        def quality_prompt(self, candidate):
            raise NotImplementedError

        def judge_prompt(self, candidate, solver_response):
            raise NotImplementedError

    with pytest.raises(ValueError, match="verify"):
        resolve_policy(_cfg("no_verify", mode="verifiable"), NoVerify())


def test_resolve_policy_rejects_unsatisfiable_gate():
    with pytest.raises(ValueError, match="unsatisfiable"):
        resolve_policy(
            _cfg("math_word_problems", mode="verifiable", strong=2, verifiable_strong_min_correct=3),
            MathWordProblems(),
        )


def test_resolve_policy_rejects_vacuous_weak_gate():
    with pytest.raises(ValueError, match="always passes"):
        resolve_policy(
            _cfg("math_word_problems", mode="verifiable", weak=2, verifiable_weak_max_correct=2),
            MathWordProblems(),
        )


def test_resolve_policy_warns_on_forced_perfection():
    # With three strong samples, the default minimum requires a perfect result.
    msgs: list[str] = []
    sink_id = logger.add(msgs.append, level="WARNING")
    try:
        pol = resolve_policy(
            _cfg("math_word_problems", mode="verifiable", weak=3, strong=3), MathWordProblems()
        )
    finally:
        logger.remove(sink_id)
    assert isinstance(pol, VerifiablePolicy)
    assert any("ALL rollouts" in m for m in msgs)
