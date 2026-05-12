from autodata.config import AcceptanceConfig
from autodata.evaluator import evaluate
from autodata.schemas import QualityCheck, SolverScore


def _score(solver: str, vals: list[float]) -> list[SolverScore]:
    return [SolverScore(solver=solver, attempt=i, raw_response="r", total=v) for i, v in enumerate(vals)]


def test_accept_paper_defaults():
    crit = AcceptanceConfig()
    rep = evaluate(
        weak_scores=_score("weak", [0.2, 0.3, 0.4]),
        strong_scores=_score("strong", [0.7, 0.8, 0.9]),
        quality=QualityCheck(passed=True),
        criteria=crit,
    )
    assert rep.accepted, rep.rejection_reasons
    assert rep.gap > 0.2


def test_reject_weak_too_high():
    crit = AcceptanceConfig()
    rep = evaluate(
        weak_scores=_score("weak", [0.7, 0.8]),
        strong_scores=_score("strong", [0.9, 0.95]),
        quality=QualityCheck(passed=True),
        criteria=crit,
    )
    assert not rep.accepted
    assert any("weak_avg" in r for r in rep.rejection_reasons)


def test_reject_strong_too_low():
    crit = AcceptanceConfig()
    rep = evaluate(
        weak_scores=_score("weak", [0.1, 0.2]),
        strong_scores=_score("strong", [0.3, 0.4]),
        quality=QualityCheck(passed=True),
        criteria=crit,
    )
    assert not rep.accepted
    assert any("strong_avg" in r for r in rep.rejection_reasons)


def test_reject_too_easy_ceiling():
    crit = AcceptanceConfig(strong_avg_max=0.95)
    rep = evaluate(
        weak_scores=_score("weak", [0.1, 0.2]),
        strong_scores=_score("strong", [0.98, 0.99]),
        quality=QualityCheck(passed=True),
        criteria=crit,
    )
    assert not rep.accepted
    assert any("ceiling" in r for r in rep.rejection_reasons)


def test_reject_quality_failed():
    crit = AcceptanceConfig()
    rep = evaluate(
        weak_scores=_score("weak", [0.2]),
        strong_scores=_score("strong", [0.8]),
        quality=QualityCheck(passed=False, failures=["leak"]),
        criteria=crit,
    )
    assert not rep.accepted
    assert any("quality_failed" in r for r in rep.rejection_reasons)


def test_reject_weak_zero():
    crit = AcceptanceConfig(forbid_weak_zero=True)
    rep = evaluate(
        weak_scores=_score("weak", [0.0, 0.3]),
        strong_scores=_score("strong", [0.8, 0.85]),
        quality=QualityCheck(passed=True),
        criteria=crit,
    )
    assert not rep.accepted
    assert any("forbid_weak_zero" in r for r in rep.rejection_reasons)


def test_reject_gap_too_small():
    crit = AcceptanceConfig(min_gap=0.3)
    rep = evaluate(
        weak_scores=_score("weak", [0.5]),
        strong_scores=_score("strong", [0.65]),
        quality=QualityCheck(passed=True),
        criteria=crit,
    )
    assert not rep.accepted
    assert any("gap" in r for r in rep.rejection_reasons)
