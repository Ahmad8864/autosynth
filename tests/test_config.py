"""Config validation: strict unknown-key rejection, bundled-config sanity, env interpolation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autosynth.config import RunConfig, load_config, load_snapshot

_CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml"))


def _min_run(**extra) -> dict:
    return {"domain": {"name": "qa_from_documents"}, **extra}


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_min_run(budget_usdd=1.0))  # typo'd budget_usd


def test_unknown_nested_key_rejected():
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_min_run(acceptance={"weak_avg_maxx": 0.6}))  # typo'd weak_avg_max


def test_known_keys_validate():
    cfg = RunConfig.model_validate(_min_run(acceptance={"weak_avg_max": 0.5}))
    assert cfg.acceptance.weak_avg_max == 0.5


def test_load_snapshot_ignores_removed_keys(tmp_path: Path):
    """A snapshot resumes even with keys removed since it was written; the strict
    loader still rejects the same file."""
    snap = tmp_path / "config.snapshot.yaml"
    snap.write_text(
        "domain: {name: qa_from_documents}\n"
        "seed: 0\n"  # removed top-level key
        "loop: {max_rounds: 4, stop_on_first_accept: true}\n"  # removed nested key
    )
    cfg = load_snapshot(snap)
    assert cfg.loop.max_rounds == 4
    with pytest.raises(ValidationError):
        load_config(snap)


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_bundled_config_loads(path: Path):
    """Every shipped config must validate — catches typos / stray keys like a
    missing space in ``strong_solver:{...}`` that strict mode now rejects."""
    load_config(path)


def test_env_var_interpolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MY_MODEL", "openai/gpt-4o")
    p = tmp_path / "c.yaml"
    p.write_text(
        "domain: {name: qa_from_documents}\n"
        "judge: {provider_model: '${MY_MODEL}'}\n"
        "weak_solver: {provider_model: '${MISSING:mock/scripted}'}\n"
    )
    cfg = load_config(p)
    assert cfg.judge.provider_model == "openai/gpt-4o"
    assert cfg.weak_solver.provider_model == "mock/scripted"  # default branch
