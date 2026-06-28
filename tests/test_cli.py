"""CLI guards."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from autosynth.cli import app

runner = CliRunner()


def test_run_resume_errors_when_run_missing(tmp_path: Path):
    """`run --resume ID` must fail loudly when no run.db exists for ID under the
    config's output_dir, rather than silently seeding a fresh empty run."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path}\ndomain: {{name: qa_from_documents, params: {{source_dir: {tmp_path}}}}}\n"
    )
    result = runner.invoke(app, ["run", "--config", str(cfg), "--resume", "ghost"])
    assert result.exit_code == 1
    assert "no run to resume" in result.stdout
