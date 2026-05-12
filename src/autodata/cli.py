"""autodata CLI.

Commands:
  autodata run --config configs/example.yaml
  autodata metaopt --config configs/metaopt.yaml
  autodata init-domain NAME --out my_domain.py
  autodata inspect-run RUN_DIR
  autodata export --run RUN_DIR --format jsonl|hf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

from autodata.config import RunConfig, load_config
from autodata.metaopt import MetaOptimizer
from autodata.orchestrator import Orchestrator
from autodata.writer import RunWriter

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="autodata: agentic synthetic data generation"
)
console = Console()


def _configure_logging(verbose: bool) -> None:
    logger.remove()
    logger.add(
        sys.stderr, level="DEBUG" if verbose else "INFO", format="<level>{level: <7}</level> | {message}"
    )


@app.command("run")
def run_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="YAML config path"),
    run_id: str | None = typer.Option(None, "--run-id", help="Override run id"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Execute the generate→verify→evaluate→reflect→refine loop."""
    _configure_logging(verbose)
    cfg = load_config(config)
    orch = Orchestrator(cfg, run_id=run_id)
    summary = orch.run()

    table = Table(title=f"run {orch.run_id} summary")
    table.add_column("metric")
    table.add_column("value")
    for k, v in summary.items():
        table.add_row(str(k), str(v))
    console.print(table)
    console.print(f"[green]output_dir:[/green] {orch.writer.root}")


@app.command("metaopt")
def metaopt_cmd(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, help="YAML config; metaopt.enabled must be true"
    ),
    rng_seed: int = typer.Option(0, "--rng-seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the paper's meta-optimization loop on the configured harness."""
    _configure_logging(verbose)
    cfg = load_config(config)
    if not cfg.metaopt.enabled:
        console.print("[red]metaopt.enabled is false in this config[/red]")
        raise typer.Exit(2)
    opt = MetaOptimizer(cfg, rng_seed=rng_seed)
    summary = opt.run()

    table = Table(title=f"metaopt {opt.run_id}")
    table.add_column("metric")
    table.add_column("value")
    for k, v in summary.items():
        table.add_row(str(k), str(v))
    console.print(table)


@app.command("init-domain")
def init_domain_cmd(
    name: str = typer.Argument(..., help="snake_case domain name"),
    out: Path = typer.Option(Path("my_domain.py"), "--out", "-o"),
):
    """Write a new DomainAdapter skeleton you can fill in."""
    if out.exists():
        console.print(f"[red]refusing to overwrite[/red] {out}")
        raise typer.Exit(1)
    out.write_text(_DOMAIN_TEMPLATE.format(name=name, cls=_camel(name)))
    console.print(f"[green]wrote[/green] {out}")
    console.print(f"Use it via:\n  domain:\n    path: {out}:{_camel(name)}")


@app.command("inspect-run")
def inspect_run_cmd(run_dir: Path = typer.Argument(..., exists=True)):
    """Print accepted/rejected counts and the per-trajectory outcome."""
    summary = run_dir / "summary.json"
    if summary.exists():
        console.print_json(summary.read_text())
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        return
    table = Table(title="trajectories")
    table.add_column("source_id")
    table.add_column("rounds")
    table.add_column("accepted_round")
    table.add_column("final weak_avg")
    table.add_column("final strong_avg")
    table.add_column("final gap")
    for path in sorted(traj_dir.glob("*.json")):
        data = json.loads(path.read_text())
        rounds = data.get("rounds", [])
        last_ev = (rounds[-1].get("evaluation") if rounds else None) or {}
        table.add_row(
            data.get("source_id", "?"),
            str(data.get("total_rounds", 0)),
            str(data.get("final_accepted_round", "—")),
            _fmt(last_ev.get("weak_avg")),
            _fmt(last_ev.get("strong_avg")),
            _fmt(last_ev.get("gap")),
        )
    console.print(table)


@app.command("export")
def export_cmd(
    run: Path = typer.Option(..., "--run", exists=True),
    format: str = typer.Option("jsonl", "--format", "-f"),
):
    """Export the accepted dataset. format: jsonl (default) or hf."""
    accepted = run / "accepted.jsonl"
    if format == "jsonl":
        if not accepted.exists():
            console.print("[red]no accepted.jsonl found[/red]")
            raise typer.Exit(1)
        console.print(f"[green]ready:[/green] {accepted}")
        return
    if format == "hf":
        snap = run / "config.snapshot.yaml"
        if not snap.exists():
            console.print("[red]missing config.snapshot.yaml[/red]")
            raise typer.Exit(1)
        cfg = RunConfig.model_validate(yaml.safe_load(snap.read_text()))
        cfg.output_dir = str(run.parent)
        writer = RunWriter(cfg, run.name)
        out = writer.export_hf()
        if out:
            console.print(f"[green]exported HF dataset to[/green] {out}")
        else:
            raise typer.Exit(1)
        return
    console.print(f"[red]unknown format[/red] {format}")
    raise typer.Exit(2)


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def _camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.replace("-", "_").split("_"))


_DOMAIN_TEMPLATE = '''"""Custom domain: {name}."""
from __future__ import annotations

import json
from typing import Any, Iterable

from autodata.domain import DomainAdapter, GroundingItem, register_domain
from autodata.schemas import Candidate


@register_domain("{name}")
class {cls}(DomainAdapter):
    description = "TODO: describe the dataset you are constructing."

    def __init__(self, **params: Any):
        super().__init__(**params)
        # Read any config-driven params from self.params here.

    def load_grounding(self) -> Iterable[GroundingItem]:
        # TODO: yield GroundingItem(source_id=..., body=..., metadata={{}})
        raise NotImplementedError

    def generation_prompt(self, item, feedback, round_n, prior_payloads):
        sys_msg = "ROLE:CHALLENGER. ... return JSON with payload, reference_output, rubric."
        usr_msg = f"ROUND={{round_n}}\\nSOURCE: {{item.body}}\\nFEEDBACK: {{feedback}}"
        return [{{"role": "system", "content": sys_msg}}, {{"role": "user", "content": usr_msg}}]

    def validate_candidate(self, candidate: Candidate) -> list[str]:
        return []  # return non-empty list to reject

    def solver_prompt(self, candidate: Candidate, solver_role: str):
        return [{{"role": "user", "content": json.dumps(candidate.payload)}}]

    def quality_prompt(self, candidate: Candidate):
        return [{{"role": "user", "content": "ROLE:QUALITY. Audit candidate; return JSON {{passed,failures,notes}}"}}]

    def judge_prompt(self, candidate: Candidate, solver_response: str, solver_role: str):
        return [{{"role": "user", "content": f"ROLE:JUDGE. Score against rubric. solver={{solver_role}} resp={{solver_response}}"}}]
'''


if __name__ == "__main__":
    app()
