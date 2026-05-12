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
from autodata.runner import Runner
from autodata.store import Store

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
    resume: str | None = typer.Option(None, "--resume", help="Resume a prior run by id"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Execute the generate→verify→evaluate→reflect→refine loop."""
    _configure_logging(verbose)
    cfg = load_config(config)
    if resume:
        run_id = resume
        cfg.resume = True
    runner = Runner(cfg, run_id=run_id)
    summary = runner.run()

    table = Table(title=f"run {runner.run_id} summary")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("accepted", str(summary.accepted))
    table.add_row("rejected", str(summary.rejected))
    table.add_row("cost_usd", f"{summary.cost_usd:.4f}")
    for k, v in summary.state_counts.items():
        table.add_row(f"  state:{k}", str(v))
    console.print(table)
    console.print(f"[green]output_dir:[/green] {runner.run_dir}")
    console.print(f"[dim]export with: autodata export --run {runner.run_dir} --format jsonl[/dim]")


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
def inspect_run_cmd(
    run_dir: Path = typer.Argument(..., exists=True),
    stuck: bool = typer.Option(False, "--stuck", help="Show only non-terminal items"),
):
    """Show per-item state, counts, and last error from the run.db."""
    db = run_dir / "run.db"
    if not db.exists():
        console.print(f"[red]no run.db at {db}[/red]")
        raise typer.Exit(1)
    store = Store(db)
    run_row = next(iter(store.conn.execute("SELECT * FROM runs").fetchall()), None)
    if run_row is None:
        console.print("[red]no run row in this database[/red]")
        raise typer.Exit(1)
    run_id = run_row["run_id"]
    counts = store.items_terminal_counts(run_id)
    cost = store.cost_so_far(run_id)
    console.print(f"[bold]run:[/bold] {run_id}  [dim](status: {run_row['status']})[/dim]")
    console.print(f"[bold]cost:[/bold] ${cost:.4f}")
    for state, n in counts.items():
        console.print(f"  {state}: {n}")

    where = "AND state NOT IN ('ACCEPTED','REJECTED')" if stuck else ""
    rows = store.conn.execute(
        f"""SELECT item_id, source_id, state, current_round, final_round, rejection_reasons
            FROM items WHERE run_id=? {where} ORDER BY updated_at""",
        (run_id,),
    ).fetchall()
    table = Table(title=f"items ({'stuck' if stuck else 'all'})")
    table.add_column("source_id")
    table.add_column("state")
    table.add_column("round")
    table.add_column("rejection")
    for row in rows:
        table.add_row(
            row["source_id"],
            row["state"],
            str(row["current_round"]),
            (row["rejection_reasons"] or "")[:80],
        )
    console.print(table)


@app.command("export")
def export_cmd(
    run: Path = typer.Option(..., "--run", exists=True),
    format: str = typer.Option("jsonl", "--format", "-f"),
    out: Path | None = typer.Option(None, "--out", "-o"),
):
    """Export the accepted dataset from a run's run.db."""
    db = run / "run.db"
    if not db.exists():
        console.print(f"[red]no run.db at {db}[/red]")
        raise typer.Exit(1)
    store = Store(db)
    run_row = next(iter(store.conn.execute("SELECT run_id FROM runs LIMIT 1")), None)
    if run_row is None:
        console.print("[red]no run row[/red]")
        raise typer.Exit(1)
    run_id = run_row["run_id"]
    if format == "jsonl":
        target = out or (run / "accepted.jsonl")
        n = store.export_jsonl(run_id, target)
        console.print(f"[green]wrote {n} records to[/green] {target}")
        return
    if format == "hf":
        target = out or (run / "hf_export")
        result = store.export_hf(run_id, target)
        if result is None:
            console.print("[red]hf export failed (install autodata[hf]?)[/red]")
            raise typer.Exit(1)
        console.print(f"[green]exported HF dataset to[/green] {result}")
        return
    console.print(f"[red]unknown format[/red] {format}")
    raise typer.Exit(2)


@app.command("resume")
def resume_cmd(
    run_dir: Path = typer.Argument(..., exists=True),
    config: Path | None = typer.Option(None, "--config", "-c",
                                       help="Override config (else use the snapshot)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Resume an interrupted run from its on-disk state."""
    _configure_logging(verbose)
    snap = run_dir / "config.snapshot.yaml"
    cfg_path = config or snap
    if not cfg_path.exists():
        console.print(f"[red]no config to resume with at {cfg_path}[/red]")
        raise typer.Exit(1)
    cfg = load_config(cfg_path) if cfg_path == config else RunConfig.model_validate(
        yaml.safe_load(snap.read_text())
    )
    # Output already includes the run_id segment; strip it for the runner.
    cfg.output_dir = str(run_dir.parent)
    cfg.resume = True
    runner = Runner(cfg, run_id=run_dir.name)
    summary = runner.run()
    console.print(f"resumed {runner.run_id}: accepted={summary.accepted} rejected={summary.rejected}")


@app.command("status")
def status_cmd(run_dir: Path = typer.Argument(..., exists=True)):
    """One-shot status table: state counts + cost. Same data as inspect-run, terser."""
    db = run_dir / "run.db"
    if not db.exists():
        console.print(f"[red]no run.db at {db}[/red]")
        raise typer.Exit(1)
    store = Store(db)
    run_row = next(iter(store.conn.execute("SELECT * FROM runs LIMIT 1")), None)
    if run_row is None:
        raise typer.Exit(1)
    counts = store.items_terminal_counts(run_row["run_id"])
    console.print(f"{run_row['run_id']}  status={run_row['status']}  "
                  f"cost=${store.cost_so_far(run_row['run_id']):.4f}  "
                  f"states={dict(counts)}")


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
