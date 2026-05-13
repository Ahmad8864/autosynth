"""autodata CLI.

Commands:
  autodata run         --config configs/example.yaml
  autodata resume      RUN_DIR
  autodata metaopt     --config configs/metaopt.yaml
  autodata init-domain NAME --out my_domain.py
  autodata inspect-run RUN_DIR [--stuck]
  autodata status      RUN_DIR
  autodata export      --run RUN_DIR --format jsonl|hf
"""

from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path
from string import Template

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

from autodata._console import STDERR_CONSOLE
from autodata.config import RunConfig, load_config
from autodata.metaopt import MetaOptimizer
from autodata.runner import Runner
from autodata.store import Store

app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="autodata: agentic synthetic data generation",
)
console = Console()


def _configure_logging(verbose: bool) -> None:
    # Route through the shared stderr console so loguru output and the
    # dispatcher's live progress bar can coexist — rich's Live keeps the
    # bar pinned at the bottom while log lines scroll above it.
    logger.remove()
    logger.add(
        lambda msg: STDERR_CONSOLE.print(msg, end="", markup=False, highlight=False),
        level="DEBUG" if verbose else "INFO",
        format="<level>{level: <7}</level> | {message}",
        colorize=True,
    )


def _open_run(run_dir: Path) -> tuple[Store, sqlite3.Row]:
    """Open the run.db at ``run_dir`` and return (store, run_row).

    Exits with code 1 if the database or its run row is missing. Returning
    the row directly lets callers use ``row["run_id"]`` / ``row["status"]``
    without re-querying or re-narrowing past ``Row | None``.
    """
    db = run_dir / "run.db"
    if not db.exists():
        console.print(f"[red]no run.db at {db}[/red]")
        raise typer.Exit(1)
    store = Store(db)
    row = store.first_run()
    if row is None:
        console.print("[red]no run row in this database[/red]")
        raise typer.Exit(1)
    return store, row


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
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Output file (defaults to ./domains/<name>.py)",
    ),
):
    """Write a new DomainAdapter skeleton you can fill in."""
    target = out if out is not None else Path("domains") / f"{name}.py"
    if target.exists():
        console.print(f"[red]refusing to overwrite[/red] {target}")
        raise typer.Exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_domain_template(name=name, cls=_camel(name), path=target))
    console.print(f"[green]wrote[/green] {target}")
    console.print(f"Use it via:\n  domain:\n    path: {target}:{_camel(name)}")


@app.command("inspect-run")
def inspect_run_cmd(
    run_dir: Path = typer.Argument(..., exists=True),
    stuck: bool = typer.Option(False, "--stuck", help="Show only non-terminal items"),
):
    """Show per-item state, counts, and last error from the run.db."""
    store, run_row = _open_run(run_dir)
    run_id = run_row["run_id"]
    counts = store.items_terminal_counts(run_id)
    console.print(f"[bold]run:[/bold] {run_id}  [dim](status: {run_row['status']})[/dim]")
    console.print(f"[bold]cost:[/bold] ${store.cost_so_far(run_id):.4f}")
    for state, n in counts.items():
        console.print(f"  {state}: {n}")

    table = Table(title=f"items ({'stuck' if stuck else 'all'})")
    table.add_column("source_id")
    table.add_column("state")
    table.add_column("round")
    table.add_column("rejection")
    for row in store.items_for_run(run_id, stuck_only=stuck):
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
    store, run_row = _open_run(run)
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
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Override config (else use the snapshot)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Resume an interrupted run from its on-disk state."""
    _configure_logging(verbose)
    if config is not None:
        cfg = load_config(config)
    else:
        snap = run_dir / "config.snapshot.yaml"
        if not snap.exists():
            console.print(f"[red]no config to resume with at {snap}[/red]")
            raise typer.Exit(1)
        cfg = RunConfig.model_validate(yaml.safe_load(snap.read_text()))

    # `run_dir` already includes the run_id segment; the Runner appends it again.
    cfg.output_dir = str(run_dir.parent)
    cfg.resume = True
    runner = Runner(cfg, run_id=run_dir.name)
    summary = runner.run()
    console.print(
        f"resumed {runner.run_id}: accepted={summary.accepted} rejected={summary.rejected}"
    )


@app.command("status")
def status_cmd(run_dir: Path = typer.Argument(..., exists=True)):
    """One-line status: state counts + cost."""
    store, run_row = _open_run(run_dir)
    run_id = run_row["run_id"]
    counts = store.items_terminal_counts(run_id)
    console.print(
        f"{run_id}  status={run_row['status']}  "
        f"cost=${store.cost_so_far(run_id):.4f}  "
        f"states={dict(counts)}"
    )


def _camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.replace("-", "_").split("_"))


def _render_domain_template(*, name: str, cls: str, path: Path) -> str:
    """Load the packaged scaffold and substitute the domain name + class name."""
    src = files("autodata.templates").joinpath("new_domain.py.tmpl").read_text(encoding="utf-8")
    return Template(src).substitute(NAME=name, CLASS=cls, YAML_PATH=path.as_posix())


if __name__ == "__main__":
    app()
