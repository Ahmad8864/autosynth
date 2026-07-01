"""Boxed run summary printed once at the top of ``autosynth run``, before the
dispatcher's live progress bar takes over.

Laid out top-to-bottom rather than in columns: columns crop long model strings
at real terminal widths.
"""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from autosynth.config import RunConfig

# Fixed rather than terminal-relative: reads as a card, and stays wide enough
# that typical run ids, paths, and models fold instead of cropping.
_WIDTH = 64

_ACCENT = "#7fbf7f"  # soft green, matches the autosynth mark

_ROLES = (
    ("orchestrator", "orchestrator"),
    ("challenger", "challenger"),
    ("weak solver", "weak_solver"),
    ("strong solver", "strong_solver"),
    ("judge", "judge"),
)

# Tint each model by its provider so a mixed setup groups visually. Keys match
# the token before the first "/" in a LiteLLM model string.
_PROVIDER_STYLES = {
    "openai": "green",
    "azure": "bright_cyan",
    "anthropic": "#d97757",
    "gemini": "bright_blue",
    "google": "bright_blue",
    "vertex_ai": "bright_blue",
    "together_ai": "magenta",
    "ollama": "yellow",
    "mock": "grey50",
}


def _model_text(provider_model: str) -> Text:
    provider = provider_model.split("/", 1)[0]
    style = _PROVIDER_STYLES.get(provider, "white")
    if "/" in provider_model:
        rest = provider_model[len(provider) :]
        return Text(provider, style=style) + Text(rest, style="bold white")
    return Text(provider_model, style=f"bold {style}")


def _section(title: str, grid: Table) -> Group:
    header = Text(title.upper(), style=f"bold {_ACCENT}")
    return Group(header, Padding(grid, (0, 0, 0, 2)))


def _kv_grid() -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="grey62", justify="left", no_wrap=True)  # labels: short, known
    grid.add_column(overflow="fold")  # values: wrap rather than crop an over-long path/model
    return grid


def _models_section(cfg: RunConfig) -> Group:
    pairs = [(label, getattr(cfg, attr).provider_model) for label, attr in _ROLES]
    distinct = {model for _, model in pairs}
    grid = _kv_grid()
    if len(distinct) == 1:
        grid.add_row("all roles", _model_text(next(iter(distinct))))
    else:
        for label, model in pairs:
            grid.add_row(label, _model_text(model))
    return _section("models", grid)


def _dispatch(cfg: RunConfig) -> str:
    d = cfg.dispatcher
    if d.mode == "batch":
        return f"batch · {d.batch_provider} · {d.batch_completion_window}"
    return f"local · concurrency {d.concurrency}"


def _loop_section(cfg: RunConfig) -> Group:
    grid = _kv_grid()
    grid.add_row(
        "rounds",
        f"{cfg.loop.max_rounds} · weak×{cfg.loop.weak_samples} strong×{cfg.loop.strong_samples}",
    )
    grid.add_row("dispatch", _dispatch(cfg))
    grid.add_row(
        "budget",
        "unlimited" if cfg.budget_usd is None else f"${cfg.budget_usd:,.2f}",
    )
    return _section("loop", grid)


def _header(cfg: RunConfig) -> Group:
    domain = cfg.domain.name or cfg.domain.path or "?"
    mode = cfg.acceptance.mode or "domain default"
    dot = Text(" · ", style="grey42")
    summary = (
        Text(domain, style=f"bold {_ACCENT}")
        + dot
        + Text(f"{cfg.max_examples} examples", style="white")
        + dot
        + Text(mode, style="white")
    )
    return Group(Text("agentic synthetic data", style="grey54"), Text(), summary)


def _run_section(run_id: str, run_dir: Path, resume: bool) -> Group:
    grid = _kv_grid()
    grid.add_row("run", Text(run_id, style="bold white"))
    grid.add_row("out", Text(str(run_dir), style="white"))
    if resume:
        grid.add_row("", Text("● resuming prior run", style="bold yellow"))
    return _section("run", grid)


def render_run_intro(cfg: RunConfig, *, run_id: str, run_dir: Path, resume: bool = False) -> Panel:
    """Build the intro panel summarizing the experiment about to run."""
    body = Group(
        _header(cfg),
        Text(),
        Rule(style="grey30"),
        Text(),
        _run_section(run_id, run_dir, resume),
        Text(),
        _models_section(cfg),
        Text(),
        _loop_section(cfg),
    )
    return Panel(
        body,
        box=box.ROUNDED,
        border_style=_ACCENT,
        title=Text("◆ autosynth", style=f"bold {_ACCENT}"),
        title_align="left",
        subtitle=Text("run", style="grey42"),
        subtitle_align="right",
        padding=(1, 2),
        width=_WIDTH + 6,  # content width + 2*padding + 2 borders
    )
