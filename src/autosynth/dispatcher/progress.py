"""Live dispatcher progress, disabled for non-interactive output."""

from __future__ import annotations

from types import TracebackType

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from autosynth._console import STDERR_CONSOLE


class DispatcherProgress:
    """Context-managed progress counters for one run."""

    def __init__(self, *, total: int) -> None:
        self._total = total
        self._enabled = total > 0 and STDERR_CONSOLE.is_terminal
        self._progress: Progress | None = None
        self._task: TaskID | None = None

    def __enter__(self) -> DispatcherProgress:
        if not self._enabled:
            return self
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]dispatcher"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[stats]}"),
            TimeElapsedColumn(),
            console=STDERR_CONSOLE,
            transient=False,
        )
        self._progress.__enter__()
        self._task = self._progress.add_task(
            "dispatcher",
            total=self._total,
            stats=_fmt_stats(0, 0, 0, 0.0),
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._progress is None:
            return
        self._progress.__exit__(exc_type, exc, tb)
        self._progress = None
        self._task = None

    def update(
        self,
        *,
        accepted: int,
        rejected: int,
        in_flight: int,
        cost_usd: float,
    ) -> None:
        if self._progress is None or self._task is None:
            return
        self._progress.update(
            self._task,
            completed=accepted + rejected,
            stats=_fmt_stats(accepted, rejected, in_flight, cost_usd),
        )


def _fmt_stats(accepted: int, rejected: int, in_flight: int, cost_usd: float) -> str:
    return (
        f"[green]accepted={accepted}[/green] "
        f"[red]rejected={rejected}[/red] "
        f"[cyan]in_flight={in_flight}[/cyan] "
        f"[yellow]cost=${cost_usd:.4f}[/yellow]"
    )
