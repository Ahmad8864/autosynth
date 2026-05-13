"""Shared rich consoles.

A single stderr-bound :class:`~rich.console.Console` is reused by the
loguru sink in :mod:`autosynth.cli` and the dispatcher's progress bar.
Sharing the writer lets rich's ``Live`` display keep the bar anchored at
the bottom of the terminal while log lines scroll above it.
"""

from __future__ import annotations

from rich.console import Console

STDERR_CONSOLE = Console(stderr=True)
