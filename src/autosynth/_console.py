"""Rich consoles shared by logging and progress output."""

from __future__ import annotations

from rich.console import Console

STDERR_CONSOLE = Console(stderr=True)
