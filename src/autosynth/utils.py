"""Shared ID, JSON, path, and timestamp helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*")
_FENCE_CLOSE = re.compile(r"\s*```\s*$")


def stable_id(*parts: Any, length: int = 12) -> str:
    """Deterministic short ID from arbitrary parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:length]


def utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)


def make_run_id(prefix: str, *seed_parts: Any) -> str:
    """Compose a run id of the form `<prefix>-<utc-ts>-<short-hash>`."""
    ts = utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}-{stable_id(*seed_parts, length=6)}"


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object, ignoring surrounding text."""
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_OPEN.sub("", text)
        text = _FENCE_CLOSE.sub("", text)
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(obj, dict):
            return obj

    for s in (i for i, c in enumerate(text) if c == "{"):
        depth = 0
        in_str = False
        esc = False
        for i in range(s, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def write_pydantic(path: Path, obj: BaseModel | Sequence[BaseModel]) -> None:
    """Serialize a Pydantic model (or list of models) to JSON on disk."""
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json")
    else:
        data = [m.model_dump(mode="json") for m in obj]
    write_json(path, data)


def write_yaml_snapshot(path: Path, model: BaseModel) -> None:
    """Snapshot a Pydantic model as YAML for later resume / inspection."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str))
        f.write("\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream records from a JSON-lines file, skipping blank/malformed lines."""
    from loguru import logger

    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("skip malformed jsonl line {} in {}: {}", n, path, e)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
