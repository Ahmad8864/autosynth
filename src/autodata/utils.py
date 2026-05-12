"""Utility helpers: deterministic IDs, JSON extraction, path helpers."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def stable_id(*parts: Any, length: int = 12) -> str:
    """Deterministic short ID from arbitrary parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:length]


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object or array from a string.

    Robust to chatty preludes ("Here's the JSON: { ... }") and trailing prose.
    Tries direct parse first, then progressively wider slices.
    """
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    # Strip common code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { or [ and last matching } or ]
    starts = [i for i, c in enumerate(text) if c in "{["]
    if not starts:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    for s in starts:
        opener = text[s]
        closer = "}" if opener == "{" else "]"
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
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    chunk = text[s : i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"failed to parse JSON from response: {text[:200]!r}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str))
        f.write("\n")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
