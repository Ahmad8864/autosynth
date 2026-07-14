"""Wire types shared by the pipeline, dispatcher, and LLM client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from autosynth.utils import extract_json

if TYPE_CHECKING:
    from pydantic import BaseModel

Message = dict[str, str]


@dataclass(frozen=True)
class LLMRequest:
    """One LLM call. Constructed by the pipeline, fulfilled by the dispatcher."""

    request_id: str
    item_id: str
    round_n: int
    role: str
    model_key: str
    messages: list[Message]
    json_mode: bool = False
    attempt: int = 0
    parent_response_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # In-memory only; never persisted (rebuilt during hydration from role+domain).
    response_schema: type[BaseModel] | None = None


@dataclass(frozen=True)
class Response:
    """Provider response in the shape the store and pipeline both consume."""

    request_id: str
    model: str
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: int = 0

    def parse_json(self) -> dict[str, Any]:
        return extract_json(self.text)
