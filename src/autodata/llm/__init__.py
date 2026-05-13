"""LLM client for the event-sourced pipeline.

One class. Handles:

  - provider routing (LiteLLM for real models, in-process mock for ``mock/*``)
  - per-(provider, model) RPM rate limiting via a token bucket
  - retries via tenacity (exponential backoff with jitter)
  - cost accounting via :func:`litellm.completion_cost` (with per-model
    overrides registered through :attr:`LLMConfig.prices`)

The pipeline emits :class:`LLMRequest` objects; the dispatcher passes them
to :meth:`LLMClient.complete`, which returns a :class:`Response`. Mock
scenarios are registered via :func:`register_mock` and addressable as the
``mock/<scenario>`` model string.
"""

from __future__ import annotations

from autodata.llm.client import LLMClient, LLMConfig
from autodata.llm.mock import MockHandler, register_mock
from autodata.llm.rate_limit import RateLimitSpec, TokenBucket
from autodata.llm.types import ROLES, LLMRequest, Message, Response

__all__ = [
    "LLMRequest",
    "Response",
    "LLMClient",
    "LLMConfig",
    "RateLimitSpec",
    "Message",
    "MockHandler",
    "TokenBucket",
    "ROLES",
    "register_mock",
]
