"""Local and batch request fulfillment."""

from __future__ import annotations

from autosynth.dispatcher.batch import (
    AnthropicBatchProvider,
    BatchHandle,
    BatchProvider,
    BatchResult,
    LiteLLMBatchProvider,
    MockBatchProvider,
    make_fulfill_batch,
    poll_outstanding_batches,
)
from autosynth.dispatcher.core import Dispatcher, Fulfill, RunSummary
from autosynth.dispatcher.local import fulfill_local

__all__ = [
    "Dispatcher",
    "RunSummary",
    "Fulfill",
    "fulfill_local",
    "BatchHandle",
    "BatchResult",
    "BatchProvider",
    "MockBatchProvider",
    "LiteLLMBatchProvider",
    "AnthropicBatchProvider",
    "make_fulfill_batch",
    "poll_outstanding_batches",
]
