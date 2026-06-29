"""Request fulfillment: the only part of the framework that touches the network.

The :class:`Dispatcher` runs the main loop; ``fulfill`` is a pluggable
strategy. :func:`fulfill_local` (thread-pool) is the default; the batch
variant lives in :mod:`autosynth.dispatcher.batch`.
"""

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
