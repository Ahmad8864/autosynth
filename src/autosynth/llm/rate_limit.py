"""RPM rate limiting via a thread-safe token bucket."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from pydantic import BaseModel


class RateLimitSpec(BaseModel):
    """RPM only; TPM is logged but not enforced (no pre-call token counting)."""

    rpm: int | None = None  # None = no limit
    burst: int | None = None  # default = rpm / 4, min 1


class TokenBucket:
    """Thread-safe RPM token bucket. `acquire()` blocks until a slot is free.

    The clock and sleep functions are injectable for deterministic tests.
    """

    def __init__(
        self,
        rate_per_sec: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self.rate = rate_per_sec
        self.burst = burst
        self._tokens = float(burst)
        self._last = clock()
        self._lock = threading.Lock()
        self._clock = clock
        self._sleep = sleep

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> None:
        deadline = (self._clock() + timeout) if timeout is not None else None
        while True:
            with self._lock:
                now = self._clock()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                need = tokens - self._tokens
                wait = need / self.rate
            if deadline is not None and self._clock() + wait > deadline:
                raise TimeoutError("token bucket acquire timeout")
            self._sleep(wait)
