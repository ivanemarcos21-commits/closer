"""Small network helpers shared by the registry client and the AI client."""

from __future__ import annotations

import ssl
import threading
import time
from collections import deque

CERTIFICATE_HELP = (
    "Could not verify the server's security certificate. A network filter (school, work or VPN) "
    "may be inspecting secure traffic - try a different network, or ask your IT administrator."
)


def is_certificate_error(exc: BaseException | None) -> bool:
    """True if ``exc``, or anything it was raised from, is a TLS certificate failure.

    Retrying cannot fix those, so callers should fail fast with ``CERTIFICATE_HELP``
    instead of the generic "try again in a moment".
    """
    for _ in range(10):  # bounded walk up the cause chain
        if exc is None:
            return False
        if isinstance(exc, ssl.SSLCertVerificationError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class SearchBudget:
    """App-wide cap on searches per hour, shared by every visitor of one server process.

    A per-visitor cooldown is easy to dodge by opening a new tab; this protects the shared
    free AI quota from being drained by a script. Thread-safe.
    """

    def __init__(self, per_hour: int, *, clock=time.monotonic) -> None:
        self._per_hour = max(1, per_hour)
        self._clock = clock
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Count one search if the hour's budget allows it; False means "too busy"."""
        now = self._clock()
        with self._lock:
            while self._times and now - self._times[0] >= 3600:
                self._times.popleft()
            if len(self._times) >= self._per_hour:
                return False
            self._times.append(now)
            return True
