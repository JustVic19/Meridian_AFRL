"""SEC fair-access spacing. The clock and sleep function are injected in tests."""

from collections.abc import Callable

SEC_MAX_REQUESTS_PER_SECOND = 10
SEC_MIN_INTERVAL_SECONDS = 1 / SEC_MAX_REQUESTS_PER_SECOND


class RateLimiter:
    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be zero or positive")
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = 0.0

    def acquire(self) -> None:
        now = self._clock()
        if now < self._next_allowed:
            self._sleep(self._next_allowed - now)
            now = self._clock()
        self._next_allowed = now + self._min_interval
