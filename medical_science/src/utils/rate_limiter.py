import time
from dataclasses import dataclass


@dataclass
class RateLimiter:
    rate_per_sec: float
    _last: float = time.time()

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last
        min_interval = 1.0 / max(self.rate_per_sec, 1e-6)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last = time.time() 