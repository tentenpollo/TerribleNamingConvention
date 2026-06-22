from __future__ import annotations

import math

from redis.asyncio import Redis

from app.core.logging import logger


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter backed by Redis sorted sets.

    Each allowed request is recorded as a sorted-set member whose score is the
    current timestamp. A pipeline first prunes entries outside the window and
    counts the remaining entries. If the count is below the limit, a second
    pipeline records the current request and refreshes the key TTL. Rejected
    requests are not recorded, so they cannot extend the window.

    Fail-open: if Redis is unavailable the limiter logs an error and allows the
    request. A broken rate limiter must not take down querying.
    """

    def __init__(self, redis: Redis, limit: int, window_seconds: float) -> None:
        self._redis = redis
        self.limit = limit
        self.window_seconds = window_seconds

    async def is_allowed(self, key: str, now: float) -> tuple[bool, float | None]:
        """Return (allowed, retry_after_seconds).

        ``retry_after_seconds`` is ``None`` when the request is allowed, otherwise
        the integer ceiling of seconds until the oldest entry in the window expires.
        """
        window_start = now - self.window_seconds
        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, "-inf", window_start)
            pipe.zcard(key)
            prune_results = await pipe.execute()
        except Exception as exc:  # pragma: no cover - defensive; Redis failures are rare in tests
            # Fail-open: a broken limiter must not block legitimate queries.
            logger.error(
                "Rate limiter Redis call failed; allowing request",
                key=key,
                error=str(exc),
            )
            return True, None

        count = prune_results[1]
        if count >= self.limit:
            retry_after = await self._retry_after(key, now)
            return False, retry_after

        try:
            record_pipe = self._redis.pipeline()
            record_pipe.zadd(key, {str(now): now})
            record_pipe.pexpire(key, int(self.window_seconds * 1000))
            await record_pipe.execute()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Rate limiter Redis call failed while recording allowed request",
                key=key,
                error=str(exc),
            )
            return True, None

        return True, None

    async def _retry_after(self, key: str, now: float) -> float:
        """Seconds until the oldest entry in the current window expires."""
        try:
            oldest = await self._redis.zrange(key, 0, 0, withscores=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Failed to fetch oldest rate-limit entry for Retry-After",
                key=key,
                error=str(exc),
            )
            return 0.0

        if not oldest:
            return 0.0
        oldest_score = float(oldest[0][1])
        return max(0.0, math.ceil((oldest_score + self.window_seconds) - now))
