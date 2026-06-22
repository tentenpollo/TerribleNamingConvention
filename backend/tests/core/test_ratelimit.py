from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.ratelimit import SlidingWindowRateLimiter


@pytest.fixture
def mock_redis() -> MagicMock:
    # redis.asyncio.Redis.pipeline() is synchronous; only execute() and zrange() are async.
    redis = MagicMock()
    pipeline = MagicMock()
    redis.pipeline.return_value = pipeline
    pipeline.execute = AsyncMock()
    redis.zrange = AsyncMock()
    return redis


@pytest.mark.unit
async def test_limiter_allows_requests_within_limit(mock_redis: MagicMock) -> None:
    limiter = SlidingWindowRateLimiter(
        redis=mock_redis,
        limit=3,
        window_seconds=60.0,
    )
    pipeline = mock_redis.pipeline.return_value
    pipeline.execute.side_effect = [
        [0, 0],  # prune + count
        [1, True],  # zadd + pexpire
    ]

    allowed, retry_after = await limiter.is_allowed("rl:test", now=1000.0)

    assert allowed is True
    assert retry_after is None
    assert mock_redis.pipeline.call_count == 2
    pipeline.zremrangebyscore.assert_called_once_with("rl:test", "-inf", 940.0)
    pipeline.zcard.assert_called_once_with("rl:test")
    pipeline.zadd.assert_called_once_with("rl:test", {"1000.0": 1000.0})
    pipeline.pexpire.assert_called_once_with("rl:test", 60000)


@pytest.mark.unit
async def test_limiter_rejects_request_over_limit_with_retry_after(
    mock_redis: MagicMock,
) -> None:
    limiter = SlidingWindowRateLimiter(
        redis=mock_redis,
        limit=2,
        window_seconds=60.0,
    )
    pipeline = mock_redis.pipeline.return_value
    pipeline.execute.return_value = [0, 3]
    mock_redis.zrange.return_value = [("999.0", 999.0)]

    allowed, retry_after = await limiter.is_allowed("rl:test", now=1000.0)

    assert allowed is False
    assert retry_after == 59
    pipeline.zadd.assert_not_called()
    pipeline.pexpire.assert_not_called()
    mock_redis.zrange.assert_awaited_once_with("rl:test", 0, 0, withscores=True)


@pytest.mark.unit
async def test_limiter_allows_again_after_window_slides(mock_redis: MagicMock) -> None:
    limiter = SlidingWindowRateLimiter(
        redis=mock_redis,
        limit=2,
        window_seconds=60.0,
    )
    pipeline = mock_redis.pipeline.return_value

    # Four limiter calls:
    # 1) t=1000, count=0 -> allow (prune + count, then zadd + pexpire)
    # 2) t=1001, count=1 -> allow
    # 3) t=1002, count=2 -> reject (no zadd)
    # 4) t=1061, prune removes t=1000, count=1 -> allow
    pipeline.execute.side_effect = [
        [0, 0],
        [1, True],
        [0, 1],
        [1, True],
        [0, 2],
        [1, 1],
        [1, True],
    ]
    mock_redis.zrange.side_effect = [
        [("1000.0", 1000.0)],  # retry_after for rejected t=1002
        [("1001.0", 1001.0)],  # retry_after for t=1061 (not used since allowed)
    ]

    allowed, _ = await limiter.is_allowed("rl:test", now=1000.0)
    assert allowed is True

    allowed, _ = await limiter.is_allowed("rl:test", now=1001.0)
    assert allowed is True

    allowed, retry_after = await limiter.is_allowed("rl:test", now=1002.0)
    assert allowed is False
    assert retry_after == 58

    allowed, _ = await limiter.is_allowed("rl:test", now=1061.0)
    assert allowed is True


@pytest.mark.unit
async def test_limiter_fail_open_on_redis_error(mock_redis: MagicMock) -> None:
    limiter = SlidingWindowRateLimiter(
        redis=mock_redis,
        limit=2,
        window_seconds=60.0,
    )
    mock_redis.pipeline.side_effect = ConnectionError("Redis unavailable")

    allowed, retry_after = await limiter.is_allowed("rl:test", now=1000.0)

    assert allowed is True
    assert retry_after is None
