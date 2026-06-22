from __future__ import annotations

import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.core.config import settings
from app.core.dependencies import (
    get_accessible_projects,
    get_current_user,
    get_query_service,
    get_rate_limit_redis,
    require_role,
)
from app.core.exceptions import InvalidQueryError, QueryGenerationError
from app.core.ratelimit import SlidingWindowRateLimiter
from app.core.roles import Role
from app.models.user import User
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query import QueryService

router = APIRouter()


async def rate_limit_query(
    current_user: User = Depends(get_current_user),
    redis: Redis = Depends(get_rate_limit_redis),
) -> None:
    """Sliding-window rate limit for query endpoints.

    Rejects requests over the configured per-minute limit with 429 and a
    Retry-After header. Redis failures are logged and fail-open so a broken
    limiter does not take down querying.
    """
    limiter = SlidingWindowRateLimiter(
        redis=redis,
        limit=settings.query_rate_limit_per_minute,
        window_seconds=60.0,
    )
    allowed, retry_after = await limiter.is_allowed(
        key=f"rl:query:{current_user.id}",
        now=time.time(),
    )
    if not allowed:
        headers = {"Retry-After": str(int(retry_after))} if retry_after is not None else {}
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers=headers,
            detail="Query rate limit exceeded",
        )


@router.post(
    "/projects/{project_id}/query",
    response_model=QueryResponse,
)
async def query_project(
    project_id: UUID,
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit_query),
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    return await query_service.query(
        question=body.question,
        project_id=project_id,
        accessible_ids=accessible_ids,
        user_id=current_user.id,
        top_k=body.top_k,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
)
async def query_cross_project(
    body: QueryRequest,
    current_user: User = Depends(require_role(Role.ADMIN)),
    _rate_limit: None = Depends(rate_limit_query),
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    return await query_service.query_cross_project(
        question=body.question,
        accessible_ids=accessible_ids,
        user_id=current_user.id,
        top_k=body.top_k,
    )


async def invalid_query_handler(request: Request, exc: InvalidQueryError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.message},
    )


async def query_generation_error_handler(
    request: Request,
    exc: QueryGenerationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": exc.message, "retryable": True},
    )
