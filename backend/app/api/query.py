from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.core.dependencies import (
    get_accessible_projects,
    get_query_service,
    require_role,
)
from app.core.exceptions import InvalidQueryError
from app.core.roles import Role
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query import QueryService

router = APIRouter()


@router.post(
    "/projects/{project_id}/query",
    response_model=QueryResponse,
)
async def query_project(
    project_id: UUID,
    body: QueryRequest,
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    return await query_service.query(
        question=body.question,
        project_id=project_id,
        accessible_ids=accessible_ids,
        top_k=body.top_k,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
)
async def query_cross_project(
    body: QueryRequest,
    _current_user: object = Depends(require_role(Role.ADMIN)),
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    return await query_service.query_cross_project(
        question=body.question,
        accessible_ids=accessible_ids,
        top_k=body.top_k,
    )


async def invalid_query_handler(request: Request, exc: InvalidQueryError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.message},
    )
