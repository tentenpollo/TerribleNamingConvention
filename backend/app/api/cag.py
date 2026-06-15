from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.core.dependencies import get_accessible_projects, get_cag_service, require_role
from app.core.exceptions import AccessDeniedError, BeliefStateNotFoundError
from app.core.roles import Role
from app.models.user import User
from app.schemas.belief_state import (
    BeliefStateRecord,
    BeliefStateVersionMeta,
    RebuildRequest,
    RebuildResponse,
)
from app.services.cag import CAGService

router = APIRouter()


@router.get(
    "/projects/{project_id}/cag",
    response_model=BeliefStateRecord,
)
async def get_cag(
    project_id: UUID,
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    cag_service: CAGService = Depends(get_cag_service),
) -> BeliefStateRecord:
    if project_id not in accessible_ids:
        raise AccessDeniedError("Access denied")

    record = await cag_service.get_latest(project_id)
    if record is None:
        raise BeliefStateNotFoundError(f"No belief state exists for project {project_id}")
    return record


@router.get(
    "/projects/{project_id}/cag/versions",
    response_model=list[BeliefStateVersionMeta],
)
async def list_cag_versions(
    project_id: UUID,
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    cag_service: CAGService = Depends(get_cag_service),
) -> list[BeliefStateVersionMeta]:
    if project_id not in accessible_ids:
        raise AccessDeniedError("Access denied")

    return await cag_service.list_versions(project_id)


@router.post(
    "/projects/{project_id}/cag/rebuild",
    response_model=RebuildResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def rebuild_cag(
    project_id: UUID,
    body: RebuildRequest,
    request: Request,
    _current_user: User = Depends(require_role(Role.ADMIN)),
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
) -> RebuildResponse:
    if project_id not in accessible_ids:
        raise AccessDeniedError("Access denied")

    arq_pool = request.app.state.arq_pool
    result = await arq_pool.enqueue_job(
        "cag_rebuild",
        project_id,
        body.mode,
        _job_id=f"cag-rebuild-{project_id}",
    )
    return RebuildResponse(queued=True, deduplicated=(result is None))


async def belief_state_not_found_handler(
    request: Request,
    exc: BeliefStateNotFoundError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": exc.message},
    )
