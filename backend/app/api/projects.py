from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_accessible_projects, get_project_service, require_role
from app.core.exceptions import AccessDeniedError, ProjectNotFoundError, TeamNotFoundError
from app.core.roles import Role
from app.models.user import User
from app.schemas.project import ProjectCreate, ProjectResponse, ProjectUpdate
from app.services.project import ProjectService

router = APIRouter(prefix="/projects")


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project(
    data: ProjectCreate,
    _current_user: User = Depends(require_role(Role.ADMIN)),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    try:
        project = await project_service.create(data)
    except TeamNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.message) from exc
    return ProjectResponse.model_validate(project)


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    project_service: ProjectService = Depends(get_project_service),
) -> list[ProjectResponse]:
    projects = await project_service.list_for_user(accessible_ids)
    return [ProjectResponse.model_validate(project) for project in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    try:
        project = await project_service.get_by_id(project_id, accessible_ids)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.message) from exc
    except AccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message) from exc
    return ProjectResponse.model_validate(project)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    data: ProjectUpdate,
    _current_user: User = Depends(require_role(Role.ADMIN)),
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    try:
        project = await project_service.update(project_id, data, accessible_ids)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.message) from exc
    except AccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message) from exc
    return ProjectResponse.model_validate(project)
