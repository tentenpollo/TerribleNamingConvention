from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.dependencies import get_team_service, require_role
from app.core.exceptions import (
    AlreadyMemberError,
    MemberNotFoundError,
    TeamNotFoundError,
    UserNotFoundError,
)
from app.core.roles import Role
from app.models.user import User
from app.schemas.team import TeamCreate, TeamMemberAdd, TeamResponse
from app.services.team import TeamService

router = APIRouter(prefix="/teams")


@router.post(
    "",
    response_model=TeamResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_team(
    data: TeamCreate,
    _current_user: User = Depends(require_role(Role.ADMIN)),
    team_service: TeamService = Depends(get_team_service),
) -> TeamResponse:
    team = await team_service.create(data)
    return TeamResponse.model_validate(team)


@router.get("", response_model=list[TeamResponse])
async def list_teams(
    _current_user: User = Depends(require_role(Role.ADMIN)),
    team_service: TeamService = Depends(get_team_service),
) -> list[TeamResponse]:
    teams = await team_service.list_all()
    return [TeamResponse.model_validate(team) for team in teams]


@router.post(
    "/{team_id}/members",
    status_code=status.HTTP_201_CREATED,
)
async def add_team_member(
    team_id: UUID,
    data: TeamMemberAdd,
    _current_user: User = Depends(require_role(Role.ADMIN)),
    team_service: TeamService = Depends(get_team_service),
) -> Response:
    try:
        await team_service.add_member(team_id=team_id, user_id=data.user_id)
    except (TeamNotFoundError, UserNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.message) from exc
    except AlreadyMemberError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message) from exc

    return Response(status_code=status.HTTP_201_CREATED)


@router.delete(
    "/{team_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_team_member(
    team_id: UUID,
    user_id: UUID,
    _current_user: User = Depends(require_role(Role.ADMIN)),
    team_service: TeamService = Depends(get_team_service),
) -> Response:
    try:
        await team_service.remove_member(team_id=team_id, user_id=user_id)
    except (TeamNotFoundError, MemberNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.message) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
