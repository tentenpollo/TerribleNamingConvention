from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AlreadyMemberError,
    MemberNotFoundError,
    TeamNotFoundError,
    UserNotFoundError,
)
from app.models.team import Team, TeamMember
from app.models.user import User
from app.schemas.team import TeamCreate


class TeamService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: TeamCreate) -> Team:
        team = Team(name=data.name)
        self.session.add(team)
        await self.session.commit()
        await self.session.refresh(team)
        return team

    async def list_all(self) -> list[Team]:
        result = await self.session.execute(select(Team).order_by(Team.created_at, Team.name))
        return list(result.scalars().all())

    async def add_member(self, team_id: UUID, user_id: UUID) -> None:
        await self._require_team(team_id)
        await self._require_user(user_id)

        existing = await self.session.get(
            TeamMember,
            {"user_id": user_id, "team_id": team_id},
        )
        if existing is not None:
            raise AlreadyMemberError(f"User {user_id} is already a member of team {team_id}")

        self.session.add(TeamMember(user_id=user_id, team_id=team_id))
        await self.session.commit()

    async def remove_member(self, team_id: UUID, user_id: UUID) -> None:
        await self._require_team(team_id)

        member = await self.session.get(
            TeamMember,
            {"user_id": user_id, "team_id": team_id},
        )
        if member is None:
            raise MemberNotFoundError(f"User {user_id} is not a member of team {team_id}")

        await self.session.delete(member)
        await self.session.commit()

    async def _require_team(self, team_id: UUID) -> Team:
        team = await self.session.get(Team, team_id)
        if team is None:
            raise TeamNotFoundError(f"Team {team_id} does not exist")
        return team

    async def _require_user(self, user_id: UUID) -> User:
        user = await self.session.get(User, user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} does not exist")
        return user
