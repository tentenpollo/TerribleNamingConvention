from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.roles import Role
from app.models.project import Project
from app.models.team import TeamMember
from app.models.user import User


async def get_accessible_project_ids(user: User, session: AsyncSession) -> list[UUID]:
    try:
        role = Role(user.role)
    except ValueError:
        return []

    if role in {Role.ADMIN, Role.SUPER_ADMIN}:
        result = await session.execute(select(Project.id))
        return list(result.scalars().all())

    team_ids_result = await session.execute(
        select(TeamMember.team_id).where(TeamMember.user_id == user.id),
    )
    team_ids = list(team_ids_result.scalars().all())
    if not team_ids:
        return []

    project_ids_result = await session.execute(
        select(Project.id).where(Project.team_id.in_(team_ids)),
    )
    return list(project_ids_result.scalars().all())
