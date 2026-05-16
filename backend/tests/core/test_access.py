from __future__ import annotations

from collections.abc import AsyncGenerator
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.access import get_accessible_project_ids
from app.core.database import AsyncSessionLocal, engine
from app.core.roles import Role
from app.models.project import Project
from app.models.team import Team, TeamMember
from app.models.user import User

pytestmark = pytest.mark.integration


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as db_session:
        # Truncate all tables in one statement — CASCADE handles FK ordering
        await db_session.execute(
            text(
                "TRUNCATE ingestion_jobs, document_summaries, documents, "
                "projects, team_members, teams, users "
                "RESTART IDENTITY CASCADE"
            )
        )
        await db_session.commit()
        yield db_session
        await db_session.rollback()
    await engine.dispose()


async def test_member_with_one_team_assigned_to_one_project_returns_only_that_project(
    session: AsyncSession,
) -> None:
    user = await _create_user(session, Role.MEMBER)
    team = await _create_team(session)
    project = await _create_project(session, team)
    other_team = await _create_team(session)
    await _create_project(session, other_team)
    await _create_team_member(session, user, team)

    accessible_project_ids = await get_accessible_project_ids(user, session)

    assert accessible_project_ids == [project.id]


async def test_member_with_two_teams_assigned_to_two_projects_returns_both_project_ids(
    session: AsyncSession,
) -> None:
    user = await _create_user(session, Role.MEMBER)
    first_team = await _create_team(session)
    second_team = await _create_team(session)
    first_project = await _create_project(session, first_team)
    second_project = await _create_project(session, second_team)
    await _create_team_member(session, user, first_team)
    await _create_team_member(session, user, second_team)

    accessible_project_ids = await get_accessible_project_ids(user, session)

    assert set(accessible_project_ids) == {first_project.id, second_project.id}


async def test_member_with_no_team_assignments_returns_empty_list(
    session: AsyncSession,
) -> None:
    user = await _create_user(session, Role.MEMBER)
    team = await _create_team(session)
    await _create_project(session, team)

    accessible_project_ids = await get_accessible_project_ids(user, session)

    assert accessible_project_ids == []


async def test_member_cannot_access_project_assigned_to_different_team(
    session: AsyncSession,
) -> None:
    user = await _create_user(session, Role.MEMBER)
    assigned_team = await _create_team(session)
    other_team = await _create_team(session)
    assigned_project = await _create_project(session, assigned_team)
    other_project = await _create_project(session, other_team)
    await _create_team_member(session, user, assigned_team)

    accessible_project_ids = await get_accessible_project_ids(user, session)

    assert assigned_project.id in accessible_project_ids
    assert other_project.id not in accessible_project_ids


async def test_admin_returns_all_project_ids_regardless_of_team_membership(
    session: AsyncSession,
) -> None:
    user = await _create_user(session, Role.ADMIN)
    first_team = await _create_team(session)
    second_team = await _create_team(session)
    first_project = await _create_project(session, first_team)
    second_project = await _create_project(session, second_team)

    accessible_project_ids = await get_accessible_project_ids(user, session)

    assert set(accessible_project_ids) == {first_project.id, second_project.id}


async def test_super_admin_returns_all_project_ids(session: AsyncSession) -> None:
    user = await _create_user(session, Role.SUPER_ADMIN)
    first_team = await _create_team(session)
    second_team = await _create_team(session)
    first_project = await _create_project(session, first_team)
    second_project = await _create_project(session, second_team)

    accessible_project_ids = await get_accessible_project_ids(user, session)

    assert set(accessible_project_ids) == {first_project.id, second_project.id}


async def _create_user(session: AsyncSession, role: Role) -> User:
    user = User(
        email=f"access-{uuid.uuid4()}@example.com",
        hashed_password="hashed",
        role=role.value,
    )
    session.add(user)
    await session.flush()
    return user


async def _create_team(session: AsyncSession) -> Team:
    team = Team(name=f"Team {uuid.uuid4()}")
    session.add(team)
    await session.flush()
    return team


async def _create_project(session: AsyncSession, team: Team) -> Project:
    project = Project(name=f"Project {uuid.uuid4()}", team_id=team.id)
    session.add(project)
    await session.flush()
    return project


async def _create_team_member(session: AsyncSession, user: User, team: Team) -> TeamMember:
    team_member = TeamMember(team_id=team.id, user_id=user.id)
    session.add(team_member)
    await session.flush()
    return team_member
