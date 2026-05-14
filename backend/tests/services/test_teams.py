from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.core.exceptions import (
    AlreadyMemberError,
    MemberNotFoundError,
    TeamNotFoundError,
    UserNotFoundError,
)
from app.models.team import Team, TeamMember
from app.models.user import Role, User
from app.schemas.team import TeamCreate
from app.services.team import TeamService


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def team_service(mock_session: AsyncMock) -> TeamService:
    return TeamService(mock_session)


@pytest.mark.asyncio
async def test_create_team_persists_and_returns_team(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    team = await team_service.create(TeamCreate(name="Platform"))

    assert team.name == "Platform"
    mock_session.add.assert_called_once_with(team)
    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(team)


@pytest.mark.asyncio
async def test_list_all_returns_teams(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    teams = [_make_team("Alpha"), _make_team("Beta")]
    result = MagicMock()
    result.scalars.return_value.all.return_value = teams
    mock_session.execute.return_value = result

    assert await team_service.list_all() == teams


@pytest.mark.asyncio
async def test_add_member_success(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    team = _make_team("Platform")
    user = _make_user()

    mock_session.get.side_effect = [team, user, None]

    await team_service.add_member(team.id, user.id)

    added_member = mock_session.add.call_args.args[0]
    assert isinstance(added_member, TeamMember)
    assert added_member.team_id == team.id
    assert added_member.user_id == user.id
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_member_team_not_found_raises(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.return_value = None

    with pytest.raises(TeamNotFoundError):
        await team_service.add_member(uuid.uuid4(), uuid.uuid4())


@pytest.mark.asyncio
async def test_add_member_user_not_found_raises(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.side_effect = [_make_team("Platform"), None]

    with pytest.raises(UserNotFoundError):
        await team_service.add_member(uuid.uuid4(), uuid.uuid4())


@pytest.mark.asyncio
async def test_add_member_already_member_raises(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    team = _make_team("Platform")
    user = _make_user()
    mock_session.get.side_effect = [team, user, TeamMember(user_id=user.id, team_id=team.id)]

    with pytest.raises(AlreadyMemberError):
        await team_service.add_member(team.id, user.id)


@pytest.mark.asyncio
async def test_remove_member_success(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    team = _make_team("Platform")
    user_id = uuid.uuid4()
    member = TeamMember(user_id=user_id, team_id=team.id)
    mock_session.get.side_effect = [team, member]

    await team_service.remove_member(team.id, user_id)

    mock_session.delete.assert_awaited_once_with(member)
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_member_team_not_found_raises(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.return_value = None

    with pytest.raises(TeamNotFoundError):
        await team_service.remove_member(uuid.uuid4(), uuid.uuid4())


@pytest.mark.asyncio
async def test_remove_member_not_found_raises(
    team_service: TeamService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.side_effect = [_make_team("Platform"), None]

    with pytest.raises(MemberNotFoundError):
        await team_service.remove_member(uuid.uuid4(), uuid.uuid4())


def _make_team(name: str) -> Team:
    return Team(
        id=uuid.uuid4(),
        name=name,
        created_at=datetime.now(UTC),
    )


def _make_user() -> User:
    return User(
        id=uuid.uuid4(),
        email="member@example.com",
        hashed_password="hashed",
        role=Role.MEMBER.value,
        is_active=True,
        created_at=datetime.now(UTC),
    )
