from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock
import uuid

from httpx import AsyncClient
import pytest

from app.core.dependencies import get_async_session, get_team_service
from app.core.exceptions import (
    AlreadyMemberError,
    MemberNotFoundError,
    TeamNotFoundError,
    UserNotFoundError,
)
from app.core.roles import Role
from app.core.security import create_access_token
from app.main import app
from app.models.team import Team
from app.models.user import User


@pytest.fixture
def member_user() -> User:
    return _make_user(Role.MEMBER)


@pytest.fixture
def admin_user() -> User:
    return _make_user(Role.ADMIN)


@pytest.fixture
def target_user() -> User:
    return _make_user(Role.MEMBER)


@pytest.fixture
def member_token(member_user: User) -> str:
    return create_access_token(member_user.id, member_user.role)


@pytest.fixture
def admin_token(admin_user: User) -> str:
    return create_access_token(admin_user.id, admin_user.role)


@pytest.fixture
def team_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture(autouse=True)
def override_dependencies(
    admin_user: User,
    member_user: User,
    target_user: User,
    team_service: AsyncMock,
) -> Generator[None]:
    users_by_id = {
        admin_user.id: admin_user,
        member_user.id: member_user,
        target_user.id: target_user,
    }
    session = AsyncMock()
    session.get.side_effect = lambda _model, user_id: users_by_id.get(user_id)

    async def override_get_async_session() -> AsyncGenerator[AsyncMock, None]:
        yield session

    async def override_get_team_service() -> AsyncMock:
        return team_service

    app.dependency_overrides[get_async_session] = override_get_async_session
    app.dependency_overrides[get_team_service] = override_get_team_service
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_team_success(
    async_client: AsyncClient,
    admin_token: str,
    team_service: AsyncMock,
) -> None:
    team_service.create.return_value = _make_team("Platform")

    response = await async_client.post(
        "/teams",
        json={"name": "Platform"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Platform"


@pytest.mark.asyncio
async def test_member_cannot_create_team(
    async_client: AsyncClient,
    member_token: str,
) -> None:
    response = await async_client.post(
        "/teams",
        json={"name": "Platform"},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_cannot_create_team(async_client: AsyncClient) -> None:
    response = await async_client.post("/teams", json={"name": "Platform"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_lists_all_teams(
    async_client: AsyncClient,
    admin_token: str,
    team_service: AsyncMock,
) -> None:
    team_service.list_all.return_value = [_make_team("Alpha"), _make_team("Beta")]

    response = await async_client.get(
        "/teams",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
    assert [team["name"] for team in response.json()] == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_member_cannot_list_teams(
    async_client: AsyncClient,
    member_token: str,
) -> None:
    response = await async_client.get(
        "/teams",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_add_member_success(
    async_client: AsyncClient,
    admin_token: str,
    target_user: User,
) -> None:
    response = await async_client.post(
        f"/teams/{uuid.uuid4()}/members",
        json={"user_id": str(target_user.id)},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 201


@pytest.mark.asyncio
async def test_add_member_team_not_found(
    async_client: AsyncClient,
    admin_token: str,
    target_user: User,
    team_service: AsyncMock,
) -> None:
    team_service.add_member.side_effect = TeamNotFoundError("Team missing")

    response = await async_client.post(
        f"/teams/{uuid.uuid4()}/members",
        json={"user_id": str(target_user.id)},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_add_member_user_not_found(
    async_client: AsyncClient,
    admin_token: str,
    target_user: User,
    team_service: AsyncMock,
) -> None:
    team_service.add_member.side_effect = UserNotFoundError("User missing")

    response = await async_client.post(
        f"/teams/{uuid.uuid4()}/members",
        json={"user_id": str(target_user.id)},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_add_member_already_member(
    async_client: AsyncClient,
    admin_token: str,
    target_user: User,
    team_service: AsyncMock,
) -> None:
    team_service.add_member.side_effect = AlreadyMemberError("Already a member")

    response = await async_client.post(
        f"/teams/{uuid.uuid4()}/members",
        json={"user_id": str(target_user.id)},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_remove_member_success(
    async_client: AsyncClient,
    admin_token: str,
) -> None:
    response = await async_client.delete(
        f"/teams/{uuid.uuid4()}/members/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 204


@pytest.mark.asyncio
async def test_remove_member_not_a_member(
    async_client: AsyncClient,
    admin_token: str,
    team_service: AsyncMock,
) -> None:
    team_service.remove_member.side_effect = MemberNotFoundError("Not a member")

    response = await async_client.delete(
        f"/teams/{uuid.uuid4()}/members/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 404


def _make_team(name: str) -> Team:
    return Team(
        id=uuid.uuid4(),
        name=name,
        created_at=datetime.now(UTC),
    )


def _make_user(role: Role) -> User:
    user_id = uuid.uuid4()
    return User(
        id=user_id,
        email=f"{role.value}-{user_id}@example.com",
        hashed_password="hashed",
        role=role.value,
        is_active=True,
        created_at=datetime.now(UTC),
    )
