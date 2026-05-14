from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock
import uuid

from httpx import AsyncClient
import pytest
from pytest import FixtureRequest

from app.core.dependencies import get_async_session
from app.core.roles import Role
from app.core.security import create_access_token
from app.main import app
from app.models.user import User


@pytest.fixture
def member_user() -> User:
    return _make_user(Role.MEMBER)


@pytest.fixture
def admin_user() -> User:
    return _make_user(Role.ADMIN)


@pytest.fixture
def super_admin_user() -> User:
    return _make_user(Role.SUPER_ADMIN)


@pytest.fixture
def member_token(member_user: User) -> str:
    return create_access_token(member_user.id, member_user.role)


@pytest.fixture
def admin_token(admin_user: User) -> str:
    return create_access_token(admin_user.id, admin_user.role)


@pytest.fixture
def super_admin_token(super_admin_user: User) -> str:
    return create_access_token(super_admin_user.id, super_admin_user.role)


@pytest.fixture(autouse=True)
def override_session_users(
    member_user: User,
    admin_user: User,
    super_admin_user: User,
) -> Generator[None]:
    users_by_id = {
        member_user.id: member_user,
        admin_user.id: admin_user,
        super_admin_user.id: super_admin_user,
    }
    session = AsyncMock()
    session.get.side_effect = lambda _model, user_id: users_by_id.get(user_id)

    async def override_get_async_session() -> AsyncGenerator[AsyncMock, None]:
        yield session

    app.dependency_overrides[get_async_session] = override_get_async_session
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_fixture", "path", "expected_status"),
    [
        ("member_token", "/test/member-only", 200),
        ("member_token", "/test/admin-only", 403),
        ("member_token", "/test/superadmin-only", 403),
        ("admin_token", "/test/member-only", 200),
        ("admin_token", "/test/admin-only", 200),
        ("admin_token", "/test/superadmin-only", 403),
        ("super_admin_token", "/test/member-only", 200),
        ("super_admin_token", "/test/admin-only", 200),
        ("super_admin_token", "/test/superadmin-only", 200),
    ],
)
async def test_rbac_routes_enforce_role_hierarchy(
    async_client: AsyncClient,
    request: FixtureRequest,
    token_fixture: str,
    path: str,
    expected_status: int,
) -> None:
    token = request.getfixturevalue(token_fixture)

    response = await async_client.get(
        path,
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/test/member-only",
        "/test/admin-only",
        "/test/superadmin-only",
    ],
)
async def test_rbac_routes_return_401_without_token(
    async_client: AsyncClient,
    path: str,
) -> None:
    response = await async_client.get(path)

    assert response.status_code == 401


def _make_user(role: Role) -> User:
    return User(
        id=uuid.uuid4(),
        email=f"{role.value}@example.com",
        hashed_password="hashed",
        role=role.value,
        is_active=True,
        created_at=datetime.now(UTC),
    )
