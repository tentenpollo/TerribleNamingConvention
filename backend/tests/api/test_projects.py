from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock
import uuid

from httpx import AsyncClient
import pytest

from app.core.dependencies import get_accessible_projects, get_async_session, get_project_service
from app.core.exceptions import AccessDeniedError, ProjectNotFoundError, TeamNotFoundError
from app.core.roles import Role
from app.core.security import create_access_token
from app.main import app
from app.models.project import Project
from app.models.user import User


@pytest.fixture
def member_user() -> User:
    return _make_user(Role.MEMBER)


@pytest.fixture
def admin_user() -> User:
    return _make_user(Role.ADMIN)


@pytest.fixture
def member_token(member_user: User) -> str:
    return create_access_token(member_user.id, member_user.role)


@pytest.fixture
def admin_token(admin_user: User) -> str:
    return create_access_token(admin_user.id, admin_user.role)


@pytest.fixture
def project_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def accessible_ids() -> list[uuid.UUID]:
    return []


@pytest.fixture(autouse=True)
def override_dependencies(
    admin_user: User,
    member_user: User,
    project_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> Generator[None]:
    users_by_id = {
        admin_user.id: admin_user,
        member_user.id: member_user,
    }
    session = AsyncMock()
    session.get.side_effect = lambda _model, user_id: users_by_id.get(user_id)

    async def override_get_async_session() -> AsyncGenerator[AsyncMock, None]:
        yield session

    async def override_get_project_service() -> AsyncMock:
        return project_service

    async def override_get_accessible_projects() -> list[uuid.UUID]:
        return accessible_ids

    app.dependency_overrides[get_async_session] = override_get_async_session
    app.dependency_overrides[get_project_service] = override_get_project_service
    app.dependency_overrides[get_accessible_projects] = override_get_accessible_projects
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_project_admin_success(
    async_client: AsyncClient,
    admin_token: str,
    project_service: AsyncMock,
) -> None:
    team_id = uuid.uuid4()
    project_service.create.return_value = _make_project("Knowledge Base", team_id)

    response = await async_client.post(
        "/projects",
        json={"name": "Knowledge Base", "team_id": str(team_id), "config": {"chunk_size": 800}},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Knowledge Base"
    assert body["team_id"] == str(team_id)


@pytest.mark.asyncio
async def test_create_project_member_blocked(
    async_client: AsyncClient,
    member_token: str,
) -> None:
    response = await async_client.post(
        "/projects",
        json={"name": "Knowledge Base", "team_id": str(uuid.uuid4())},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_project_invalid_team_id_returns_404(
    async_client: AsyncClient,
    admin_token: str,
    project_service: AsyncMock,
) -> None:
    project_service.create.side_effect = TeamNotFoundError("Team missing")

    response = await async_client.post(
        "/projects",
        json={"name": "Knowledge Base", "team_id": str(uuid.uuid4())},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_member_lists_only_their_team_projects(
    async_client: AsyncClient,
    member_token: str,
    project_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project = _make_project("Member Project", uuid.uuid4())
    accessible_ids.append(project.id)
    project_service.list_for_user.return_value = [project]

    response = await async_client.get(
        "/projects",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 200
    assert [item["name"] for item in response.json()] == ["Member Project"]
    project_service.list_for_user.assert_awaited_once_with([project.id])


@pytest.mark.asyncio
async def test_admin_lists_all_projects(
    async_client: AsyncClient,
    admin_token: str,
    project_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    projects = [_make_project("Alpha", uuid.uuid4()), _make_project("Beta", uuid.uuid4())]
    accessible_ids.extend(project.id for project in projects)
    project_service.list_for_user.return_value = projects

    response = await async_client.get(
        "/projects",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
    assert [item["name"] for item in response.json()] == ["Alpha", "Beta"]
    project_service.list_for_user.assert_awaited_once_with([project.id for project in projects])


@pytest.mark.asyncio
async def test_member_can_get_their_project(
    async_client: AsyncClient,
    member_token: str,
    project_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project = _make_project("Member Project", uuid.uuid4())
    accessible_ids.append(project.id)
    project_service.get_by_id.return_value = project

    response = await async_client.get(
        f"/projects/{project.id}",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(project.id)
    project_service.get_by_id.assert_awaited_once_with(project.id, [project.id])


@pytest.mark.asyncio
async def test_member_blocked_from_other_team_project(
    async_client: AsyncClient,
    member_token: str,
    project_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    project_service.get_by_id.side_effect = AccessDeniedError("Denied")

    response = await async_client.get(
        f"/projects/{project_id}",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_project_not_found(
    async_client: AsyncClient,
    member_token: str,
    project_service: AsyncMock,
) -> None:
    project_service.get_by_id.side_effect = ProjectNotFoundError("Missing")

    response = await async_client.get(
        f"/projects/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_project_admin_success(
    async_client: AsyncClient,
    admin_token: str,
    project_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project = _make_project("New", uuid.uuid4(), description="Updated")
    accessible_ids.append(project.id)
    project_service.update.return_value = project

    response = await async_client.patch(
        f"/projects/{project.id}",
        json={"name": "New", "description": "Updated"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
    assert response.json()["description"] == "Updated"
    project_service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_project_member_blocked(
    async_client: AsyncClient,
    member_token: str,
) -> None:
    response = await async_client.patch(
        f"/projects/{uuid.uuid4()}",
        json={"name": "New"},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 403


def _make_project(name: str, team_id: uuid.UUID, description: str | None = None) -> Project:
    return Project(
        id=uuid.uuid4(),
        name=name,
        description=description,
        team_id=team_id,
        config={},
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
