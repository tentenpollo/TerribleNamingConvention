from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
import uuid

from httpx import AsyncClient
import pytest

from app.core.dependencies import (
    get_accessible_projects,
    get_async_session,
    get_cag_service,
)
from app.core.roles import Role
from app.core.security import create_access_token
from app.main import app
from app.models.user import User
from app.schemas.belief_state import BeliefStateRecord


def _make_user(role: Role) -> User:
    return User(
        id=uuid.uuid4(),
        email=f"{role.value}@example.com",
        hashed_password="secret",
        role=role.value,
        is_active=True,
        created_at=datetime.now(UTC),
    )


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
def cag_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def accessible_ids() -> list[uuid.UUID]:
    return []


@pytest.fixture(autouse=True)
def override_dependencies(
    admin_user: User,
    member_user: User,
    cag_service: AsyncMock,
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

    async def override_get_cag_service() -> AsyncMock:
        return cag_service

    async def override_get_accessible_projects() -> list[uuid.UUID]:
        return accessible_ids

    app.dependency_overrides[get_async_session] = override_get_async_session
    app.dependency_overrides[get_cag_service] = override_get_cag_service
    app.dependency_overrides[get_accessible_projects] = override_get_accessible_projects
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def arq_pool() -> AsyncMock:
    original = getattr(app.state, "arq_pool", None)
    pool = AsyncMock()
    app.state.arq_pool = pool
    yield pool
    app.state.arq_pool = original


@pytest.mark.asyncio
async def test_get_cag_member_of_owning_team_returns_200(
    async_client: AsyncClient,
    member_token: str,
    cag_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)

    record = MagicMock()
    record.id = uuid.uuid4()
    record.project_id = project_id
    record.version = 2
    record.rebuild_type = "full"
    record.last_summary_created_at = datetime.now(UTC)
    record.summary_count_covered = 120
    record.created_at = datetime.now(UTC)
    record.state = {
        "project_summary": "Project summary.",
        "decisions": [],
        "open_items": [],
        "key_people": [],
        "recurring_themes": [],
    }
    cag_service.get_latest.return_value = BeliefStateRecord.model_validate(record)

    response = await async_client.get(
        f"/projects/{project_id}/cag",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 200
    assert response.json()["project_id"] == str(project_id)


@pytest.mark.asyncio
async def test_get_cag_member_of_other_team_returns_403(
    async_client: AsyncClient,
    member_token: str,
) -> None:
    project_id = uuid.uuid4()

    response = await async_client.get(
        f"/projects/{project_id}/cag",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_cag_no_state_returns_404(
    async_client: AsyncClient,
    member_token: str,
    cag_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)
    cag_service.get_latest.return_value = None

    response = await async_client.get(
        f"/projects/{project_id}/cag",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_cag_versions_member_of_owning_team_returns_200(
    async_client: AsyncClient,
    member_token: str,
    cag_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)
    cag_service.list_versions.return_value = []

    response = await async_client.get(
        f"/projects/{project_id}/cag/versions",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_post_rebuild_member_returns_403(
    async_client: AsyncClient,
    member_token: str,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)

    response = await async_client.post(
        f"/projects/{project_id}/cag/rebuild",
        json={"mode": "genesis"},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_post_rebuild_admin_returns_202(
    async_client: AsyncClient,
    admin_token: str,
    arq_pool: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)
    arq_pool.enqueue_job.return_value = MagicMock()

    response = await async_client.post(
        f"/projects/{project_id}/cag/rebuild",
        json={"mode": "genesis"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 202
    assert response.json() == {"queued": True, "deduplicated": False}
    arq_pool.enqueue_job.assert_awaited_once()
    call_args = arq_pool.enqueue_job.call_args
    assert call_args.args[0] == "cag_rebuild"
    assert call_args.args[1] == project_id
    assert call_args.args[2] == "genesis"
    assert call_args.kwargs.get("_job_id") == f"cag-rebuild-{project_id}"


@pytest.mark.asyncio
async def test_post_rebuild_admin_deduplicated_returns_202(
    async_client: AsyncClient,
    admin_token: str,
    arq_pool: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)
    arq_pool.enqueue_job.return_value = None

    response = await async_client.post(
        f"/projects/{project_id}/cag/rebuild",
        json={"mode": "compaction"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 202
    assert response.json() == {"queued": True, "deduplicated": True}
