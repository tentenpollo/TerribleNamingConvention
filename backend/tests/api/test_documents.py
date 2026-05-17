from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock
import uuid

from fastapi import HTTPException, status
from httpx import ASGITransport, AsyncClient
import pytest

from app.core.dependencies import (
    get_accessible_projects,
    get_arq_pool,
    get_async_session,
    get_current_user,
    get_document_service,
)
from app.core.exceptions import (
    AccessDeniedError,
)
from app.core.roles import Role
from app.core.security import create_access_token
from app.main import app
from app.models.document import FileType
from app.models.project import Project
from app.models.team import Team, TeamMember
from app.models.user import User
from app.schemas.document import DocumentResponse, IngestionJobResponse


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
def mock_arq_pool() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def document_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def accessible_ids() -> list[uuid.UUID]:
    return []


@pytest.fixture(autouse=True)
def override_dependencies(
    request: pytest.FixtureRequest,
    admin_user: User,
    member_user: User,
    mock_arq_pool: AsyncMock,
    document_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> Generator[None, None, None]:
    if request.node.get_closest_marker("integration"):
        yield
        return

    users_by_id = {
        admin_user.id: admin_user,
        member_user.id: member_user,
    }
    session = AsyncMock()
    session.get.side_effect = lambda _model, user_id: users_by_id.get(user_id)

    async def override_get_async_session() -> AsyncGenerator[AsyncMock, None]:
        yield session

    async def override_get_arq_pool() -> AsyncMock:
        return mock_arq_pool

    async def override_get_document_service() -> AsyncMock:
        return document_service

    async def override_get_accessible_projects() -> list[uuid.UUID]:
        return accessible_ids

    def override_get_current_user() -> User:
        return member_user

    app.dependency_overrides[get_async_session] = override_get_async_session
    app.dependency_overrides[get_arq_pool] = override_get_arq_pool
    app.dependency_overrides[get_document_service] = override_get_document_service
    app.dependency_overrides[get_accessible_projects] = override_get_accessible_projects
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_document_success(
    async_client: AsyncClient,
    member_token: str,
    document_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)
    job_response = _make_job_response(project_id)
    document_service.upload.return_value = job_response

    response = await async_client.post(
        f"/projects/{project_id}/documents",
        files={"file": ("test.md", b"# Hello", "text/markdown")},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["id"] == str(job_response.id)
    assert body["status"] == "pending"
    document_service.upload.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_document_member_to_own_project(
    async_client: AsyncClient,
    member_token: str,
    document_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)
    document_service.upload.return_value = _make_job_response(project_id)

    response = await async_client.post(
        f"/projects/{project_id}/documents",
        files={"file": ("test.md", b"content", "text/markdown")},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 202


@pytest.mark.asyncio
async def test_upload_document_member_to_other_project(
    async_client: AsyncClient,
    member_token: str,
    document_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    document_service.upload.side_effect = AccessDeniedError("Denied")

    response = await async_client.post(
        f"/projects/{project_id}/documents",
        files={"file": ("test.md", b"content", "text/markdown")},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_upload_document_unsupported_file_type(
    async_client: AsyncClient,
    member_token: str,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)

    response = await async_client.post(
        f"/projects/{project_id}/documents",
        files={"file": ("test.exe", b"binary", "application/octet-stream")},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_document_exceeds_max_size(
    async_client: AsyncClient,
    member_token: str,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)
    large_content = b"x" * (51 * 1024 * 1024)

    response = await async_client.post(
        f"/projects/{project_id}/documents",
        files={"file": ("test.md", large_content, "text/markdown")},
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_document_unauthenticated(
    async_client: AsyncClient,
) -> None:
    async def raise_unauthorized() -> list[uuid.UUID]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    app.dependency_overrides[get_accessible_projects] = raise_unauthorized

    response = await async_client.post(
        f"/projects/{uuid.uuid4()}/documents",
        files={"file": ("test.md", b"content", "text/markdown")},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_documents_returns_list(
    async_client: AsyncClient,
    member_token: str,
    document_service: AsyncMock,
    accessible_ids: list[uuid.UUID],
) -> None:
    project_id = uuid.uuid4()
    accessible_ids.append(project_id)
    docs = [
        _make_document_response(project_id, "a.md"),
        _make_document_response(project_id, "b.txt"),
    ]
    document_service.list_documents.return_value = docs

    response = await async_client.get(
        f"/projects/{project_id}/documents",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["filename"] == "a.md"
    document_service.list_documents.assert_awaited_once_with(
        project_id=project_id,
        accessible_ids=accessible_ids,
    )


@pytest.mark.asyncio
async def test_get_job_returns_status(
    async_client: AsyncClient,
    member_token: str,
    document_service: AsyncMock,
) -> None:
    job_id = uuid.uuid4()
    job_response = _make_job_response(uuid.uuid4(), job_id=job_id, status="complete")
    document_service.get_job.return_value = job_response

    response = await async_client.get(
        f"/jobs/{job_id}",
        headers={"Authorization": f"Bearer {member_token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(job_id)
    assert body["status"] == "complete"
    document_service.get_job.assert_awaited_once_with(job_id=job_id)


def _make_job_response(
    project_id: uuid.UUID,
    job_id: uuid.UUID | None = None,
    status: str = "pending",
) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=job_id or uuid.uuid4(),
        project_id=project_id,
        document_id=uuid.uuid4(),
        status=status,
        error_message=None,
        created_at=datetime.now(UTC),
        completed_at=None,
    )


def _make_document_response(project_id: uuid.UUID, filename: str) -> DocumentResponse:
    return DocumentResponse(
        id=uuid.uuid4(),
        project_id=project_id,
        filename=filename,
        file_type=FileType.MARKDOWN.value,
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


# ---------------------------------------------------------------------------
# Integration tests — require real Postgres, Redis
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_upload_document_creates_job_and_pollable() -> None:
    from arq.connections import ArqRedis, RedisSettings, create_pool
    from sqlalchemy import delete

    from app.core.database import AsyncSessionLocal
    from app.core.security import create_access_token

    arq_pool: ArqRedis | None = None
    team_id: uuid.UUID | None = None

    try:
        arq_pool = await create_pool(RedisSettings.from_dsn("redis://localhost:6379/0"))
        app.state.arq_pool = arq_pool

        async with AsyncSessionLocal() as session:
            user = User(
                email="doc-integration@example.com",
                hashed_password="hashed",
                role=Role.MEMBER.value,
                is_active=True,
            )
            session.add(user)
            await session.flush()

            team = Team(name="Doc Integration Team")
            session.add(team)
            await session.flush()

            membership = TeamMember(team_id=team.id, user_id=user.id)
            session.add(membership)

            project = Project(
                name="Doc Integration Project",
                team_id=team.id,
                config={},
            )
            session.add(project)
            await session.commit()

            user_id = user.id
            project_id = project.id
            team_id = team.id

        token = create_access_token(user_id, Role.MEMBER.value)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/projects/{project_id}/documents",
                files={
                    "file": (
                        "integration.md",
                        b"# Integration\n\nTest content.",
                        "text/markdown",
                    ),
                },
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 202
        body = response.json()
        job_id = body["id"]
        assert body["status"] == "pending"
        assert body["project_id"] == str(project_id)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            job_response = await client.get(
                f"/jobs/{job_id}",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert job_response.status_code == 200
        job_body = job_response.json()
        assert job_body["id"] == job_id
        assert job_body["status"] == "pending"

        async with AsyncSessionLocal() as session:
            from sqlalchemy import select as sa_select

            from app.models.document import Document as DocModel
            from app.models.ingestion_job import IngestionJob as JobModel

            doc_result = await session.execute(
                sa_select(DocModel).where(DocModel.project_id == project_id),
            )
            doc = doc_result.scalar_one()
            assert doc.filename == "integration.md"
            assert doc.raw_content == "# Integration\n\nTest content."

            job_result = await session.execute(
                sa_select(JobModel).where(JobModel.id == uuid.UUID(job_id)),
            )
            job = job_result.scalar_one()
            assert job.status == "pending"
            assert job.document_id == doc.id

    finally:
        if arq_pool is not None:
            await arq_pool.close()
            app.state.arq_pool = None

        if team_id is not None:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    delete(TeamMember).where(TeamMember.team_id == team_id),
                )
                await session.execute(delete(Team).where(Team.id == team_id))
                await session.execute(
                    delete(User).where(User.email == "doc-integration@example.com"),
                )
                await session.commit()
