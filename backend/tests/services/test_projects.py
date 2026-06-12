from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.core.exceptions import AccessDeniedError, ProjectNotFoundError, TeamNotFoundError
from app.models.project import Project
from app.models.team import Team
from app.schemas.project import ProjectCreate, ProjectUpdate
from app.services.project import ProjectService


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def project_service(mock_session: AsyncMock, mock_vector_store: AsyncMock) -> ProjectService:
    return ProjectService(mock_session, mock_vector_store)


@pytest.mark.asyncio
async def test_create_project_persists_and_returns_project(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    team = _make_team("Platform")
    mock_session.get.return_value = team

    project = await project_service.create(
        ProjectCreate(name="Knowledge Base", description="Docs", team_id=team.id, config={}),
    )

    assert project.name == "Knowledge Base"
    assert project.description == "Docs"
    assert project.team_id == team.id
    mock_session.add.assert_called_once_with(project)
    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(project)


@pytest.mark.asyncio
async def test_create_project_team_not_found_raises(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.return_value = None

    with pytest.raises(TeamNotFoundError):
        await project_service.create(ProjectCreate(name="Knowledge Base", team_id=uuid.uuid4()))


@pytest.mark.asyncio
async def test_get_by_id_returns_accessible_project(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    project = _make_project("Knowledge Base")
    mock_session.get.return_value = project

    assert await project_service.get_by_id(project.id, [project.id]) == project


@pytest.mark.asyncio
async def test_get_by_id_raises_access_denied_when_out_of_scope(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    project = _make_project("Knowledge Base")
    mock_session.get.return_value = project

    with pytest.raises(AccessDeniedError):
        await project_service.get_by_id(project.id, [])


@pytest.mark.asyncio
async def test_get_by_id_not_found_raises(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.return_value = None

    with pytest.raises(ProjectNotFoundError):
        await project_service.get_by_id(uuid.uuid4(), [])


@pytest.mark.asyncio
async def test_list_for_user_returns_only_accessible_projects(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    projects = [_make_project("Alpha"), _make_project("Beta")]
    result = MagicMock()
    result.scalars.return_value.all.return_value = projects
    mock_session.execute.return_value = result

    assert await project_service.list_for_user([project.id for project in projects]) == projects


@pytest.mark.asyncio
async def test_list_for_user_empty_scope_returns_empty_list(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    assert await project_service.list_for_user([]) == []
    mock_session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_project_success(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    project = _make_project("Old")
    mock_session.get.return_value = project

    updated = await project_service.update(
        project.id,
        ProjectUpdate(name="New", description="Updated"),
        [project.id],
    )

    assert updated.name == "New"
    assert updated.description == "Updated"
    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(project)


@pytest.mark.asyncio
async def test_update_project_out_of_scope_raises(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    project = _make_project("Knowledge Base")
    mock_session.get.return_value = project

    with pytest.raises(AccessDeniedError):
        await project_service.update(project.id, ProjectUpdate(name="New"), [])


@pytest.mark.asyncio
async def test_delete_project_commits_and_deletes_collection(
    project_service: ProjectService,
    mock_session: AsyncMock,
    mock_vector_store: AsyncMock,
) -> None:
    project = _make_project("To Delete")
    mock_session.get.return_value = project

    await project_service.delete(project.id, [project.id])

    mock_session.delete.assert_awaited_once_with(project)
    mock_session.commit.assert_awaited_once()
    mock_vector_store.delete_collection.assert_awaited_once_with(project.id)


@pytest.mark.asyncio
async def test_delete_project_qdrant_failure_does_not_raise(
    project_service: ProjectService,
    mock_session: AsyncMock,
    mock_vector_store: AsyncMock,
) -> None:
    from app.core.exceptions import QdrantError

    project = _make_project("To Delete")
    mock_session.get.return_value = project
    mock_vector_store.delete_collection.side_effect = QdrantError("connection lost")

    # Must not raise — Qdrant failure is logged but the project row is already gone
    await project_service.delete(project.id, [project.id])

    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_project_not_found_raises(
    project_service: ProjectService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.return_value = None

    with pytest.raises(ProjectNotFoundError):
        await project_service.delete(uuid.uuid4(), [])


# ---------------------------------------------------------------------------
# Integration tests — require real Postgres + Qdrant
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_delete_project_removes_qdrant_collection() -> None:
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy import delete as sa_delete

    from app.core import database
    from app.core.config import settings
    from app.ingestion.vector_store import VectorStore
    from app.models.project import Project
    from app.models.team import Team
    from app.services.project import ProjectService

    vector_store = VectorStore(client=AsyncQdrantClient(url=settings.qdrant_url))
    project_id = None

    try:
        async with database.AsyncSessionLocal() as session:
            team = Team(name="Delete Integration Team")
            session.add(team)
            await session.flush()

            project = Project(
                name="Delete Integration Project",
                team_id=team.id,
                config={},
            )
            session.add(project)
            await session.flush()
            project_id = project.id
            await session.commit()

        await vector_store.ensure_collection(project_id)
        assert await vector_store._client.collection_exists(f"project_{project_id}")

        async with database.AsyncSessionLocal() as session:
            svc = ProjectService(session, vector_store)
            await svc.delete(project_id, [project_id])

        assert not await vector_store._client.collection_exists(f"project_{project_id}")

    finally:
        if project_id is not None:
            try:
                await vector_store.delete_collection(project_id)
            except Exception:
                pass
            async with database.AsyncSessionLocal() as session:
                await session.execute(sa_delete(Project).where(Project.id == project_id))
                await session.commit()


def _make_team(name: str) -> Team:
    return Team(id=uuid.uuid4(), name=name, created_at=datetime.now(UTC))


def _make_project(name: str) -> Project:
    return Project(
        id=uuid.uuid4(),
        name=name,
        description=None,
        team_id=uuid.uuid4(),
        config={},
        created_at=datetime.now(UTC),
    )
