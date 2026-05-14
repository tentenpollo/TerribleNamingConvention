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
def project_service(mock_session: AsyncMock) -> ProjectService:
    return ProjectService(mock_session)


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
