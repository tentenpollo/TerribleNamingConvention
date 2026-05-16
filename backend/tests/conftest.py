from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
import uuid

from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio

from app.main import app
from app.models.document import Document, FileType
from app.models.project import Project
from app.models.team import Team
from app.models.user import Role, User


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_user() -> User:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.role = Role.MEMBER.value
    user.is_active = True
    user.created_at = datetime.now(UTC)
    return user


# ---------------------------------------------------------------------------
# Integration test fixtures — require real Postgres
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def db_session():
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings

    engine = create_async_engine(settings.database_url, echo=False)
    local_session = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with local_session() as session:
        async with session.begin():
            yield session


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def test_team(db_session) -> Team:
    team = Team(name="Test Team")
    db_session.add(team)
    await db_session.flush()
    return team


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def test_project(db_session, test_team: Team) -> Project:
    project = Project(
        name="Test Project",
        description="Integration test project",
        team_id=test_team.id,
        config={},
    )
    db_session.add(project)
    await db_session.flush()
    return project


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def test_document(db_session, test_project: Project) -> Document:
    doc = Document(
        project_id=test_project.id,
        filename="test.md",
        file_type=FileType.MARKDOWN.value,
        raw_content="# Test Document\n\nContent here.",
    )
    db_session.add(doc)
    await db_session.flush()
    return doc
