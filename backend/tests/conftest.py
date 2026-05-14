from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
import uuid

from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app
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
