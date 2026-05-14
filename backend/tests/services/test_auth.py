from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from app.core.exceptions import (
    DuplicateEmailError,
    InvalidCredentialsError,
    UserNotFoundError,
)
from app.models.user import Role, User
from app.schemas.user import LoginRequest, UserCreate
from app.services.auth import AuthService


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def auth_service(mock_session: AsyncMock) -> AuthService:
    return AuthService(mock_session)


class TestRegister:
    @pytest.mark.asyncio
    async def test_register_creates_user_with_hashed_password(
        self,
        auth_service: AuthService,
        mock_session: AsyncMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        data = UserCreate(email="new@example.com", password="plainpassword")
        user = await auth_service.register(data)

        assert isinstance(user, User)
        assert user.email == "new@example.com"
        assert user.hashed_password != "plainpassword"
        assert user.hashed_password.startswith("$2")
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
        mock_session.refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_register_duplicate_email_raises(
        self,
        auth_service: AuthService,
        mock_session: AsyncMock,
    ) -> None:
        existing_user = User(
            id=uuid.uuid4(),
            email="existing@example.com",
            hashed_password="hashed",
            role=Role.MEMBER.value,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_user
        mock_session.execute.return_value = mock_result

        data = UserCreate(email="existing@example.com", password="plainpassword")
        with pytest.raises(DuplicateEmailError):
            await auth_service.register(data)


class TestLogin:
    @pytest.mark.asyncio
    async def test_login_returns_valid_jwt(
        self,
        auth_service: AuthService,
        mock_session: AsyncMock,
    ) -> None:
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            hashed_password="$2b$12$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            role=Role.MEMBER.value,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        with patch("app.services.auth.verify_password", return_value=True):
            with patch("app.services.auth.create_access_token", return_value="jwt-token"):
                token = await auth_service.login(
                    LoginRequest(email="user@example.com", password="correct")
                )

        assert token == "jwt-token"

    @pytest.mark.asyncio
    async def test_login_wrong_password_raises(
        self,
        auth_service: AuthService,
        mock_session: AsyncMock,
    ) -> None:
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            hashed_password="$2b$12$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            role=Role.MEMBER.value,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        with patch("app.services.auth.verify_password", return_value=False):
            with pytest.raises(InvalidCredentialsError):
                await auth_service.login(LoginRequest(email="user@example.com", password="wrong"))

    @pytest.mark.asyncio
    async def test_login_nonexistent_email_raises(
        self,
        auth_service: AuthService,
        mock_session: AsyncMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(InvalidCredentialsError):
            await auth_service.login(LoginRequest(email="nobody@example.com", password="password"))


class TestGetById:
    @pytest.mark.asyncio
    async def test_get_by_id_returns_user(
        self,
        auth_service: AuthService,
        mock_session: AsyncMock,
    ) -> None:
        user_id = uuid.uuid4()
        user = User(
            id=user_id,
            email="user@example.com",
            hashed_password="hashed",
            role=Role.MEMBER.value,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        mock_session.get.return_value = user

        result = await auth_service.get_by_id(user_id)
        assert result == user

    @pytest.mark.asyncio
    async def test_get_by_id_not_found_raises(
        self,
        auth_service: AuthService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.get.return_value = None

        with pytest.raises(UserNotFoundError):
            await auth_service.get_by_id(uuid.uuid4())
