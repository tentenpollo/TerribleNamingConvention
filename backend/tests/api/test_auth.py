from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
import uuid

from httpx import AsyncClient
import jwt
import pytest

from app.core.config import settings
from app.core.dependencies import get_async_session
from app.core.security import ALGORITHM, create_access_token
from app.main import app
from app.models.user import Role, User


class TestRegister:
    @pytest.mark.asyncio
    async def test_register_success(self, async_client: AsyncClient) -> None:
        mock_user = AsyncMock()
        mock_user.id = "123e4567-e89b-12d3-a456-426614174000"
        mock_user.email = "new@example.com"
        mock_user.role = Role.MEMBER.value
        mock_user.is_active = True
        mock_user.created_at = "2024-01-01T00:00:00+00:00"

        with patch(
            "app.api.auth.AuthService.register",
            new_callable=AsyncMock,
            return_value=mock_user,
        ):
            response = await async_client.post(
                "/auth/register",
                json={"email": "new@example.com", "password": "securepassword123"},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "new@example.com"
        assert data["role"] == Role.MEMBER.value
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, async_client: AsyncClient) -> None:
        from app.core.exceptions import DuplicateEmailError

        with patch(
            "app.api.auth.AuthService.register",
            new_callable=AsyncMock,
            side_effect=DuplicateEmailError("Email new@example.com is already registered"),
        ):
            response = await async_client.post(
                "/auth/register",
                json={"email": "new@example.com", "password": "securepassword123"},
            )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "securepassword123"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_too_short(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/auth/register",
            json={"email": "new@example.com", "password": "short"},
        )
        assert response.status_code == 422


class TestLogin:
    @pytest.mark.asyncio
    async def test_login_success(self, async_client: AsyncClient) -> None:
        with patch(
            "app.api.auth.AuthService.login",
            new_callable=AsyncMock,
            return_value="fake-jwt-token",
        ):
            response = await async_client.post(
                "/auth/login",
                json={"email": "user@example.com", "password": "securepassword123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "fake-jwt-token"
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, async_client: AsyncClient) -> None:
        from app.core.exceptions import InvalidCredentialsError

        with patch(
            "app.api.auth.AuthService.login",
            new_callable=AsyncMock,
            side_effect=InvalidCredentialsError("Invalid email or password"),
        ):
            response = await async_client.post(
                "/auth/login",
                json={"email": "user@example.com", "password": "wrongpassword"},
            )

        assert response.status_code == 401
        assert "Invalid" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_nonexistent_email(self, async_client: AsyncClient) -> None:
        from app.core.exceptions import InvalidCredentialsError

        with patch(
            "app.api.auth.AuthService.login",
            new_callable=AsyncMock,
            side_effect=InvalidCredentialsError("Invalid email or password"),
        ):
            response = await async_client.post(
                "/auth/login",
                json={"email": "nobody@example.com", "password": "securepassword123"},
            )

        assert response.status_code == 401
        assert "Invalid" in response.json()["detail"]


class TestGetMe:
    @pytest.mark.asyncio
    async def test_get_me_with_valid_token_returns_user_data(
        self,
        async_client: AsyncClient,
    ) -> None:
        user = _make_user()
        token = create_access_token(user.id, user.role)
        _override_session_user(user)

        response = await async_client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(user.id)
        assert data["email"] == user.email
        assert data["role"] == user.role
        assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_me_with_no_token_returns_401(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/auth/me")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_me_with_malformed_token_returns_401(
        self,
        async_client: AsyncClient,
    ) -> None:
        response = await async_client.get(
            "/auth/me",
            headers={"Authorization": "Bearer not-a-jwt"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_me_with_expired_token_returns_401(
        self,
        async_client: AsyncClient,
    ) -> None:
        token = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "role": Role.MEMBER.value,
                "exp": datetime.now(UTC) - timedelta(minutes=1),
            },
            settings.jwt_secret,
            algorithm=ALGORITHM,
        )

        response = await async_client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 401


class TestProtected:
    @pytest.mark.asyncio
    async def test_protected_with_valid_token_returns_user_context(
        self,
        async_client: AsyncClient,
    ) -> None:
        user = _make_user()
        token = create_access_token(user.id, user.role)
        _override_session_user(user)

        response = await async_client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json() == {"user_id": str(user.id), "role": user.role}

    @pytest.mark.asyncio
    async def test_protected_with_no_token_returns_401(
        self,
        async_client: AsyncClient,
    ) -> None:
        response = await async_client.get("/protected")
        assert response.status_code == 401


def _make_user() -> User:
    return User(
        id=uuid.uuid4(),
        email="user@example.com",
        hashed_password="hashed",
        role=Role.MEMBER.value,
        is_active=True,
        created_at=datetime.now(UTC),
    )


def _override_session_user(user: User | None) -> None:
    session = AsyncMock()
    session.get.return_value = user

    async def override_get_async_session():
        yield session

    app.dependency_overrides[get_async_session] = override_get_async_session
