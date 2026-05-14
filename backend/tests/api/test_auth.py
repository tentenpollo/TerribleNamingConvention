from __future__ import annotations

from unittest.mock import AsyncMock, patch

from httpx import AsyncClient
import pytest

from app.models.user import Role


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
    async def test_get_me_stub(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/auth/me")
        assert response.status_code == 200
        assert response.json()["message"] == "not implemented yet"
