from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DuplicateEmailError,
    InvalidCredentialsError,
    UserNotFoundError,
)
from app.core.logging import logger
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.user import LoginRequest, UserCreate


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register(self, data: UserCreate) -> User:
        stmt = select(User).where(User.email == data.email)
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            logger.warning("Duplicate registration attempt", email=data.email)
            raise DuplicateEmailError(f"Email {data.email} is already registered")

        user = User(
            email=data.email,
            hashed_password=hash_password(data.password),
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        logger.info("User registered", user_id=str(user.id), email=user.email)
        return user

    async def login(self, data: LoginRequest) -> str:
        stmt = select(User).where(User.email == data.email)
        result = await self.session.execute(stmt)
        user = result.scalar_one_or_none()
        if user is None:
            logger.warning("Login attempt for non-existent user", email=data.email)
            raise InvalidCredentialsError("Invalid email or password")

        if not verify_password(data.password, user.hashed_password):
            logger.warning("Login attempt with wrong password", email=data.email)
            raise InvalidCredentialsError("Invalid email or password")

        token = create_access_token(user.id, user.role)
        logger.info("User logged in", user_id=str(user.id))
        return token

    async def get_by_id(self, user_id: uuid.UUID) -> User:
        user = await self.session.get(User, user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")
        return user
