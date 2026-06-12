from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import cast
import uuid

from arq.connections import ArqRedis
from fastapi import Depends, Header, HTTPException, Request, status
import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.access import get_accessible_project_ids
from app.core.database import AsyncSessionLocal
from app.core.qdrant import get_qdrant_client
from app.core.roles import Role, has_permission
from app.core.security import decode_access_token
from app.ingestion.embedder import Embedder
from app.ingestion.embedder import get_embedder as _get_embedder
from app.ingestion.vector_store import VectorStore
from app.models.user import User
from app.services.auth import AuthService
from app.services.document import DocumentService
from app.services.project import ProjectService
from app.services.team import TeamService


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def get_auth_service(
    session: AsyncSession = Depends(get_async_session),
) -> AuthService:
    return AuthService(session)


async def get_team_service(
    session: AsyncSession = Depends(get_async_session),
) -> TeamService:
    return TeamService(session)


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(client=get_qdrant_client())
    return _vector_store


async def get_project_service(
    session: AsyncSession = Depends(get_async_session),
    vector_store: VectorStore = Depends(get_vector_store),
) -> ProjectService:
    return ProjectService(session, vector_store)


async def get_current_user(
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_async_session),
) -> User:
    if authorization is None:
        raise _unauthorized()

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized()

    try:
        payload = decode_access_token(token)
        subject = payload.get("sub")
        if not isinstance(subject, str):
            raise ValueError("JWT subject is missing")
        user_id = uuid.UUID(subject)
    except (jwt.PyJWTError, ValueError):
        raise _unauthorized() from None

    user = await session.get(User, user_id)
    if user is None:
        raise _unauthorized()

    return user


async def get_accessible_projects(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[uuid.UUID]:
    return await get_accessible_project_ids(current_user, session)


def require_role(required: Role) -> Callable[..., Awaitable[User]]:
    async def dependency(current_user: User = Depends(get_current_user)) -> User:
        try:
            user_role = Role(current_user.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            ) from None

        if not has_permission(user_role, required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return current_user

    return dependency


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
    )


def get_embedder() -> Embedder:
    return _get_embedder()


def get_arq_pool(request: Request) -> ArqRedis:
    return cast(ArqRedis, request.app.state.arq_pool)


async def get_document_service(
    session: AsyncSession = Depends(get_async_session),
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> DocumentService:
    return DocumentService(session, arq_pool)
