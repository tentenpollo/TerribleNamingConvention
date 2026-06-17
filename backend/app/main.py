from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from arq.connections import RedisSettings, create_pool
from fastapi import FastAPI

from app.api import auth, cag, documents, health, projects, query, teams
from app.api.auth import (
    duplicate_email_handler,
    invalid_credentials_handler,
)
from app.api.cag import belief_state_not_found_handler
from app.api.documents import (
    access_denied_handler,
    document_not_found_handler,
    ingestion_job_not_found_handler,
    unsupported_file_type_handler,
)
from app.api.query import invalid_query_handler
from app.core.config import settings
from app.core.exceptions import (
    AccessDeniedError,
    BeliefStateNotFoundError,
    DocumentNotFoundError,
    DuplicateEmailError,
    IngestionJobNotFoundError,
    InvalidCredentialsError,
    InvalidQueryError,
    UnsupportedFileTypeError,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.arq_pool = await create_pool(
        RedisSettings.from_dsn(settings.redis_url),
    )
    yield
    await app.state.arq_pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Project API", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, tags=["auth"])
    app.include_router(teams.router, tags=["teams"])
    app.include_router(projects.router, tags=["projects"])
    app.include_router(documents.router, tags=["documents"])
    app.include_router(cag.router, tags=["cag"])
    app.include_router(query.router, tags=["query"])

    if settings.app_env in ("development", "test"):
        from app.api import dev

        app.include_router(dev.router, tags=["test"])

    app.add_exception_handler(DuplicateEmailError, duplicate_email_handler)  # type: ignore[arg-type]
    app.add_exception_handler(InvalidCredentialsError, invalid_credentials_handler)  # type: ignore[arg-type]
    app.add_exception_handler(AccessDeniedError, access_denied_handler)  # type: ignore[arg-type]
    app.add_exception_handler(UnsupportedFileTypeError, unsupported_file_type_handler)  # type: ignore[arg-type]
    app.add_exception_handler(IngestionJobNotFoundError, ingestion_job_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(DocumentNotFoundError, document_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(BeliefStateNotFoundError, belief_state_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(InvalidQueryError, invalid_query_handler)  # type: ignore[arg-type]

    return app


app = create_app()
