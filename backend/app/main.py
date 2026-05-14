from __future__ import annotations

from fastapi import FastAPI

from app.api import auth, health, projects, teams
from app.api.auth import (
    duplicate_email_handler,
    invalid_credentials_handler,
)
from app.core.config import settings
from app.core.exceptions import DuplicateEmailError, InvalidCredentialsError


def create_app() -> FastAPI:
    app = FastAPI(title="Project API", version="0.1.0")
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, tags=["auth"])
    app.include_router(teams.router, tags=["teams"])
    app.include_router(projects.router, tags=["projects"])

    if settings.app_env in ("development", "test"):
        from app.api import dev

        app.include_router(dev.router, tags=["test"])

    app.add_exception_handler(DuplicateEmailError, duplicate_email_handler)  # type: ignore[arg-type]
    app.add_exception_handler(InvalidCredentialsError, invalid_credentials_handler)  # type: ignore[arg-type]

    return app


app = create_app()
