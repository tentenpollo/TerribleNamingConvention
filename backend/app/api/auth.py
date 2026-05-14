from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.core.dependencies import get_auth_service
from app.core.exceptions import DuplicateEmailError, InvalidCredentialsError
from app.schemas.user import LoginRequest, TokenResponse, UserCreate, UserResponse
from app.services.auth import AuthService

router = APIRouter(prefix="/auth")


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    user = await auth_service.register(data)
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(
    data: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    token = await auth_service.login(data)
    return TokenResponse(access_token=token)


@router.get("/me")
async def get_current_user_stub() -> dict[str, str]:
    return {"message": "not implemented yet"}


async def duplicate_email_handler(
    request: Request,
    exc: DuplicateEmailError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message},
    )


async def invalid_credentials_handler(
    request: Request,
    exc: InvalidCredentialsError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": exc.message},
    )
