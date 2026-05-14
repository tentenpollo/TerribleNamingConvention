from __future__ import annotations


class AppError(Exception):
    """Base class for all application-specific errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DuplicateEmailError(AppError):
    """Raised when attempting to register with an email that already exists."""


class InvalidCredentialsError(AppError):
    """Raised when login credentials are invalid."""


class UserNotFoundError(AppError):
    """Raised when a requested user cannot be found."""
