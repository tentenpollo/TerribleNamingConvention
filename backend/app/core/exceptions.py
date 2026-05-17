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


class TeamNotFoundError(AppError):
    """Raised when a requested team cannot be found."""


class ProjectNotFoundError(AppError):
    """Raised when a requested project cannot be found."""


class AlreadyMemberError(AppError):
    """Raised when adding a user who is already in a team."""


class MemberNotFoundError(AppError):
    """Raised when a requested team membership cannot be found."""


class AccessDeniedError(AppError):
    """Raised when a user requests a resource outside their accessible scope."""


class DocumentNotFoundError(AppError):
    """Raised when a requested document cannot be found."""


class IngestionJobNotFoundError(AppError):
    """Raised when a requested ingestion job cannot be found."""


class UnsupportedFileTypeError(AppError):
    """Raised when an uploaded file type is not supported."""


class QdrantError(AppError):
    """Raised when a Qdrant operation fails unexpectedly."""
