from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.dependencies import get_accessible_projects, get_document_service
from app.core.exceptions import (
    AccessDeniedError,
    DocumentNotFoundError,
    IngestionJobNotFoundError,
    UnsupportedFileTypeError,
)
from app.models.document import FileType
from app.schemas.document import DocumentResponse, IngestionJobResponse
from app.services.document import DocumentService

router = APIRouter()

EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    ".md": FileType.MARKDOWN,
    ".txt": FileType.TXT,
    ".pdf": FileType.PDF,
}


@router.post(
    "/projects/{project_id}/documents",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    project_id: UUID,
    file: UploadFile,
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    document_service: DocumentService = Depends(get_document_service),
) -> IngestionJobResponse:
    extension = _get_extension(file.filename or "")
    file_type = EXTENSION_TO_FILE_TYPE.get(extension)
    if file_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {extension}",
        )

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum upload size of {settings.max_upload_size_mb}MB",
        )

    return await document_service.upload(
        project_id=project_id,
        filename=file.filename or "unknown",
        file_type=file_type,
        content=content,
        accessible_ids=accessible_ids,
    )


@router.get(
    "/projects/{project_id}/documents",
    response_model=list[DocumentResponse],
)
async def list_documents(
    project_id: UUID,
    accessible_ids: list[UUID] = Depends(get_accessible_projects),
    document_service: DocumentService = Depends(get_document_service),
) -> list[DocumentResponse]:
    return await document_service.list_documents(
        project_id=project_id,
        accessible_ids=accessible_ids,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJobResponse,
)
async def get_job(
    job_id: UUID,
    document_service: DocumentService = Depends(get_document_service),
) -> IngestionJobResponse:
    return await document_service.get_job(job_id=job_id)


async def access_denied_handler(request: Request, exc: AccessDeniedError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": exc.message},
    )


async def unsupported_file_type_handler(
    request: Request,
    exc: UnsupportedFileTypeError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message},
    )


async def ingestion_job_not_found_handler(
    request: Request,
    exc: IngestionJobNotFoundError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": exc.message},
    )


async def document_not_found_handler(
    request: Request,
    exc: DocumentNotFoundError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": exc.message},
    )


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[1].lower()
