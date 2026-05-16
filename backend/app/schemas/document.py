from __future__ import annotations

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    filename: str
    file_type: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IngestionJobResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    document_id: uuid.UUID | None
    status: str
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)
