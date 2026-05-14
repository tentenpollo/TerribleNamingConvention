from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings


class ProjectConfig(BaseModel):
    chunking_strategy: Literal["naive", "contextual", "late"] = "contextual"
    context_model: str = Field(default_factory=lambda: settings.litellm_context_model)
    query_model: str = Field(default_factory=lambda: settings.litellm_query_model)
    cag_rebuild_threshold: int = Field(
        default_factory=lambda: settings.default_cag_rebuild_threshold,
    )


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=2)
    description: str | None = None
    team_id: uuid.UUID
    config: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class ProjectResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    team_id: uuid.UUID
    config: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
