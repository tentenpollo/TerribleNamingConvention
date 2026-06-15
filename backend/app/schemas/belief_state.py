from __future__ import annotations

from datetime import datetime
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_Theme = Annotated[str, Field(max_length=80)]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(..., max_length=500)
    approximate_date: str | None = None
    summary_id_ref: UUID | None = None

    @field_validator("approximate_date")
    @classmethod
    def _validate_date_format(cls, v: str | None) -> str | None:
        if v is not None and not _ISO_DATE_RE.match(v):
            raise ValueError("approximate_date must be an ISO date string (YYYY-MM-DD) or None")
        return v


class OpenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(..., max_length=500)
    first_seen_summary_id: UUID | None = None


class KeyPerson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., max_length=120)
    role: str | None = Field(default=None, max_length=120)


class BeliefStateContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_summary: str = Field(..., max_length=1200)
    decisions: Annotated[list[Decision], Field(max_length=100)] = Field(default_factory=list)
    open_items: Annotated[list[OpenItem], Field(max_length=100)] = Field(default_factory=list)
    key_people: Annotated[list[KeyPerson], Field(max_length=50)] = Field(default_factory=list)
    recurring_themes: Annotated[list[_Theme], Field(max_length=30)] = Field(default_factory=list)


class BeliefStateRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    version: int
    rebuild_type: str
    last_summary_created_at: datetime
    summary_count_covered: int
    created_at: datetime
    state: BeliefStateContent


class BeliefStateVersionMeta(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: int
    rebuild_type: str
    created_at: datetime
    summary_count_covered: int


class RebuildRequest(BaseModel):
    mode: Literal["compaction", "genesis"] = "compaction"


class RebuildResponse(BaseModel):
    queued: bool
    deduplicated: bool
