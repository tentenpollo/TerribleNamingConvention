from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SourceChunk(BaseModel):
    """A single retrieved chunk used to ground the answer."""

    model_config = ConfigDict(from_attributes=True)

    document_id: UUID = Field(description="ID of the source document.")
    filename: str = Field(description="Filename of the source document.")
    chunk_index: int = Field(description="Zero-based index of the chunk within the document.")
    text: str = Field(description="Text content of the retrieved chunk.")
    score: float = Field(description="Hybrid retrieval score for this chunk.")
    label: str = Field(description="Inline citation label, e.g. S1, S2.")
    project_id: UUID = Field(description="ID of the project this chunk belongs to.")


class QueryRequest(BaseModel):
    """Request body for project or cross-project queries."""

    question: str = Field(
        ...,
        min_length=1,
        description="The question to answer. Must be non-empty and no more than 4000 characters.",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of chunks to retrieve for grounding.",
    )


class QueryResponse(BaseModel):
    """Response from a project or cross-project query."""

    answer: str = Field(description="Generated answer to the question.")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Retrieved chunks used to ground the answer, with citation labels.",
    )
    belief_state_version: int | None = Field(
        default=None,
        description="Version of the project's belief state used for orientation, if any.",
    )
    grounded: bool = Field(
        description="True when retrieval returned at least one chunk (grounding material existed).",
    )
