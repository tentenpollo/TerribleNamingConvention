from __future__ import annotations

import asyncio
import time
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    BeliefStateNotFoundError,
    InvalidBeliefStateError,
    InvalidQueryError,
    LLMError,
    ProjectNotFoundError,
    QueryGenerationError,
)
from app.core.llm import LLMResult, llm_call
from app.core.logging import logger
from app.ingestion.embedder import Embedder, SparseEmbedder
from app.ingestion.vector_store import VectorStore
from app.models.project import Project
from app.retrieval.prompting import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import RetrievedChunk, retrieve, retrieve_multi
from app.schemas.belief_state import BeliefStateContent
from app.schemas.query import QueryResponse, SourceChunk
from app.services.cag import CAGService

MAX_QUESTION_LENGTH = 4000


class QueryService:
    def __init__(
        self,
        session: AsyncSession,
        vector_store: VectorStore,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder,
        cag_service: CAGService,
    ) -> None:
        self.session = session
        self.vector_store = vector_store
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.cag_service = cag_service

    async def query(
        self,
        question: str,
        project_id: UUID,
        accessible_ids: list[UUID],
        user_id: UUID,
        top_k: int = 8,
    ) -> QueryResponse:
        """Answer a question scoped to a single project."""
        start_time = time.perf_counter()
        self._validate_question(question)

        project = await self.session.get(Project, project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project {project_id} not found")

        belief_record = None
        try:
            belief_record = await self.cag_service.get_latest(project_id)
        except (InvalidBeliefStateError, BeliefStateNotFoundError) as exc:
            logger.error(
                "Corrupt or missing belief state ignored during query",
                project_id=str(project_id),
                error=str(exc),
            )

        belief_state: BeliefStateContent | None = None
        belief_version: int | None = None
        if belief_record is not None:
            belief_state = belief_record.state
            belief_version = belief_record.version

        chunks = await retrieve(
            project_id=project_id,
            query_text=question,
            accessible_ids=accessible_ids,
            vector_store=self.vector_store,
            embedder=self.embedder,
            sparse_embedder=self.sparse_embedder,
            top_k=top_k,
        )

        if not chunks and belief_state is None:
            return QueryResponse(
                answer="No indexed content is available for this project yet.",
                sources=[],
                belief_state_version=None,
                grounded=False,
            )

        sources = _to_source_chunks(chunks)
        model: str = project.config.get("query_model", settings.litellm_query_model)
        user_prompt = build_user_prompt(belief_state, chunks, question)

        try:
            llm_result = await self._generate_answer(model=model, user_prompt=user_prompt)
        except (TimeoutError, LLMError) as exc:
            self._log_generation_failure(
                exc=exc,
                project_id=project_id,
                chunk_count=len(chunks),
                belief_state_version=belief_version,
            )
            raise QueryGenerationError("Query generation failed; please retry") from exc

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        self._emit_query_completed(
            user_id=user_id,
            project_id=project_id,
            top_k=top_k,
            chunk_count=len(chunks),
            belief_state_version=belief_version,
            llm_result=llm_result,
            duration_ms=duration_ms,
            grounded=len(chunks) > 0,
        )

        return QueryResponse(
            answer=llm_result.text,
            sources=sources,
            belief_state_version=belief_version,
            grounded=len(chunks) > 0,
        )

    async def query_cross_project(
        self,
        question: str,
        accessible_ids: list[UUID],
        user_id: UUID,
        top_k: int = 8,
    ) -> QueryResponse:
        """Answer a question across all accessible projects.

        Cross-project state fusion is undesigned in v1, so no belief state is
        injected. Sources carry their project_id so the client can group results.
        """
        start_time = time.perf_counter()
        self._validate_question(question)

        chunks: list[RetrievedChunk] = []
        if accessible_ids:
            chunks = await retrieve_multi(
                project_ids=accessible_ids,
                query_text=question,
                accessible_ids=accessible_ids,
                vector_store=self.vector_store,
                embedder=self.embedder,
                sparse_embedder=self.sparse_embedder,
                top_k=top_k,
            )

        if not chunks:
            return QueryResponse(
                answer="No indexed content is available across any accessible projects.",
                sources=[],
                belief_state_version=None,
                grounded=False,
            )

        sources = _to_source_chunks(chunks)
        model: str = settings.litellm_query_model
        user_prompt = build_user_prompt(None, chunks, question, include_project_id=True)

        try:
            llm_result = await self._generate_answer(model=model, user_prompt=user_prompt)
        except (TimeoutError, LLMError) as exc:
            self._log_generation_failure(
                exc=exc,
                project_id=None,
                chunk_count=len(chunks),
                belief_state_version=None,
            )
            raise QueryGenerationError("Query generation failed; please retry") from exc

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        self._emit_query_completed(
            user_id=user_id,
            project_id=None,
            top_k=top_k,
            chunk_count=len(chunks),
            belief_state_version=None,
            llm_result=llm_result,
            duration_ms=duration_ms,
            grounded=True,
        )

        return QueryResponse(
            answer=llm_result.text,
            sources=sources,
            belief_state_version=None,
            grounded=True,
        )

    async def _generate_answer(self, model: str, user_prompt: str) -> LLMResult:
        async with asyncio.timeout(settings.query_llm_timeout_seconds):
            return await llm_call(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=0.2,
                max_tokens=2000,
            )

    def _log_generation_failure(
        self,
        exc: BaseException,
        project_id: UUID | None,
        chunk_count: int,
        belief_state_version: int | None,
    ) -> None:
        logger.error(
            "Query generation failed",
            project_id=str(project_id) if project_id else "cross",
            chunk_count=chunk_count,
            belief_state_version=belief_state_version,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def _emit_query_completed(
        self,
        user_id: UUID | None,
        project_id: UUID | None,
        top_k: int,
        chunk_count: int,
        belief_state_version: int | None,
        llm_result: LLMResult,
        duration_ms: int,
        grounded: bool,
    ) -> None:
        logger.info(
            "query_completed",
            user_id=str(user_id) if user_id else None,
            project_id=str(project_id) if project_id else "cross",
            top_k=top_k,
            chunk_count=chunk_count,
            belief_state_version=belief_state_version,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            model=llm_result.model,
            duration_ms=duration_ms,
            grounded=grounded,
        )

    def _validate_question(self, question: str) -> None:
        if not question or not question.strip():
            raise InvalidQueryError("Question cannot be empty")
        if len(question) > MAX_QUESTION_LENGTH:
            raise InvalidQueryError(
                f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters"
            )


def _to_source_chunks(chunks: list[RetrievedChunk]) -> list[SourceChunk]:
    return [
        SourceChunk(
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=chunk.score,
            label=f"S{index}",
            project_id=chunk.project_id,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
