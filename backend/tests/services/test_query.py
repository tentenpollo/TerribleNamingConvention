from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from structlog.testing import capture_logs

from app.core.exceptions import (
    AccessDeniedError,
    InvalidBeliefStateError,
    InvalidQueryError,
    ProjectNotFoundError,
    QueryGenerationError,
)
from app.core.llm import LLMResult
from app.models.project import Project
from app.retrieval.retriever import RetrievedChunk
from app.schemas.belief_state import BeliefStateContent
from app.schemas.query import SourceChunk
from app.services.query import QueryService


@pytest.fixture
def mock_session() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_embedder() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_sparse_embedder() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_cag_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def query_service(
    mock_session: AsyncMock,
    mock_vector_store: AsyncMock,
    mock_embedder: MagicMock,
    mock_sparse_embedder: MagicMock,
    mock_cag_service: AsyncMock,
) -> QueryService:
    return QueryService(
        session=mock_session,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        sparse_embedder=mock_sparse_embedder,
        cag_service=mock_cag_service,
    )


@pytest.mark.asyncio
async def test_query_happy_path_response_shape(
    query_service: QueryService,
    mock_session: AsyncMock,
    mock_cag_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    document_id = uuid.uuid4()
    project = _make_project(project_id)
    mock_session.get.return_value = project

    belief_state = BeliefStateContent(project_summary="Summary")
    mock_cag_service.get_latest.return_value = _make_belief_record(
        project_id=project_id,
        version=3,
        state=belief_state,
    )

    chunks = [
        RetrievedChunk(
            document_id=document_id,
            chunk_index=0,
            text="chunk one",
            filename="notes.md",
            score=0.9,
            project_id=project_id,
        ),
    ]

    user_id = uuid.uuid4()

    with (
        patch("app.services.query.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call,
    ):
        mock_retrieve.return_value = chunks
        mock_llm_call.return_value = LLMResult(
            text="The answer is 42.",
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4o-mini",
        )

        response = await query_service.query(
            question="What is the answer?",
            project_id=project_id,
            accessible_ids=[project_id],
            user_id=user_id,
            top_k=8,
        )

    assert response.answer == "The answer is 42."
    assert response.belief_state_version == 3
    assert response.grounded is True
    assert len(response.sources) == 1
    assert response.sources[0] == SourceChunk(
        document_id=document_id,
        filename="notes.md",
        chunk_index=0,
        text="chunk one",
        score=0.9,
        label="S1",
        project_id=project_id,
    )
    mock_llm_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_empty_everything_short_circuits_without_llm(
    query_service: QueryService,
    mock_session: AsyncMock,
    mock_cag_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()
    mock_session.get.return_value = _make_project(project_id)
    mock_cag_service.get_latest.return_value = None

    with (
        patch("app.services.query.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call,
    ):
        mock_retrieve.return_value = []

        response = await query_service.query(
            question="What?",
            project_id=project_id,
            accessible_ids=[project_id],
            user_id=user_id,
        )

    assert response.grounded is False
    assert response.belief_state_version is None
    assert response.sources == []
    assert "No indexed content" in response.answer
    mock_llm_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_corrupt_belief_state_degrades_to_rag_only_and_logs(
    query_service: QueryService,
    mock_session: AsyncMock,
    mock_cag_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()
    document_id = uuid.uuid4()
    mock_session.get.return_value = _make_project(project_id)
    mock_cag_service.get_latest.side_effect = InvalidBeliefStateError("bad state")

    chunks = [
        RetrievedChunk(
            document_id=document_id,
            chunk_index=0,
            text="chunk",
            filename="notes.md",
            score=0.5,
            project_id=project_id,
        ),
    ]

    with (
        patch("app.services.query.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call,
        patch("app.services.query.logger") as mock_logger,
    ):
        mock_retrieve.return_value = chunks
        mock_llm_call.return_value = LLMResult(
            text="answer",
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4o-mini",
        )

        response = await query_service.query(
            question="Q",
            project_id=project_id,
            accessible_ids=[project_id],
            user_id=user_id,
        )

    assert response.belief_state_version is None
    assert response.grounded is True
    mock_logger.error.assert_called_once()
    mock_llm_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_project_not_found_raises(
    query_service: QueryService,
    mock_session: AsyncMock,
) -> None:
    mock_session.get.return_value = None
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()

    with pytest.raises(ProjectNotFoundError):
        await query_service.query(
            question="Q",
            project_id=project_id,
            accessible_ids=[project_id],
            user_id=user_id,
        )


@pytest.mark.asyncio
async def test_query_out_of_scope_raises_access_denied(
    query_service: QueryService,
    mock_session: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()
    mock_session.get.return_value = _make_project(project_id)

    with patch("app.services.query.retrieve", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.side_effect = AccessDeniedError("Denied")

        with pytest.raises(AccessDeniedError):
            await query_service.query(
                question="Q",
                project_id=project_id,
                accessible_ids=[uuid.uuid4()],
                user_id=user_id,
            )


@pytest.mark.asyncio
async def test_query_question_too_long_raises(
    query_service: QueryService,
) -> None:
    with pytest.raises(InvalidQueryError):
        await query_service.query(
            question="x" * 4001,
            project_id=uuid.uuid4(),
            accessible_ids=[],
            user_id=uuid.uuid4(),
        )


@pytest.mark.asyncio
async def test_query_whitespace_only_question_raises(
    query_service: QueryService,
) -> None:
    with pytest.raises(InvalidQueryError):
        await query_service.query(
            question="   \n\t  ",
            project_id=uuid.uuid4(),
            accessible_ids=[],
            user_id=uuid.uuid4(),
        )


@pytest.mark.asyncio
async def test_query_uses_project_config_model(
    query_service: QueryService,
    mock_session: AsyncMock,
    mock_cag_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()
    project = _make_project(project_id, config={"query_model": "custom-model"})
    mock_session.get.return_value = project
    mock_cag_service.get_latest.return_value = None

    with (
        patch("app.services.query.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call,
    ):
        mock_retrieve.return_value = [
            RetrievedChunk(
                document_id=uuid.uuid4(),
                chunk_index=0,
                text="chunk",
                filename="notes.md",
                score=0.5,
                project_id=project_id,
            ),
        ]
        mock_llm_call.return_value = LLMResult(
            text="ok",
            prompt_tokens=10,
            completion_tokens=5,
            model="custom-model",
        )

        await query_service.query(
            question="Q",
            project_id=project_id,
            accessible_ids=[project_id],
            user_id=user_id,
        )

    call_kwargs = mock_llm_call.call_args.kwargs
    assert call_kwargs["model"] == "custom-model"


@pytest.mark.asyncio
async def test_query_cross_project_uses_default_model_and_no_belief(
    query_service: QueryService,
    mock_cag_service: AsyncMock,
) -> None:
    user_id = uuid.uuid4()
    project_a = uuid.uuid4()
    project_b = uuid.uuid4()
    accessible_ids = [project_a, project_b]

    chunks = [
        RetrievedChunk(
            document_id=uuid.uuid4(),
            chunk_index=0,
            text="chunk a",
            filename="a.md",
            score=0.9,
            project_id=project_a,
        ),
        RetrievedChunk(
            document_id=uuid.uuid4(),
            chunk_index=1,
            text="chunk b",
            filename="b.md",
            score=0.8,
            project_id=project_b,
        ),
    ]

    with (
        patch("app.services.query.retrieve_multi", new_callable=AsyncMock) as mock_retrieve_multi,
        patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call,
    ):
        mock_retrieve_multi.return_value = chunks
        mock_llm_call.return_value = LLMResult(
            text="cross answer",
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4o-mini",
        )

        response = await query_service.query_cross_project(
            question="Q",
            accessible_ids=accessible_ids,
            user_id=user_id,
            top_k=4,
        )

    assert response.belief_state_version is None
    assert response.grounded is True
    assert {source.project_id for source in response.sources} == {project_a, project_b}
    mock_cag_service.get_latest.assert_not_awaited()
    mock_llm_call.assert_awaited_once()
    call_kwargs = mock_llm_call.call_args.kwargs
    assert "temperature" in call_kwargs


@pytest.mark.asyncio
async def test_query_cross_project_empty_accessible_short_circuits(
    query_service: QueryService,
) -> None:
    with patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call:
        response = await query_service.query_cross_project(
            question="Q",
            accessible_ids=[],
            user_id=uuid.uuid4(),
        )

    assert response.grounded is False
    assert response.sources == []
    assert response.belief_state_version is None
    mock_llm_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_generation_timeout_raises_query_generation_error(
    query_service: QueryService,
    mock_session: AsyncMock,
    mock_cag_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()
    mock_session.get.return_value = _make_project(project_id)
    mock_cag_service.get_latest.return_value = None

    with (
        patch("app.services.query.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call,
        patch("app.services.query.logger") as mock_logger,
    ):
        mock_retrieve.return_value = [
            RetrievedChunk(
                document_id=uuid.uuid4(),
                chunk_index=0,
                text="chunk",
                filename="notes.md",
                score=0.5,
                project_id=project_id,
            ),
        ]
        mock_llm_call.side_effect = TimeoutError("LLM timed out")

        with pytest.raises(QueryGenerationError):
            await query_service.query(
                question="Q",
                project_id=project_id,
                accessible_ids=[project_id],
                user_id=user_id,
            )

    mock_logger.error.assert_called_once()
    call_kwargs = mock_logger.error.call_args.kwargs
    assert call_kwargs["chunk_count"] == 1
    assert call_kwargs["belief_state_version"] is None


@pytest.mark.asyncio
async def test_query_generation_llm_error_raises_query_generation_error(
    query_service: QueryService,
    mock_session: AsyncMock,
    mock_cag_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()
    mock_session.get.return_value = _make_project(project_id)
    mock_cag_service.get_latest.return_value = _make_belief_record(
        project_id=project_id,
        version=2,
        state=BeliefStateContent(project_summary="Summary"),
    )

    with (
        patch("app.services.query.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call,
        patch("app.services.query.logger") as mock_logger,
    ):
        mock_retrieve.return_value = [
            RetrievedChunk(
                document_id=uuid.uuid4(),
                chunk_index=0,
                text="chunk",
                filename="notes.md",
                score=0.5,
                project_id=project_id,
            ),
        ]
        from app.core.exceptions import LLMError

        mock_llm_call.side_effect = LLMError("provider down")

        with pytest.raises(QueryGenerationError):
            await query_service.query(
                question="Q",
                project_id=project_id,
                accessible_ids=[project_id],
                user_id=user_id,
            )

    mock_logger.error.assert_called_once()
    call_kwargs = mock_logger.error.call_args.kwargs
    assert call_kwargs["chunk_count"] == 1
    assert call_kwargs["belief_state_version"] == 2


@pytest.mark.asyncio
async def test_query_completed_log_contains_no_user_content(
    query_service: QueryService,
    mock_session: AsyncMock,
    mock_cag_service: AsyncMock,
) -> None:
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()
    document_id = uuid.uuid4()
    mock_session.get.return_value = _make_project(project_id)
    mock_cag_service.get_latest.return_value = _make_belief_record(
        project_id=project_id,
        version=1,
        state=BeliefStateContent(project_summary="Summary"),
    )

    chunks = [
        RetrievedChunk(
            document_id=document_id,
            chunk_index=0,
            text="secret chunk text",
            filename="secret-notes.md",
            score=0.9,
            project_id=project_id,
        ),
    ]

    with (
        patch("app.services.query.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.query.llm_call", new_callable=AsyncMock) as mock_llm_call,
    ):
        mock_retrieve.return_value = chunks
        mock_llm_call.return_value = LLMResult(
            text="secret answer text",
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4o-mini",
        )

        with capture_logs() as cap_logs:
            await query_service.query(
                question="secret question text",
                project_id=project_id,
                accessible_ids=[project_id],
                user_id=user_id,
                top_k=8,
            )

    query_completed_events = [
        event for event in cap_logs if event.get("event") == "query_completed"
    ]
    assert len(query_completed_events) == 1
    event = query_completed_events[0]
    assert event["prompt_tokens"] == 10
    assert event["completion_tokens"] == 5
    assert event["model"] == "gpt-4o-mini"
    assert event["chunk_count"] == 1
    assert event["belief_state_version"] == 1

    all_log_text = " ".join(str(entry) for entry in cap_logs)
    assert "secret question text" not in all_log_text
    assert "secret chunk text" not in all_log_text
    assert "secret answer text" not in all_log_text


def _make_project(project_id: uuid.UUID, config: dict | None = None) -> Project:
    return Project(
        id=project_id,
        name="Test Project",
        description="",
        team_id=uuid.uuid4(),
        config=config or {},
        created_at=datetime.now(UTC),
    )


def _make_belief_record(
    project_id: uuid.UUID,
    version: int,
    state: BeliefStateContent,
) -> object:
    from app.schemas.belief_state import BeliefStateRecord

    return BeliefStateRecord(
        id=uuid.uuid4(),
        project_id=project_id,
        version=version,
        rebuild_type="incremental",
        last_summary_created_at=datetime.now(UTC),
        summary_count_covered=1,
        created_at=datetime.now(UTC),
        state=state,
    )
