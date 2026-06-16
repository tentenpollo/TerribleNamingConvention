from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from unittest.mock import AsyncMock, patch
import uuid

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from sqlalchemy import delete, select, text

from app.core import database
from app.core.config import settings
from app.core.exceptions import BeliefStateVersionConflictError, InvalidBeliefStateError
from app.ingestion.embedder import Embedder, SparseEmbedder
from app.ingestion.vector_store import VectorStore
from app.models.belief_state import BeliefState
from app.models.document import Document, FileType
from app.models.document_summary import DocumentSummary
from app.models.ingestion_job import IngestionJob, JobStatus
from app.models.project import Project
from app.models.team import Team
from app.schemas.belief_state import BeliefStateContent
from app.services.cag import CAGService
from app.workers.cag import cag_rebuild, cag_update
from app.workers.ingest import ingest_document


def _sample_content() -> BeliefStateContent:
    return BeliefStateContent(
        project_summary="Integration test project summary.",
        decisions=[],
        open_items=[],
        key_people=[],
        recurring_themes=[],
    )


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def cag_project() -> dict:
    """Creates a team + project for CAG integration tests, cleans up after."""
    team_id = uuid.uuid4()
    project_id = uuid.uuid4()

    async with database.AsyncSessionLocal() as session:
        await session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team_id, "name": "CAG Test Team"},
        )
        await session.execute(
            text(
                "INSERT INTO projects (id, name, team_id, config) "
                "VALUES (:id, :name, :team_id, CAST(:config AS jsonb))"
            ),
            {
                "id": project_id,
                "name": "CAG Test Project",
                "team_id": team_id,
                "config": "{}",
            },
        )
        await session.commit()

    yield {"project_id": project_id, "team_id": team_id}

    async with database.AsyncSessionLocal() as session:
        try:
            await session.execute(
                text("DELETE FROM belief_states WHERE project_id = :pid"),
                {"pid": project_id},
            )
            await session.execute(
                text("DELETE FROM projects WHERE id = :id"),
                {"id": project_id},
            )
            await session.execute(
                text("DELETE FROM teams WHERE id = :id"),
                {"id": team_id},
            )
            await session.commit()
        finally:
            pass


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_insert_version_no_gaps_no_duplicates(cag_project) -> None:
    """20 concurrent tasks write version N+1 via separate sessions.
    All succeed; final versions are exactly 1..20 with no gaps/duplicates."""
    project_id = cag_project["project_id"]
    content = _sample_content()
    watermark = datetime.now(UTC)
    conflict_errors: list[BeliefStateVersionConflictError] = []

    async def write_one() -> int | None:
        async with database.AsyncSessionLocal() as session:
            service = CAGService(session)
            try:
                record = await service.insert_version(
                    project_id=project_id,
                    content=content,
                    rebuild_type="incremental",
                    last_summary_created_at=watermark,
                    summary_count_covered=1,
                )
                return record.version
            except BeliefStateVersionConflictError as exc:
                conflict_errors.append(exc)
                return None

    versions = await asyncio.gather(*[write_one() for _ in range(20)])

    assert not conflict_errors, f"Got {len(conflict_errors)} conflict errors"
    successful = sorted(v for v in versions if v is not None)
    assert successful == list(range(1, 21)), f"Expected 1..20, got {successful}"


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_get_latest_corrupted_state_raises_invalid(cag_project) -> None:
    """A row with an invalid state JSONB must raise InvalidBeliefStateError on get_latest."""
    project_id = cag_project["project_id"]

    async with database.AsyncSessionLocal() as session:
        await session.execute(
            text(
                "INSERT INTO belief_states "
                "(project_id, version, state, rebuild_type, "
                "last_summary_created_at, summary_count_covered) "
                "VALUES (:pid, 1, CAST(:state AS jsonb), 'full', now(), 0)"
            ),
            {"pid": project_id, "state": '{"not_valid": "shape"}'},
        )
        await session.commit()

    async with database.AsyncSessionLocal() as session:
        service = CAGService(session)
        with pytest.raises(InvalidBeliefStateError):
            await service.get_latest(project_id)


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_insert_and_read_roundtrip(cag_project) -> None:
    project_id = cag_project["project_id"]
    content = _sample_content()
    watermark = datetime.now(UTC)

    async with database.AsyncSessionLocal() as session:
        service = CAGService(session)
        record = await service.insert_version(
            project_id=project_id,
            content=content,
            rebuild_type="full",
            last_summary_created_at=watermark,
            summary_count_covered=5,
        )

    assert record.version == 1
    assert record.rebuild_type == "full"
    assert record.summary_count_covered == 5
    assert record.state.project_summary == content.project_summary

    async with database.AsyncSessionLocal() as session:
        service = CAGService(session)
        latest = await service.get_latest(project_id)

    assert latest is not None
    assert latest.version == 1


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_get_window_since_excludes_raw_text_fallback(cag_project) -> None:
    """get_window_since must exclude rows where summary->>'raw_text_fallback' is true,
    but include rows where the key is absent or the value is false."""
    project_id = cag_project["project_id"]

    # Three document_summaries rows:
    #   normal        — no raw_text_fallback key  → INCLUDED
    #   raw_fallback  — raw_text_fallback: true    → EXCLUDED
    #   explicit_false — raw_text_fallback: false  → INCLUDED
    async with database.AsyncSessionLocal() as session:
        for summary_json in [
            '{"text": "normal"}',
            '{"text": "raw", "raw_text_fallback": true}',
            '{"text": "explicit_false", "raw_text_fallback": false}',
        ]:
            doc_id = uuid.uuid4()
            await session.execute(
                text(
                    "INSERT INTO documents "
                    "(id, project_id, filename, file_type, raw_bytes) "
                    "VALUES (:id, :pid, :fn, 'txt', '')"
                ),
                {"id": doc_id, "pid": project_id, "fn": f"doc_{doc_id}.txt"},
            )
            await session.execute(
                text(
                    "INSERT INTO document_summaries "
                    "(id, document_id, project_id, summary) "
                    "VALUES (:id, :doc_id, :pid, CAST(:summary AS jsonb))"
                ),
                {
                    "id": uuid.uuid4(),
                    "doc_id": doc_id,
                    "pid": project_id,
                    "summary": summary_json,
                },
            )
        await session.commit()

    async with database.AsyncSessionLocal() as session:
        service = CAGService(session)
        rows = await service.get_window_since(project_id, watermark=None)

    texts = [r.summary["text"] for r in rows]
    assert "raw" not in texts, "raw_text_fallback:true row must be excluded"
    assert "normal" in texts
    assert "explicit_false" in texts
    # order is created_at ASC — just confirm both are present in any order
    assert len(texts) == 2


# ---------------------------------------------------------------------------
# Integration: CAG update job
# ---------------------------------------------------------------------------


_VALID_BELIEF_STATE_JSON = (
    '{"project_summary": "Integration test project summary.", '
    '"decisions": [], "open_items": [], '
    '"key_people": [], "recurring_themes": []}'
)


_SUMMARY_MOCK_RETURN = {
    "summary": "Integration test document summary.",
    "key_points": ["test"],
    "technical_concepts": [],
    "architectural_components": [],
    "decisions": [],
    "action_items": [],
    "entities": {
        "people": [],
        "organizations": [],
        "technologies": [],
        "repositories": [],
        "services": [],
    },
    "topics": [],
    "important_relationships": [],
    "document_type": "other",
    "confidence": 0.9,
}


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_first_document_ingest_creates_belief_state_v1(cag_project) -> None:
    """Ingesting the first document of a fresh project must produce belief state v1."""
    project_id = cag_project["project_id"]
    team_id = cag_project["team_id"]

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        async with database.AsyncSessionLocal() as session:
            doc = Document(
                project_id=project_id,
                filename="first-doc.md",
                file_type=FileType.MARKDOWN.value,
                raw_bytes=b"# First document\n\nThis is the first document.",
            )
            session.add(doc)
            await session.flush()

            job = IngestionJob(
                project_id=project_id,
                document_id=doc.id,
                status=JobStatus.PENDING.value,
            )
            session.add(job)
            await session.commit()

            document_id = doc.id
            job_id = job.id

        embedder = Embedder()
        sparse_embedder = SparseEmbedder()
        vector_store = VectorStore(client=qdrant_client)

        mock_arq = AsyncMock()
        mock_arq.enqueue_job.return_value = None  # simulate ARQ dedup

        ctx: dict[str, object] = {
            "embedder": embedder,
            "sparse_embedder": sparse_embedder,
            "vector_store": vector_store,
            "redis": mock_arq,
        }

        with patch("app.workers.ingest.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch("app.workers.ingest.summarize_document", return_value=_SUMMARY_MOCK_RETURN):
                await ingest_document(ctx, job_id, document_id, project_id)

        mock_arq.enqueue_job.assert_awaited_once()
        enqueue_kwargs = mock_arq.enqueue_job.call_args.kwargs
        assert enqueue_kwargs.get("_job_id") == f"cag-update-{project_id}"

        with patch("app.workers.cag.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch("app.workers.cag.llm_call", return_value=_VALID_BELIEF_STATE_JSON):
                await cag_update({"redis": mock_arq}, project_id)

        async with database.AsyncSessionLocal() as session:
            result = await session.execute(
                select(BeliefState).where(BeliefState.project_id == project_id)
            )
            rows = list(result.scalars().all())

        assert len(rows) == 1
        assert rows[0].version == 1
        assert rows[0].rebuild_type == "incremental"
        assert rows[0].summary_count_covered == 1

    finally:
        await vector_store.delete_collection(project_id)
        async with database.AsyncSessionLocal() as session:
            await session.execute(delete(BeliefState).where(BeliefState.project_id == project_id))
            await session.execute(
                delete(DocumentSummary).where(DocumentSummary.project_id == project_id)
            )
            await session.execute(delete(Document).where(Document.project_id == project_id))
            await session.execute(delete(IngestionJob).where(IngestionJob.project_id == project_id))
            await session.execute(delete(Project).where(Project.id == project_id))
            await session.execute(delete(Team).where(Team.id == team_id))
            await session.commit()


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_ingest_triggers_single_cag_version(cag_project) -> None:
    """With threshold=5, 10 concurrent ingests should produce one version covering all 10."""
    project_id = cag_project["project_id"]
    team_id = cag_project["team_id"]

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        async with database.AsyncSessionLocal() as session:
            # Set threshold to 5
            await session.execute(
                text("UPDATE projects SET config = CAST(:config AS jsonb) WHERE id = :id"),
                {"id": project_id, "config": '{"cag_rebuild_threshold": 5}'},
            )

            doc_ids: list[uuid.UUID] = []
            job_ids: list[uuid.UUID] = []
            for i in range(10):
                doc = Document(
                    project_id=project_id,
                    filename=f"concurrent-{i}.md",
                    file_type=FileType.MARKDOWN.value,
                    raw_bytes=f"# Document {i}\n\nContent for document {i}.".encode(),
                )
                session.add(doc)
                await session.flush()

                job = IngestionJob(
                    project_id=project_id,
                    document_id=doc.id,
                    status=JobStatus.PENDING.value,
                )
                session.add(job)
                await session.flush()

                doc_ids.append(doc.id)
                job_ids.append(job.id)

            await session.commit()

        embedder = Embedder()
        sparse_embedder = SparseEmbedder()
        vector_store = VectorStore(client=qdrant_client)

        # Pre-create the collection so concurrent upserts do not race on creation.
        await vector_store.ensure_collection(project_id)

        mock_arq = AsyncMock()
        mock_arq.enqueue_job.return_value = None  # simulate ARQ dedup

        ctx: dict[str, object] = {
            "embedder": embedder,
            "sparse_embedder": sparse_embedder,
            "vector_store": vector_store,
            "redis": mock_arq,
        }

        with patch("app.workers.ingest.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch("app.workers.ingest.summarize_document", return_value=_SUMMARY_MOCK_RETURN):
                await asyncio.gather(
                    *[ingest_document(ctx, job_ids[i], doc_ids[i], project_id) for i in range(10)]
                )

        # Every ingest that crossed the threshold attempted enqueue with the fixed job id.
        enqueue_attempts = [
            call
            for call in mock_arq.enqueue_job.call_args_list
            if call.kwargs.get("_job_id") == f"cag-update-{project_id}"
        ]
        assert len(enqueue_attempts) >= 1

        # Simulate ARQ dedup: only one cag_update actually runs.
        with patch("app.workers.cag.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch("app.workers.cag.llm_call", return_value=_VALID_BELIEF_STATE_JSON):
                await cag_update({"redis": mock_arq}, project_id)

        async with database.AsyncSessionLocal() as session:
            result = await session.execute(
                select(BeliefState).where(BeliefState.project_id == project_id)
            )
            rows = list(result.scalars().all())

        # Either one version covering all 10, or multiple versions totaling 10 with
        # monotonically increasing watermarks and no double-counting.
        total_covered = sum(r.summary_count_covered for r in rows)
        assert total_covered == 10, f"Expected total coverage 10, got {total_covered}"

        if len(rows) > 1:
            watermarks = [r.last_summary_created_at for r in rows]
            assert watermarks == sorted(watermarks), "Watermarks must be monotonically increasing"
            counts = [r.summary_count_covered for r in rows]
            assert counts == sorted(counts), "summary_count_covered must increase across versions"

    finally:
        await vector_store.delete_collection(project_id)
        async with database.AsyncSessionLocal() as session:
            await session.execute(delete(BeliefState).where(BeliefState.project_id == project_id))
            await session.execute(
                delete(DocumentSummary).where(DocumentSummary.project_id == project_id)
            )
            await session.execute(delete(Document).where(Document.project_id == project_id))
            await session.execute(delete(IngestionJob).where(IngestionJob.project_id == project_id))
            await session.execute(delete(Project).where(Project.id == project_id))
            await session.execute(delete(Team).where(Team.id == team_id))
            await session.commit()


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_fallback_only_summaries_never_trigger_insert(cag_project) -> None:
    """If all summaries for a project are raw_text_fallback rows, cag_update must no-op."""
    project_id = cag_project["project_id"]
    team_id = cag_project["team_id"]

    try:
        async with database.AsyncSessionLocal() as session:
            doc_id = uuid.uuid4()
            await session.execute(
                text(
                    "INSERT INTO documents "
                    "(id, project_id, filename, file_type, raw_bytes) "
                    "VALUES (:id, :pid, :fn, 'txt', '')"
                ),
                {"id": doc_id, "pid": project_id, "fn": "fallback.txt"},
            )
            await session.execute(
                text(
                    "INSERT INTO document_summaries "
                    "(id, document_id, project_id, summary) "
                    "VALUES (:id, :doc_id, :pid, CAST(:summary AS jsonb))"
                ),
                {
                    "id": uuid.uuid4(),
                    "doc_id": doc_id,
                    "pid": project_id,
                    "summary": '{"raw_text_fallback": true, "text": "fallback"}',
                },
            )
            await session.commit()

        mock_llm = AsyncMock()
        mock_arq = AsyncMock()

        with patch("app.workers.cag.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch("app.workers.cag.llm_call", mock_llm):
                await cag_update({"redis": mock_arq}, project_id)

        mock_llm.assert_not_awaited()

        async with database.AsyncSessionLocal() as session:
            result = await session.execute(
                select(BeliefState).where(BeliefState.project_id == project_id)
            )
            rows = list(result.scalars().all())

        assert len(rows) == 0

    finally:
        async with database.AsyncSessionLocal() as session:
            await session.execute(delete(BeliefState).where(BeliefState.project_id == project_id))
            await session.execute(
                delete(DocumentSummary).where(DocumentSummary.project_id == project_id)
            )
            await session.execute(delete(Document).where(Document.project_id == project_id))
            await session.execute(delete(Project).where(Project.id == project_id))
            await session.execute(delete(Team).where(Team.id == team_id))
            await session.commit()


# ---------------------------------------------------------------------------
# Integration: CAG rebuild job
# ---------------------------------------------------------------------------


_DETERMINISTIC_BELIEF_STATE_JSON = (
    '{"project_summary": "Union of inputs.", '
    '"decisions": [], "open_items": [], '
    '"key_people": [], "recurring_themes": []}'
)


_DETERMINISTIC_OPEN_ITEM_STATE = (
    '{"project_summary": "Open item tracked.", '
    '"decisions": [], '
    '"open_items": [{"description": "Need API spec", "first_seen_summary_id": null}], '
    '"key_people": [], "recurring_themes": []}'
)


_DETERMINISTIC_RESOLVED_STATE = (
    '{"project_summary": "Open item resolved.", '
    '"decisions": [], "open_items": [], '
    '"key_people": [], "recurring_themes": []}'
)


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_genesis_rebuild_closes_explicitly_resolved_open_item(cag_project) -> None:
    """A genesis rebuild closes an open item when a later summary explicitly resolves it."""
    project_id = cag_project["project_id"]
    team_id = cag_project["team_id"]

    try:
        async with database.AsyncSessionLocal() as session:
            early_doc_id = uuid.uuid4()
            early_summary_id = uuid.uuid4()
            early_created_at = datetime.now(UTC)
            await session.execute(
                text(
                    "INSERT INTO documents "
                    "(id, project_id, filename, file_type, raw_bytes) "
                    "VALUES (:id, :pid, :fn, 'txt', '')"
                ),
                {"id": early_doc_id, "pid": project_id, "fn": "early.txt"},
            )
            await session.execute(
                text(
                    "INSERT INTO document_summaries "
                    "(id, document_id, project_id, summary, created_at) "
                    "VALUES (:id, :doc_id, :pid, CAST(:summary AS jsonb), :created_at)"
                ),
                {
                    "id": early_summary_id,
                    "doc_id": early_doc_id,
                    "pid": project_id,
                    "summary": json.dumps({"text": "We still need an API spec."}),
                    "created_at": early_created_at,
                },
            )

            later_doc_id = uuid.uuid4()
            later_summary_id = uuid.uuid4()
            later_created_at = early_created_at + timedelta(seconds=1)
            await session.execute(
                text(
                    "INSERT INTO documents "
                    "(id, project_id, filename, file_type, raw_bytes) "
                    "VALUES (:id, :pid, :fn, 'txt', '')"
                ),
                {"id": later_doc_id, "pid": project_id, "fn": "later.txt"},
            )
            await session.execute(
                text(
                    "INSERT INTO document_summaries "
                    "(id, document_id, project_id, summary, created_at) "
                    "VALUES (:id, :doc_id, :pid, CAST(:summary AS jsonb), :created_at)"
                ),
                {
                    "id": later_summary_id,
                    "doc_id": later_doc_id,
                    "pid": project_id,
                    "summary": json.dumps(
                        {"text": "The API spec has been finalized and approved."}
                    ),
                    "created_at": later_created_at,
                },
            )
            await session.commit()

        async def resolve_open_item(*args: object, **kwargs: object) -> str:
            messages = kwargs.get("messages", args[0] if args else [])
            content = messages[-1]["content"] if messages else ""
            if "API spec has been finalized" in content:
                return _DETERMINISTIC_RESOLVED_STATE
            return _DETERMINISTIC_OPEN_ITEM_STATE

        mock_arq = AsyncMock()
        ctx: dict[str, object] = {"redis": mock_arq}

        with patch("app.workers.cag.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch("app.workers.cag.llm_call", side_effect=resolve_open_item):
                await cag_rebuild(ctx, project_id, "genesis")

        async with database.AsyncSessionLocal() as session:
            result = await session.execute(
                select(BeliefState)
                .where(BeliefState.project_id == project_id)
                .order_by(BeliefState.version.desc())
            )
            rows = list(result.scalars().all())

            assert len(rows) == 1
            full_row = rows[0]
            assert full_row.rebuild_type == "full"
            assert full_row.summary_count_covered == 2
            state = BeliefStateContent.model_validate(full_row.state)
            assert state.open_items == []

    finally:
        async with database.AsyncSessionLocal() as session:
            await session.execute(delete(BeliefState).where(BeliefState.project_id == project_id))
            await session.execute(
                delete(DocumentSummary).where(DocumentSummary.project_id == project_id)
            )
            await session.execute(delete(Document).where(Document.project_id == project_id))
            await session.execute(delete(Project).where(Project.id == project_id))
            await session.execute(delete(Team).where(Team.id == team_id))
            await session.commit()


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_genesis_rebuild_on_120_summaries(cag_project) -> None:
    """Genesis rebuild over 120 seeded summaries produces one full belief-state row."""
    project_id = cag_project["project_id"]
    team_id = cag_project["team_id"]

    try:
        async with database.AsyncSessionLocal() as session:
            latest_created_at: datetime | None = None
            for i in range(120):
                doc_id = uuid.uuid4()
                summary_id = uuid.uuid4()
                created_at = datetime.now(UTC) + timedelta(seconds=i)
                latest_created_at = created_at
                await session.execute(
                    text(
                        "INSERT INTO documents "
                        "(id, project_id, filename, file_type, raw_bytes) "
                        "VALUES (:id, :pid, :fn, 'txt', '')"
                    ),
                    {"id": doc_id, "pid": project_id, "fn": f"seed-{i}.txt"},
                )
                await session.execute(
                    text(
                        "INSERT INTO document_summaries "
                        "(id, document_id, project_id, summary, created_at) "
                        "VALUES (:id, :doc_id, :pid, CAST(:summary AS jsonb), :created_at)"
                    ),
                    {
                        "id": summary_id,
                        "doc_id": doc_id,
                        "pid": project_id,
                        "summary": json.dumps({"text": f"summary {i}"}),
                        "created_at": created_at,
                    },
                )
            await session.commit()

        # Seed an older incremental version so we can verify prior versions survive.
        async with database.AsyncSessionLocal() as session:
            await session.execute(
                text(
                    "INSERT INTO belief_states "
                    "(project_id, version, state, rebuild_type, "
                    "last_summary_created_at, summary_count_covered) "
                    "VALUES (:pid, 1, CAST(:state AS jsonb), 'incremental', now(), 1)"
                ),
                {
                    "pid": project_id,
                    "state": _DETERMINISTIC_BELIEF_STATE_JSON,
                },
            )
            await session.commit()

        embedder = Embedder()
        mock_arq = AsyncMock()
        ctx: dict[str, object] = {
            "embedder": embedder,
            "redis": mock_arq,
        }

        with patch("app.workers.cag.AsyncSessionLocal", database.AsyncSessionLocal):
            with patch(
                "app.workers.cag.llm_call",
                return_value=_DETERMINISTIC_BELIEF_STATE_JSON,
            ):
                await cag_rebuild(ctx, project_id, "genesis")

        async with database.AsyncSessionLocal() as session:
            result = await session.execute(
                select(BeliefState)
                .where(BeliefState.project_id == project_id)
                .order_by(BeliefState.version.desc())
            )
            rows = list(result.scalars().all())

            assert len(rows) == 2
            full_row = rows[0]
            assert full_row.rebuild_type == "full"
            assert full_row.summary_count_covered == 120
            assert full_row.last_summary_created_at == latest_created_at

            service = CAGService(session)
            versions = await service.list_versions(project_id)
            assert len(versions) == 2
            assert {v.rebuild_type for v in versions} == {"incremental", "full"}

    finally:
        async with database.AsyncSessionLocal() as session:
            await session.execute(delete(BeliefState).where(BeliefState.project_id == project_id))
            await session.execute(
                delete(DocumentSummary).where(DocumentSummary.project_id == project_id)
            )
            await session.execute(delete(Document).where(Document.project_id == project_id))
            await session.execute(delete(Project).where(Project.id == project_id))
            await session.execute(delete(Team).where(Team.id == team_id))
            await session.commit()
