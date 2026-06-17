from __future__ import annotations

from unittest.mock import patch
import uuid

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from sqlalchemy import delete, select

from app.core import database
from app.core.config import settings
from app.ingestion.chunker import ChunkingStrategy, chunk_text
from app.ingestion.embedder import Embedder, SparseEmbedder
from app.ingestion.parser import parse_file
from app.ingestion.vector_store import VectorStore, collection_name
from app.models.document import Document, FileType
from app.models.project import Project
from app.models.team import Team
from app.retrieval.retriever import retrieve, retrieve_multi
from app.workers.maintenance import reindex_project

_CORPUS = [
    "The quarterly budget review is scheduled for next Tuesday.",
    "Marketing requested new landing page copy by Friday.",
    "Backend migration from Postgres 15 to 16 is complete.",
    "The CI pipeline now runs lint and test on every pull request.",
    "Customer support reported an outage in the EU region last night.",
    "The design system tokens were updated to support dark mode.",
    (
        "Quarterly planning, marketing campaigns, backend migrations, CI pipelines, "
        "customer support, design systems, new hire onboarding, mobile releases, "
        "analytics dashboards, and legal approvals all require attention. "
        "Also noted: XJ9KL2MQ4ROTATION."
    ),
    "The ZX81 home computer firmware rotation issue requires urgent review and testing.",
    "Firmware rotation is a common maintenance task for embedded systems.",
    "Home computer enthusiasts often discuss ZX81 hardware modifications.",
    "Testing and review are essential parts of the firmware development lifecycle.",
    "New hire onboarding docs need a section about SSH key setup.",
    "The mobile app release candidate passed regression testing.",
    "Analytics dashboard shows a 12% increase in signups this month.",
    "The legal team approved the updated terms of service.",
]

_REINDEX_CORPUS = [
    "The quarterly budget review is scheduled for next Tuesday.",
    (
        "During the retro computing review we noted that ZX81-FIRMWARE-ROTATION "
        "is a concern, alongside several other minor issues."
    ),
    "The ZX81 home computer firmware rotation issue requires urgent review and testing.",
]


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def qdrant_client() -> AsyncQdrantClient:
    client = AsyncQdrantClient(url=settings.qdrant_url)
    yield client
    await client.close()


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def vector_store(qdrant_client: AsyncQdrantClient) -> VectorStore:
    return VectorStore(client=qdrant_client)


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def embedders() -> tuple[Embedder, SparseEmbedder]:
    return Embedder(), SparseEmbedder()


async def _create_project_with_corpus(
    vector_store: VectorStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    texts: list[str],
) -> uuid.UUID:
    async with database.AsyncSessionLocal() as session:
        team = Team(name=f"Retrieval Test Team {uuid.uuid4().hex[:8]}")
        session.add(team)
        await session.flush()

        project = Project(
            name="Retrieval Test Project",
            team_id=team.id,
            config={},
        )
        session.add(project)
        await session.flush()
        project_id = project.id

        for text in texts:
            doc = Document(
                project_id=project.id,
                filename="corpus.md",
                file_type=FileType.MARKDOWN.value,
                raw_bytes=text.encode("utf-8"),
            )
            session.add(doc)
            await session.flush()
            await _index_document_directly(
                vector_store,
                embedder,
                sparse_embedder,
                doc,
                project,
            )

        await session.commit()

    return project_id


async def _index_document_directly(
    vector_store: VectorStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    doc: Document,
    project: Project,
) -> None:
    parsed = parse_file(content=doc.raw_bytes, file_type=doc.file_type)
    chunks = chunk_text(
        text=parsed,
        strategy=ChunkingStrategy.NAIVE,
        chunk_size=64,
        overlap=8,
    )
    for chunk in chunks:
        chunk.metadata["filename"] = doc.filename

    dense_results = embedder.embed(chunks)
    sparse_results = sparse_embedder.embed(chunks)
    await vector_store.upsert(project.id, dense_results, doc.id, sparse_results)


async def _cleanup_project(project_id: uuid.UUID) -> None:
    async with database.AsyncSessionLocal() as session:
        await session.execute(delete(Document).where(Document.project_id == project_id))
        await session.execute(delete(Project).where(Project.id == project_id))
        result = await session.execute(
            select(Team.id).where(Team.name.like("Retrieval Test Team%"))
        )
        team_ids = list(result.scalars().all())
        await session.execute(delete(Team).where(Team.id.in_(team_ids)))
        await session.commit()


@pytest.mark.integration
@pytest.mark.slow
async def test_hybrid_retrieval_sparse_boosts_exact_token(
    vector_store: VectorStore,
    embedders: tuple[Embedder, SparseEmbedder],
) -> None:
    embedder, sparse_embedder = embedders
    project_id = await _create_project_with_corpus(
        vector_store,
        embedder,
        sparse_embedder,
        _CORPUS,
    )

    try:
        rare_token = "XJ9KL2MQ4ROTATION"

        hybrid_results = await retrieve(
            project_id=project_id,
            query_text=rare_token,
            accessible_ids=[project_id],
            vector_store=vector_store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            top_k=3,
        )
        hybrid_texts = [r.text for r in hybrid_results]
        assert any(rare_token in text for text in hybrid_texts), (
            "Exact-token query should rank the rare-token chunk in top-3 via BM25"
        )

        dense_only_hits = await vector_store.search(
            project_id,
            query_vector=embedder.embed_query(rare_token),
            top_k=3,
        )
        dense_only_texts = [hit.payload.get("text", "") for hit in dense_only_hits if hit.payload]
        assert not any(rare_token in text for text in dense_only_texts), (
            "Dense-only search should not rank the rare-token chunk in the top-3; "
            "otherwise the hybrid assertion does not prove sparse contribution"
        )

        paraphrase_results = await retrieve(
            project_id=project_id,
            query_text="vintage home computer firmware defect",
            accessible_ids=[project_id],
            vector_store=vector_store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            top_k=3,
        )
        paraphrase_texts = [r.text for r in paraphrase_results]
        assert any("ZX81 home computer firmware rotation" in text for text in paraphrase_texts), (
            "Paraphrase query should still retrieve the right chunk via dense semantics"
        )

        # Confirm the ingest path actually persisted sparse vectors to Qdrant.
        # A hybrid search with missing sparse vectors would still return dense-only
        # results and could hide a broken sparse embed/upsert path.
        qdrant_client = vector_store._client
        scroll_response = await qdrant_client.scroll(
            collection_name=collection_name(project_id),
            limit=100,
            with_vectors=True,
        )
        points = scroll_response[0]
        assert points, "Collection should contain indexed points"
        assert all("dense" in point.vector for point in points)
        assert all("bm25" in point.vector for point in points), (
            "Every upserted point must include the sparse 'bm25' vector"
        )
    finally:
        await vector_store.delete_collection(project_id)
        await _cleanup_project(project_id)


@pytest.mark.integration
@pytest.mark.slow
async def test_reindex_project_restores_collection_and_hybrid_search(
    vector_store: VectorStore,
    embedders: tuple[Embedder, SparseEmbedder],
) -> None:
    embedder, sparse_embedder = embedders
    project_id = await _create_project_with_corpus(
        vector_store,
        embedder,
        sparse_embedder,
        _REINDEX_CORPUS,
    )

    try:
        hits_before = await vector_store.search(
            project_id,
            query_vector=embedder.embed_query("budget review"),
            top_k=10,
        )
        point_count_before = len(hits_before)
        assert point_count_before > 0

        await vector_store.delete_collection(project_id)

        ctx: dict[str, object] = {
            "embedder": embedder,
            "sparse_embedder": sparse_embedder,
            "vector_store": vector_store,
        }
        with patch("app.workers.maintenance.AsyncSessionLocal", database.AsyncSessionLocal):
            await reindex_project(ctx, project_id)

        hits_after = await vector_store.search(
            project_id,
            query_vector=embedder.embed_query("budget review"),
            top_k=10,
        )
        assert len(hits_after) == point_count_before

        hybrid_after = await retrieve(
            project_id=project_id,
            query_text="ZX81-FIRMWARE-ROTATION",
            accessible_ids=[project_id],
            vector_store=vector_store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            top_k=3,
        )
        assert any("ZX81-FIRMWARE-ROTATION" in r.text for r in hybrid_after)
    finally:
        await vector_store.delete_collection(project_id)
        await _cleanup_project(project_id)


@pytest.mark.integration
@pytest.mark.slow
async def test_hybrid_retrieval_project_isolation(
    vector_store: VectorStore,
    embedders: tuple[Embedder, SparseEmbedder],
) -> None:
    embedder, sparse_embedder = embedders
    project_a_texts = [
        "Project Alpha discussed the ZX81-FIRMWARE-ROTATION bug.",
        "Project Alpha also planned the next sprint.",
    ]
    project_b_texts = [
        "Project Beta is unrelated and talks about marketing copy.",
        "Project Beta released a new feature flag system.",
    ]

    project_a_id = await _create_project_with_corpus(
        vector_store, embedder, sparse_embedder, project_a_texts
    )
    project_b_id = await _create_project_with_corpus(
        vector_store, embedder, sparse_embedder, project_b_texts
    )

    try:
        results = await retrieve(
            project_id=project_a_id,
            query_text="ZX81-FIRMWARE-ROTATION",
            accessible_ids=[project_a_id, project_b_id],
            vector_store=vector_store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            top_k=5,
        )
        assert len(results) > 0
        for chunk in results:
            assert chunk.project_id == project_a_id
            assert "Project Alpha" in chunk.text

        multi_results = await retrieve_multi(
            project_ids=[project_a_id, project_b_id],
            query_text="feature flag",
            accessible_ids=[project_a_id, project_b_id],
            vector_store=vector_store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            top_k=5,
        )
        for chunk in multi_results:
            assert chunk.project_id in {project_a_id, project_b_id}
    finally:
        await vector_store.delete_collection(project_a_id)
        await vector_store.delete_collection(project_b_id)
        await _cleanup_project(project_a_id)
        await _cleanup_project(project_b_id)
