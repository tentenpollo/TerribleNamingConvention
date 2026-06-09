# Phase 2 — Ingestion Pipeline: Implementation Summary

> Built: May 17, 2026
> Commits: 8
> Files changed: 31 (+2,731 / -16)
> Tests: 124 unit tests passing, 21 integration tests

---

## What Was Built

Phase 2 implements the full ingestion pipeline: file upload → parse → chunk → embed → store in Qdrant + Postgres. Everything runs async via ARQ workers. The pipeline is modular and pluggable.

---

## Feature Breakdown

### 2.1 — Document and IngestionJob Models + Migrations
**Commit:** `132c9ee feat(2.1)`

- `Document` model — stores raw file content, filename, file type, project association
- `IngestionJob` model — tracks pipeline status (pending/running/complete/failed), error messages, timestamps
- `DocumentSummary` model — immutable event log entry, stores structured JSON summary per document
- Alembic migrations 005 (documents), 006 (document_summaries), 007 (ingestion_jobs)
- Pydantic schemas for API request/response validation

### 2.2 — File Parser
**Commit:** `b31c77e feat(2.2)`

- `backend/app/ingestion/parser.py` — pure text extraction from raw bytes
- Supports: markdown (direct), txt (UTF-8 decode), PDF (via PyMuPDF)
- `UnsupportedFileTypeError` for unknown file types
- Whitespace normalization on output
- 7 unit tests covering all formats and error cases

### 2.3 — Naive Chunker + FastEmbed Embedder
**Commit:** `266f7da feat(2.3)`

- `backend/app/ingestion/chunker.py`
  - `ChunkingStrategy` enum: naive (implemented), contextual (stub), late (stub)
  - Word-based sliding window chunker with configurable size (default 512) and overlap (default 64)
  - Returns `Chunk` objects with text, index, and metadata
  - 8 unit tests

- `backend/app/ingestion/embedder.py`
  - `Embedder` class wrapping FastEmbed (local, no API key needed)
  - `lru_cache` singleton for model reuse across requests
  - Batch embedding and single query embedding
  - Returns `EmbeddingResult` objects pairing chunks with vectors
  - 5 unit tests + 2 slow integration tests

- `get_embedder()` FastAPI dependency injection

### 2.4 — Qdrant Client Wrapper + VectorStore
**Commit:** `9141801 feat(2.4)`

- `backend/app/ingestion/vector_store.py`
  - `VectorStore` class with `ensure_collection`, `upsert`, `search`, `delete_collection`
  - Collection naming: `project_{project_id}` — one collection per project, provably isolated
  - Upsert builds `PointStruct` with payload: document_id, project_id, chunk_index, text, filename, created_at
  - Search uses `query_points` API, returns empty list if collection doesn't exist
  - All Qdrant exceptions wrapped in `QdrantError` with context
  - 6 integration tests: create, idempotent, roundtrip, payload, delete, project isolation
  - 13 unit tests added later (see 2.4 test follow-up)

- `backend/app/core/qdrant.py` — `AsyncQdrantClient` singleton via `get_qdrant_client()`
- `get_vector_store()` FastAPI dependency as module-level cached singleton

### 2.5 — ARQ Worker Setup + IngestJob Pipeline
**Commit:** `7ccdde0 feat(2.5)`

- `backend/app/workers/settings.py`
  - `WorkerSettings` class — ARQ worker configuration
  - `on_startup` initializes `Embedder` and `VectorStore` once, shares via `ctx`
  - `on_shutdown` closes Qdrant client cleanly
  - Reuses expensive model instances across jobs

- `backend/app/workers/ingest.py`
  - `ingest_document()` orchestrates: parse → chunk (naive, 512/64) → embed → upsert to Qdrant
  - Immediate status commits: PENDING → RUNNING → COMPLETE
  - Failed jobs write `error_message` to DB before re-raising for ARQ retry
  - Generates document summary, stores `DocumentSummary`, checks CAG rebuild threshold
  - 4 unit tests + 1 full integration test (real DB + Qdrant)

### 2.6 — Document Upload Endpoint + Job Status Endpoint
**Commit:** `bda456c feat(2.6)`

- `backend/app/services/document.py` — `DocumentService`
  - `upload()` — validates access scope, stores document, creates ingestion job, queues ARQ job, returns 202
  - `get_job()` — retrieves ingestion job by ID
  - `list_documents()` — lists documents for a project, access-scoped

- `backend/app/api/documents.py`
  - `POST /projects/{id}/documents` — file upload, returns ingestion job
  - `GET /projects/{id}/documents` — list documents for a project
  - `GET /jobs/{id}` — poll ingestion job status
  - File type validation (.md, .txt, .pdf), size limit enforcement

- ARQ pool initialized in FastAPI lifespan, accessed via `request.app.state.arq_pool`
- `get_document_service()` dependency injection
- 10 API integration tests + 8 service unit tests
- Integration test rules documented in AGENTS.md (NullPool pattern, session boundaries, cleanup)

### 2.7 — Document Summary Generation + Postgres Event Log
**Commit:** `bdaf9ce feat(2.7)` + `0876d56 refactor(2.7)`

- `backend/app/core/llm.py` — `llm_call()` wrapper
  - LiteLLM `acompletion` with structured logging
  - `LLMError` exception for failures
  - Model name from config, never hardcoded

- `backend/app/ingestion/summarizer.py` — `summarize_document()`
  - Generates structured JSON summary via LLM call
  - Schema: summary, key_points, technical_concepts, architectural_components, decisions, action_items, entities, topics, important_relationships, document_type, confidence
  - Never raises — returns fallback dict on LLM failure or invalid JSON
  - Fallback includes `raw_text_fallback: true` flag
  - 5 unit tests

- Ingest worker updated to:
  - Call `summarize_document()` after Qdrant upsert
  - Store `DocumentSummary` row in Postgres
  - Check CAG rebuild threshold (logs when reached, rebuild not yet implemented)

---

## Test Coverage

| Module | Coverage |
|---|---|
| `app/ingestion/chunker.py` | 97% |
| `app/ingestion/embedder.py` | 93% |
| `app/ingestion/parser.py` | 100% |
| `app/ingestion/summarizer.py` | 100% |
| `app/ingestion/vector_store.py` | 98% |
| `app/workers/ingest.py` | 95% |
| **Combined** | **87%** |

---

## Verification Results

| Check | Result |
|---|---|
| Unit tests | 124 passed |
| Integration tests | 21 passed |
| ruff check | Clean |
| ruff format | Clean |
| mypy | No issues |
| Docker build | Clean from scratch |
| Health check | OK |
| End-to-end smoke test | Upload → complete in ~1s |
| Qdrant collection isolation | Verified — two projects, separate collections, no cross-contamination |
| LLM failure resilience | Verified — ingestion completes with fallback summary |
| Failed job error handling | Verified — error_message written, status set to failed |

---

## Docker Compose Changes

Added `worker` service to `docker-compose.dev.yml`:
- Runs `arq app.workers.settings.WorkerSettings`
- Shares volumes with backend for hot reload
- Depends on postgres, redis, qdrant (all healthy)

---

## What's Not Yet Implemented (Stubbed)

- Contextual chunking strategy — stubbed, not implemented
- Late chunking strategy — stubbed, not implemented
- CAG rebuild job — threshold check logs but rebuild not yet implemented
- Belief states table — schema defined in architecture but not yet migrated
- ---

## Known Defects (discovered post-phase, June 2026)

- **PDF upload broken in practice** — `DocumentService.upload()` UTF-8 decodes raw bytes before storage; real binary PDFs fail at upload. Parser unit tests pass because they call `parse_file` directly with bytes. Fix tracked in ROADMAP Phase 3.0.
- **Ingestion is not idempotent under ARQ retry** — random Qdrant point IDs and no uniqueness constraint on document_summaries mean a retried job duplicates vectors and event-log rows. Fix tracked in Phase 3.0.
- **Qdrant collections created in Phase 2 are dense-only** — incompatible with Phase 4 hybrid (sparse must be declared at collection creation). Collections need schema v2 + re-index. Fix tracked in Phase 3.0.
