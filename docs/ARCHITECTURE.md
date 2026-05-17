# ARCHITECTURE.md

> System design, component breakdown, data flows, and schema.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│   Next.js Frontend    │    REST API consumers                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTPS
┌────────────────────────────────▼────────────────────────────────┐
│                        API LAYER                                │
│                    FastAPI (Python 3.12)                        │
│   /auth  /projects  /documents  /query  /admin  /jobs           │
│                                                                 │
│   JWT Middleware → RBAC Dependencies → Access Scope Helper      │
└──────────┬────────────────────┬────────────────────────────────┘
           │                    │
     sync responses      async job queue
           │                    │
┌──────────▼──────┐    ┌────────▼────────┐
│   SERVICE LAYER │    │   ARQ WORKERS   │
│                 │    │                 │
│  AuthService    │    │  IngestJob      │
│  ProjectService │    │  CAGUpdateJob   │
│  QueryService   │    │  CAGRebuildJob  │
│  IngestService  │    │  WeeklyRebuild  │
│  CAGService     │    └────────┬────────┘
└──────────┬──────┘             │
           │                    │
┌──────────▼────────────────────▼────────────────────────────────┐
│                     PERSISTENCE LAYER                           │
│                                                                 │
│  PostgreSQL 16              Redis              Qdrant           │
│  ─────────────              ─────              ──────           │
│  users                      job queue          per-project      │
│  teams                      caching            collections      │
│  projects                                      (RAG chunks)     │
│  documents                                                      │
│  document_summaries                                             │
│  belief_states                                                  │
│  ingestion_jobs                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### API Layer (`/backend/app/api/`)
Thin route handlers only. No business logic. Responsibilities:
- Validate request via Pydantic schemas
- Authenticate via JWT middleware
- Check role via RBAC dependency
- Delegate to service layer
- Return response schema

### Service Layer (`/backend/app/services/`)
All business logic lives here. Services are plain async Python classes, injected via FastAPI's dependency system. Never import FastAPI's `Request` inside a service — services are framework-agnostic.

### Retrieval Layer (`/backend/app/retrieval/`)
The core of the project. Owns:
- CAG orientation (belief state injection)
- RAG retrieval (hybrid dense + BM25 against Qdrant)
- Query routing logic
- Response construction with source attribution
- Access scope enforcement

### Ingestion Pipeline (`/backend/app/ingestion/`)
Owns the pipeline from raw file to stored embeddings:
- File parsing (markdown, txt, PDF via PyMuPDF)
- Chunking strategies (naive implemented, contextual/late stubbed)
- Embedding via FastEmbed (local, no API key)
- Qdrant upsert with per-project collection isolation
- Summary generation via LLM (structured JSON, never raises)
- Document summaries stored as immutable event log in Postgres

### CAG Layer (`/backend/app/services/cag.py`)
Owns belief state lifecycle:
- Incremental update (rolling window synthesis)
- Full rebuild (Map-Reduce over event log)
- Threshold checking (trigger rebuild at N docs)
- Versioned belief state storage

---

## Ingestion Data Flow

```
User uploads file
       │
       ▼
POST /projects/{id}/documents
       │
       ▼
DocumentService.upload()
  → validates file type (.md, .txt, .pdf) and size
  → stores raw document in Postgres (documents table)
  → creates IngestionJob (status: pending)
  → queues ARQ job via arq_pool.enqueue_job()
  → returns 202 with job_id
       │
       ▼ (async, in ARQ worker process)
ingest_document(ctx, job_id, document_id, project_id)
  │
  ├─ status → RUNNING (commit)
  │
  ├─ parse_file(content, file_type) → raw text
  │    markdown: direct, txt: UTF-8 decode, PDF: PyMuPDF
  │
  ├─ chunk_text(text, strategy=naive, size=512, overlap=64) → chunks[]
  │    (contextual and late strategies stubbed)
  │
  ├─ embedder.embed(chunks) → EmbeddingResult[]
  │    FastEmbed, local model, lru_cache singleton
  │
  ├─ vector_store.upsert(project_id, results, document_id)
  │    collection: "project_{project_id}"
  │    payload: document_id, project_id, chunk_index, text, filename, created_at
  │
  ├─ summarize_document(text, filename, model) → dict
  │    LiteLLM call → structured JSON summary
  │    never raises — returns fallback on LLM failure or invalid JSON
  │    stored in document_summaries (immutable event log)
  │
  ├─ check CAG rebuild threshold
  │    if doc_count % threshold == 0:
  │      log (rebuild job not yet implemented)
  │
  ├─ status → COMPLETE, completed_at = now (commit)
  │
  └─ on any exception:
       status → FAILED, error_message = str(exc) (commit)
       re-raise for ARQ retry
```

---

## Query Data Flow

```
User submits query
       │
       ▼
POST /projects/{id}/query  (or POST /query for admin cross-project)
       │
       ▼
Access check
  → get_accessible_project_ids(user)
  → verify requested project_id is in accessible set
  → if admin cross-project: use full accessible set
       │
       ▼
QueryService.query()
  │
  ├─ 1. CAG ORIENTATION
  │    load belief_state for project (from Postgres)
  │    construct system context:
  │      "You are answering questions about [project].
  │       Here is what is known about this project: {belief_state}"
  │
  ├─ 2. RAG RETRIEVAL
  │    hybrid retrieval against project Qdrant collection:
  │      dense search (FastEmbed query vector)
  │      + BM25 sparse search
  │    re-rank by combined score
  │    take top-k chunks
  │
  ├─ 3. PROMPT CONSTRUCTION
  │    system: CAG belief state context
  │    user: retrieved chunks + original query
  │
  ├─ 4. LLM GENERATION
  │    LiteLLM call (query model from project config)
  │
  └─ 5. RESPONSE
       answer text
       + source_chunks[] (document name, chunk text, score)
       + belief_state_version (so client knows how fresh CAG is)
```

---

## CAG Belief State Lifecycle

```
                    document_summaries (event log)
                    ──────────────────────────────
doc 1 ingested  →   summary_1  (immutable, never deleted)
doc 2 ingested  →   summary_2
...
doc N ingested  →   summary_N
                              │
                    ┌─────────▼──────────────┐
                    │  Threshold check        │
                    │  N % threshold == 0?    │
                    └─────────┬──────────────┘
                              │ yes
                    ┌─────────▼──────────────┐
                    │  CAGUpdateJob           │
                    │                         │
                    │  input:                 │
                    │    current belief_state │
                    │    last N summaries     │
                    │    (rolling window)     │
                    │                         │
                    │  LLM synthesizes        │
                    │  new belief_state       │
                    └─────────┬──────────────┘
                              │
                    ┌─────────▼──────────────┐
                    │  belief_states table    │
                    │  new versioned row      │
                    │  (old versions kept)    │
                    └─────────────────────────┘

  Weekly / manual rebuild:
    Map-Reduce over ALL document_summaries for project
    → fresh belief_state from scratch
    → guarantees no drift accumulates indefinitely
```

---

## Belief State Schema

```json
{
  "project_id": "uuid",
  "version": 14,
  "generated_at": "ISO8601",
  "source_summary_ids": ["uuid", "uuid", "..."],
  "decisions": [
    {
      "description": "Chose Qdrant over Pinecone for self-hostability",
      "approximate_date": "2025-04-01",
      "summary_id_ref": "uuid"
    }
  ],
  "open_items": [
    {
      "description": "Auth flow for SSO unresolved",
      "first_seen_summary_id": "uuid"
    }
  ],
  "key_people": [
    { "name": "Alice", "role": "backend lead" }
  ],
  "recurring_themes": ["performance", "API versioning"],
  "project_summary": "Two-sentence plain language summary of what this project is and where it is.",
  "last_rebuild_type": "incremental | full"
}
```

---

## PostgreSQL Schema

```sql
-- Users and auth
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  hashed_password TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'member',  -- member | admin | super_admin
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Teams
CREATE TABLE teams (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE team_members (
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, team_id)
);

-- Projects
CREATE TABLE projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  description TEXT,
  team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
  config JSONB NOT NULL DEFAULT '{}',  -- ingestion config
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Documents (raw, immutable after creation)
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  file_type TEXT NOT NULL,
  raw_content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Document summaries (immutable event log)
CREATE TABLE document_summaries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  summary JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Belief states (derived, versioned, never mutated)
CREATE TABLE belief_states (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  version INTEGER NOT NULL,
  state JSONB NOT NULL,
  rebuild_type TEXT NOT NULL,  -- incremental | full
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Ingestion jobs
CREATE TABLE ingestion_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  document_id UUID REFERENCES documents(id),
  status TEXT NOT NULL DEFAULT 'pending',  -- pending | running | complete | failed
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ
);
```

---

## Qdrant Collection Naming

Each project gets one Qdrant collection: `project_{project_id}`.

Example: project ID `a3f2b1c4-...` → collection `project_a3f2b1c4-...`

Chunks stored with payload:
```json
{
  "document_id": "uuid",
  "project_id": "uuid",
  "chunk_index": 3,
  "text": "the chunk text",
  "filename": "meeting-2025-04-01.md",
  "created_at": "ISO8601"
}
```

---

## Access Control Architecture

Access is enforced at **two layers**:

**Layer 1 — API layer (FastAPI dependency)**
```python
# Every protected route declares this dependency
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    ...

async def require_role(required_role: Role):
    ...
```

**Layer 2 — Retrieval layer (access scope)**
```python
# Before any Qdrant query, scope is resolved
async def get_accessible_project_ids(user: User) -> list[UUID]:
    # member: returns only projects where user's team is assigned
    # admin/super_admin: returns all project IDs
    ...

# Retrieval always receives scoped list
async def query(user_query: str, project_id: UUID, user: User):
    accessible = await get_accessible_project_ids(user)
    if project_id not in accessible:
        raise AccessDeniedError()
    # proceed with retrieval
```

Even if the API layer is bypassed, the retrieval layer will deny access to out-of-scope projects. This is enforced by keeping `get_accessible_project_ids` as a required parameter in every retrieval function signature — it cannot be called without it.

---

## LLM Abstraction

All LLM calls go through `/backend/app/core/llm.py`:

```python
async def llm_call(
    messages: list[dict],
    model: str,           # from project config, never hardcoded
    max_tokens: int = 1000,
    response_format: dict | None = None,
) -> str:
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

Model names come from project config or system defaults in `.env`. Never hardcoded in service or retrieval code.
