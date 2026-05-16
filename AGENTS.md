# AGENTS.md

> Instructions for AI coding agents (Claude Code, Cursor, Copilot, etc.) working on this codebase.
> Read this file fully before making any changes.

---

## What This Project Is

A self-hostable hybrid RAG/CAG knowledge platform. Each project gets its own belief state (CAG layer — compressed, structured project understanding) and retrieval layer (RAG — specific chunk retrieval). Teams are access-scoped. The system is modular and pluggable.

Read [`/docs/ARCHITECTURE.md`](/docs/ARCHITECTURE.md) before touching core retrieval or ingestion logic.
Read [`/docs/RULES.md`](/docs/RULES.md) before writing any code.

---

## How to Run the Project

### Full stack (Docker)
```bash
docker compose -f docker-compose.dev.yml up
```

### Backend only (for backend development)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

### Run the ARQ worker
```bash
cd backend
arq workers.main.WorkerSettings
```

### Frontend only
```bash
cd frontend
npm install
npm run dev
```

---

## How to Run Tests

Activate the virtual environment first:
```bash
source .venv/bin/activate
```

Run from the project root (`/home/sandro/Documents/BaseProject`), not from `backend/`. The `pythonpath = ["backend"]` in `pyproject.toml` handles the import path.

```bash
# All tests (unit only by default — skips integration/e2e)
pytest --import-mode=importlib -m "not integration and not e2e"

# Integration tests only (requires Postgres, Redis, Qdrant running)
pytest --import-mode=importlib -m integration

# All tests including integration
pytest --import-mode=importlib

# Specific module
pytest --import-mode=importlib backend/tests/ingestion/test_parser.py

# With coverage
pytest --import-mode=importlib --cov=app --cov-report=term-missing

# Verbose
pytest --import-mode=importlib -v
```

Prerequisites for integration tests:
```bash
docker compose -f docker-compose.dev.yml up postgres redis qdrant -d
```

---

## Directory Structure

```
/backend
  /app
    /api            → FastAPI route handlers (thin — no business logic here)
    /core           → Config, database session, security utilities
    /models         → SQLAlchemy ORM models
    /schemas        → Pydantic request/response schemas
    /services       → Business logic (ingestion, CAG, query, auth)
    /retrieval      → RAG/CAG retrieval logic — the core of this project
    /ingestion      → Chunking, embedding, ingestion pipeline
    /workers        → ARQ job definitions
    main.py         → FastAPI app factory
  /alembic          → Database migrations
  /tests            → Pytest test suite (mirrors /app structure)

/frontend
  /app              → Next.js App Router pages
  /components       → Reusable UI components
  /lib              → API client, auth helpers, types
  /hooks            → Custom React hooks

/docs               → Architecture, stack, rules, roadmap
/tasks              → Feature files and task breakdowns
```

---

## Key Concepts to Understand Before Coding

### Belief State (CAG Layer)
Each project has a `belief_state` — a structured JSON document summarizing the project's decisions, open items, people, and themes. It is a **derived artifact** generated from the immutable `document_summaries` event log. Never mutate a belief state directly. Always regenerate it from the event log. See `/docs/ARCHITECTURE.md#cag-layer`.

### Access Scoping
Access control is enforced **at the retrieval layer**, not just the API layer. The function `get_accessible_project_ids(user_id)` returns the scoped list of project IDs a user can access. Every retrieval call must receive this scoped list. Do not bypass this. See `/backend/app/core/access.py`.

### Ingestion Pipeline
Ingestion is always async via ARQ. Never run ingestion synchronously in an API handler. The pipeline: parse → chunk → (optionally) contextualize → embed → upsert to Qdrant + store summary in Postgres → trigger CAG update check.

### LLM Calls
All LLM calls go through LiteLLM via the wrapper in `/backend/app/core/llm.py`. Never call OpenAI, Anthropic, or Ollama SDKs directly. This is what makes the platform LLM-agnostic.

---

## Patterns to Follow

- **Services own business logic.** Route handlers call services, never implement logic themselves.
- **Schemas validate at the boundary.** All API input/output goes through Pydantic schemas.
- **Dependencies for auth and access.** Use FastAPI dependency injection for `get_current_user`, `require_role`, `get_accessible_projects`.
- **Async everywhere in backend.** All DB calls, LLM calls, and Qdrant calls are async.
- **Tests mirror app structure.** `tests/services/test_ingestion.py` tests `app/services/ingestion.py`.
- **One Alembic migration per PR** that touches the schema.

## Patterns to Avoid

- Do not put business logic in route handlers.
- Do not call LLM providers directly — always use the LiteLLM wrapper.
- Do not mutate belief states directly — always regenerate from the event log.
- Do not query Qdrant without passing a scoped collection list.
- Do not hardcode model names — read from project config.
- Do not skip tests. Every new function needs a test.
- Do not use `print()` for logging — use the structured logger in `/backend/app/core/logging.py`.

---

## Environment Variables

See `.env.example` for all required variables. Key ones:

```
DATABASE_URL          Postgres connection string
REDIS_URL             Redis connection string
QDRANT_URL            Qdrant connection string
LITELLM_DEFAULT_MODEL Default LLM model string (e.g. gpt-4o-mini)
JWT_SECRET            Secret for signing JWTs
```

---

## Database Migrations

```bash
# Create a new migration after changing a model
alembic revision --autogenerate -m "describe what changed"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

Always review autogenerated migrations before committing.

---

## Common Mistakes

- **Forgetting to scope Qdrant queries** — the collection name is always `project_{project_id}`. Never query a collection without first verifying the user has access to that project ID.
- **Mutating belief state JSON in place** — don't. Regenerate from event log.
- **Running ingestion in the request handler** — always queue an ARQ job.
- **Using sync SQLAlchemy** — this project uses `asyncpg` + async SQLAlchemy. Use `async with session` and `await session.execute(...)`.
