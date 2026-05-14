# STACK.md

> Every technology choice with rationale and what was considered and rejected.

---

## Backend — Python 3.12 + FastAPI

**Why Python:** The entire LLM/RAG ecosystem is Python-first. LiteLLM, LlamaIndex, FastEmbed, Qdrant SDK, ARQ — all Python-native. Using any other language means fighting the ecosystem from day one.

**Why FastAPI:** Async-native, automatic OpenAPI docs, excellent dependency injection system (critical for RBAC and access scoping), Pydantic integration is first-class, strong community.

**Considered and rejected:**
- Django — too opinionated, ORM not async-native, overkill for an API
- Flask — no async support, no dependency injection, would need too many additions
- Node/Express — loses the Python LLM ecosystem entirely

---

## Vector Store — Qdrant

**Why Qdrant:**
- Self-hostable with a single Docker image — critical for a self-hosted open-source tool
- Native collection isolation: each project gets its own collection, no metadata filtering hacks needed
- Hybrid search (dense + sparse BM25) built in — exactly the retrieval pattern this project needs
- Strong async Python SDK
- Active development, good documentation

**Considered and rejected:**
- Pinecone — SaaS only, not self-hostable, against core goals
- Chroma — good for prototyping, not production-grade at scale, weaker hybrid search
- Weaviate — self-hostable but heavier operationally, more complex configuration
- pgvector — tempting (reduce stack complexity), but hybrid search is significantly weaker and performance degrades at scale

---

## Embeddings — FastEmbed

**Why FastEmbed:**
- Runs entirely locally — no API key, no external dependency, no cost per embedding
- Fast enough for ingestion workloads
- Qdrant-native (same team), integrates cleanly
- Falls back gracefully to OpenAI embeddings if configured

**Considered and rejected:**
- OpenAI embeddings — cost at scale, external dependency, not self-contained
- sentence-transformers directly — more setup, FastEmbed wraps this more cleanly
- Cohere embeddings — SaaS dependency

---

## LLM Interface — LiteLLM

**Why LiteLLM:**
- Single unified interface for 100+ LLM providers — OpenAI, Anthropic, Ollama, Cohere, etc.
- Async support (`litellm.acompletion`)
- Drop-in model switching via config string — `gpt-4o`, `claude-sonnet-4-5`, `ollama/llama3`
- This is the single most important piece for making the platform LLM-agnostic

**Considered and rejected:**
- Calling provider SDKs directly — locks users to one provider, defeats the open-source self-hosting goal
- LangChain LLM wrappers — heavier abstraction, more magic, harder to debug

---

## Database — PostgreSQL 16

**Why PostgreSQL:**
- The immutable event log and versioned belief states need ACID guarantees
- JSONB for belief state and project config — flexible schema without a separate document store
- Mature, battle-tested, excellent async support via asyncpg
- Alembic migrations work first-class with SQLAlchemy async

**Considered and rejected:**
- SQLite — not suitable for concurrent async writes, not production-ready for multi-user
- MongoDB — JSONB in Postgres gives the document flexibility without adding another service
- MySQL — weaker JSONB support, less ecosystem tooling

---

## Task Queue — ARQ + Redis

**Why ARQ:**
- Built for async Python — integrates natively with FastAPI's async model
- Uses Redis (already in stack for caching) — no extra service
- Simple API, minimal boilerplate
- Supports cron scheduling (weekly CAG rebuild jobs)
- Significantly leaner than Celery for this use case

**Why not Celery:**
- Celery requires its own configuration layer (flower, beat, worker nodes)
- Not async-native — requires workarounds with FastAPI
- Heavy Docker Compose footprint for what are essentially simple background jobs
- Overkill for the job types this project needs

**Considered and rejected:**
- Celery + RabbitMQ — even heavier
- FastAPI BackgroundTasks — no persistence, no retry, no scheduling, jobs die with the process
- RQ (Redis Queue) — good but sync-first, ARQ is the async equivalent

---

## Frontend — Next.js 16

**Why Next.js:**
- App Router is well-suited for the mix of static and dynamic pages this project has
- TypeScript first-class
- Good ecosystem for building clean admin-style dashboards
- API routes useful for proxying auth tokens without exposing them to the browser
- Wide familiarity — lowers the barrier for contributors

**Considered and rejected:**
- SvelteKit — smaller contributor pool, less familiar to most engineers
- Remix — good choice but Next.js has wider reach for an open-source project
- Plain React SPA — no SSR benefits, more configuration

---

## Auth — JWT + Custom RBAC

**Why custom JWT RBAC:**
- Simple to understand, easy to extend, no external auth service dependency
- Role is embedded in the JWT claim — RBAC checks are pure functions with no DB call
- Access scoping (which projects a user can access) is a DB query kept in a dedicated helper — easy to audit

**Considered and rejected:**
- Auth0 / Clerk — SaaS dependency, against self-hosting goals
- Keycloak — self-hostable but significant operational overhead for v1
- Supabase Auth — couples the project to Supabase's stack

SSO / OAuth is v2 scope. The auth module is designed to be swappable — the RBAC dependency layer is cleanly separated so SSO can be dropped in later.

---

## Containerization — Docker Compose

**Why Docker Compose:**
- Single command (`docker compose up`) to bring up the full stack
- Standard, universally understood
- Easy to extend with `.override` files for dev vs prod configs
- No Kubernetes overhead for a self-hosted internal tool

**Considered and rejected:**
- Kubernetes — massive overkill for a self-hosted internal tool, raises the ops bar too high
- Bare metal scripts — not reproducible, not portable

---

## Testing — Pytest + HTTPX

**Why Pytest:**
- Industry standard for Python
- `pytest-asyncio` handles async test functions cleanly
- HTTPX's `AsyncClient` works directly with FastAPI's `app` object — no server needed
- Fixture system is excellent for setting up DB state, auth tokens, projects

**Test strategy:**
- Unit tests for services and retrieval logic (fast, no external dependencies, use mocks for LLM and Qdrant)
- Integration tests for API endpoints (use real Postgres and Qdrant via Docker in CI)
- E2E tests for critical flows (Playwright on frontend in CI)

---

## Linting and Formatting — Ruff + pre-commit

**Why Ruff:**
- Replaces both flake8 and black in one tool
- Extremely fast (written in Rust)
- Configurable via `pyproject.toml` — single config file
- Handles import sorting (replaces isort)

**Pre-commit hooks run:**
- `ruff check` — linting
- `ruff format` — formatting
- `mypy` — type checking
- `pytest` — not on pre-commit (too slow), but required to pass in CI
