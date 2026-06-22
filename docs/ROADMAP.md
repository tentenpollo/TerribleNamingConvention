# ROADMAP.md

> Phase-by-phase plan for v1. v2 backlog at the bottom.

---

## Status Legend

| Symbol | Meaning |
|---|---|
| ⬜ | Not started |
| 🟡 | In progress |
| ✅ | Complete |

---

## Phase 0 — Project Skeleton
**Goal: One command brings up the full dev environment. Nothing works yet but everything is wired.**

| Task | Status |
|---|---|
| Monorepo directory structure | ✅ |
| Docker Compose (FastAPI, Postgres, Redis, Qdrant, Next.js) | ✅ |
| docker-compose.dev.yml with hot reload | ✅ |
| Alembic initialized, first migration (users table) | ✅ |
| Pytest configured with pytest-asyncio + HTTPX | ✅ |
| Ruff linting + formatting configured in pyproject.toml | ✅ |
| mypy configured | ✅ |
| pre-commit hooks wired | ✅ |
| .env.example with all required vars | ✅ |
| PR template | ✅ |
| Health check endpoint GET /health | ✅ |
| CI pipeline (GitHub Actions) — lint + test on PR | ✅ |

**Done when:** `docker compose -f docker-compose.dev.yml up` runs clean, health check returns 200, all tests pass.

---

## Phase 1 — Auth and Access Control
**Goal: Users exist, roles work, JWT is enforced, access is scoped.**

| Task | Status |
|---|---|
| Postgres schema: users, teams, team_members, projects | ✅ |
| Alembic migrations for all four tables | ✅ |
| User registration endpoint POST /auth/register | ✅ |
| User login endpoint POST /auth/login → JWT | ✅ |
| GET /auth/me endpoint | ✅ |
| JWT middleware — extract user + role from token | ✅ |
| RBAC dependency — require_role(Role.X) composable | ✅ |
| Access scope helper — get_accessible_project_ids(user) | ✅ |
| Team CRUD endpoints (admin only) | ✅ |
| Project CRUD endpoints | ✅ (delete added in Phase 3.0 hardening) |
| Team member assignment | ✅ |
| Tests: register, login, token validation | ✅ |
| Tests: role enforcement — member cannot hit admin routes | ✅ |
| Tests: access scope returns correct project IDs per user | ✅ |

**Done when:** A member cannot access another team's projects at the API level.

---

## Phase 2 — Ingestion Pipeline
**Goal: Drop a file in, it gets chunked, embedded, and stored.**

| Task | Status |
|---|---|
| Document upload endpoint POST /projects/{id}/documents | ✅ |
| File parsing — markdown, txt, PDF (via pymupdf) | ✅ |
| Naive chunking strategy | ✅ |
| Contextual chunking strategy (LLM annotation per chunk) | ⬜ |
| Late chunking strategy | ⬜ |
| Per-project ingestion config (chunking_strategy, context_model, query_model, cag_rebuild_threshold) | ✅ |
| FastEmbed embedding | ✅ |
| Qdrant upsert — collection per project (project_{id}) | ✅ |
| ARQ IngestJob — full async pipeline | ✅ |
| Raw document stored in Postgres (documents table) | ✅ |
| Per-document summary stored in Postgres (document_summaries table — immutable) | ✅ |
| Ingestion job status endpoint GET /jobs/{id} | ✅ |
| Tests: upload creates job, job completes, chunks in Qdrant | ✅ |
| Tests: summaries stored in Postgres | ✅ |
| Tests: two projects have isolated Qdrant collections | ✅ |

**Done when:** Two projects each have documents ingested into fully isolated Qdrant collections. ✅

---

## Phase 3.0 — Phase 2 Ingestion Hardening
**Goal: Make the completed ingestion pipeline safe to retry and ready for CAG/query work.**

| Task | Status |
|---|---|
| Store raw document bytes instead of UTF-8 text | ✅ |
| Deterministic Qdrant chunk point IDs | ✅ |
| Idempotent document summaries | ✅ |
| Event-log index for project summary counts | ✅ |
| Qdrant collection schema v2 with named dense and sparse vectors | ✅ |
| Offload CPU-bound embedding from the ARQ event loop | ✅ |
| Stale pending ingestion job sweeper | ✅ |
| Delete project Qdrant collection on project delete | ✅ |
| Remove vestigial root workers directory | ✅ |
| Default chunking strategy reflects implemented strategies | ✅ |

**Done when:** Phase 2 ingestion retries do not duplicate vectors or summaries, and stale pending jobs can be recovered.

---

## Phase 3 — CAG Layer
**Goal: Each project has a living structured belief state. It updates. It rebuilds cleanly.**

| Task | Status |
|---|---|
| belief_states table and Alembic migration | ✅ |
| Belief state JSON schema defined and validated (Pydantic) | ✅ |
| Initial belief state generation on first ingest | ✅ |
| Incremental update — rolling window of recent summaries + current state | ✅ |
| Threshold check after each ingest job | ✅ |
| ARQ CAGUpdateJob | ✅ |
| ARQ CAGRebuildJob — full Map-Reduce over all summaries | ✅ |
| Weekly cron rebuild via ARQ scheduler | ✅ |
| Manual rebuild endpoint POST /projects/{id}/cag/rebuild (admin only) | ✅ |
| Belief state read endpoint GET /projects/{id}/cag | ✅ |
| Tests: belief state created on first ingest | ✅ |
| Tests: incremental update produces valid state shape | ✅ |
| Tests: full rebuild from scratch matches incremental state | ✅ |
| Tests: old versions preserved in DB after rebuild | ✅ |
| Tests: manual rebuild is admin-only | ✅ |

**Done when:** A project's belief state accurately reflects its ingested documents and can be rebuilt cleanly from the event log.

---

## Phase 4 — Query Layer
**Goal: Users can query their projects and get grounded, oriented answers.**

| Task | Status |
|---|---|
| Project query endpoint POST /projects/{id}/query | ✅ |
| CAG orientation — belief state injected as system context | ✅ |
| RAG retrieval — hybrid dense + BM25 against project collection | ✅ |
| Chunk re-ranking by combined score | ✅ |
| Prompt construction — CAG system context + retrieved chunks + query | ✅ |
| LiteLLM call with project's configured query model | ✅ |
| Response with source chunk attribution | ✅ |
| Access enforcement at retrieval layer | ✅ |
| Admin cross-project query endpoint POST /query | ✅ |
| Prompt-injection containment for belief-state and chunk content | ✅ |
| Per-user sliding-window rate limiting over Redis | ✅ |
| LLM timeout and graceful 503 retryable failure | ✅ |
| Token/cost telemetry via structured query_completed logs | ✅ |
| Tests: member query scoped to their project only | ✅ |
| Tests: admin query spans accessible projects | ✅ |
| Tests: belief state injected into context | ✅ |
| Tests: source attribution returned in response | ✅ |
| Tests: member cannot access admin cross-project endpoint | ✅ |
| Tests: red-team structural containment of stored prompt injections | ✅ |
| Tests: rate limiter allows/rejects and returns Retry-After | ✅ |
| Tests: 503 retryable handler | ✅ |
| Tests: log privacy (no question/chunk/answer text) | ✅ |

**Done when:** A member can ask "what did we decide about X?" and get a grounded answer with sources.

---

## Phase 5 — Frontend
**Goal: The whole thing is usable without touching the API directly.**

| Task | Status |
|---|---|
| Next.js project setup with TypeScript | ⬜ |
| Typed API client (fetch wrapper against FastAPI) | ⬜ |
| Auth — JWT in httpOnly cookie, login/register pages | ⬜ |
| Project list page (scoped to user's teams) | ⬜ |
| Project page — query interface | ⬜ |
| Project page — document upload with job status polling | ⬜ |
| Project page — document list | ⬜ |
| CAG belief state viewer (read-only, shows current project understanding) | ⬜ |
| Admin dashboard — all projects overview | ⬜ |
| Admin dashboard — cross-project query interface | ⬜ |
| Admin dashboard — team and member management | ⬜ |
| Super admin panel — user management, system config | ⬜ |
| Playwright E2E: login → upload doc → query → verify response | ⬜ |
| Playwright E2E: admin cross-project query scoping | ⬜ |

**Done when:** A non-technical user can onboard, upload meeting notes, and query them through the UI without touching the API.

---

## Phase 6 — Hardening and Open Source Prep
**Goal: Anyone can clone this, run it, and understand it.**

| Task | Status |
|---|---|
| README.md polished — what, why, quick start, config | ⬜ |
| .env.example fully documented | ⬜ |
| docker-compose.yml production-ready | ⬜ |
| CONTRIBUTING.md — branch naming, PR process, adapter interface | ⬜ |
| Ingestion adapter abstract base class for community extensions | ⬜ |
| API documentation polished — descriptions on every endpoint | ⬜ |
| Input validation pass — all endpoints hardened | ⬜ |
| Rate limiting on auth endpoints | ⬜ |
| Secrets audit — no hardcoded values, nothing sensitive in logs | ⬜ |
| Locust load test script — baseline query and ingestion performance | ⬜ |
| Full test suite passes in CI | ⬜ |
| Docker build clean from scratch | ⬜ |

**Done when:** The repo is something you'd be proud to link in a job application or post publicly on GitHub.

---

## Rough Timeline

| Phase | Estimate (solo) |
|---|---|
| 0 — Skeleton | 2–3 days |
| 1 — Auth | 1 week |
| 2 — Ingestion | 1.5 weeks |
| 3 — CAG Layer | 1.5 weeks |
| 4 — Query Layer | 1 week |
| 5 — Frontend | 2 weeks |
| 6 — Hardening | 1 week |
| **Total** | **~9–10 weeks** |

With a second contributor joining after Phase 2: frontend (Phase 5) runs in parallel with backend Phases 3–4. Cuts total to ~6 weeks.

---

## v2 Backlog

These are explicitly out of scope for v1. Do not let them creep in.

| Feature | Notes |
|---|---|
| Pagination on list endpoints (teams, projects, documents) | Unbounded lists fine for v1 scale | !IMPORTANT
| Notion ingestion adapter | Read from Notion database/pages |
| Slack ingestion adapter | Ingest channel history per project |
| Google Docs adapter | Read from Drive folder |
| GitHub issues adapter | Ingest issue/PR history per repo |
| GraphRAG entity layer | Entity-relationship graph for admin cross-project queries |
| SSO / OAuth | Google, GitHub, SAML |
| Multi-org support | One deployment serving multiple isolated organizations |
| Analytics dashboard | Query frequency, knowledge health metrics, CAG freshness |
| Document-level ACL | Fine-grained permissions below project level |
| Contradiction detection | Surface when two documents make conflicting claims |
| CLI tool | `project-cli ingest ./meeting-notes/` from local filesystem |
| Semantic versioning for belief states | Tag and compare belief states across time |
