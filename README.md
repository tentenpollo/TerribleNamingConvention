# [PROJECT_NAME]

> A self-hostable, open-source hybrid RAG/CAG knowledge platform for teams. Each project gets its own living, structured understanding — not just a search index.

---

## The Problem

Teams lose institutional context constantly. Meeting notes pile up, decisions get buried, new members can't find why things were done a certain way. Existing tools treat documents as a flat search index with no understanding of the project as a whole.

## What This Does Differently

Every existing RAG tool is stateless — each query hits a flat index independently. This platform maintains a **structured, evolving belief state (CAG) per project** that orients every query before dropping into specific chunk retrieval (RAG). The system understands your project, not just your documents.

Access is team-scoped. Members only see their projects. Admins see everything. Enforced at the retrieval layer — not just the UI.

---

## Features

- **Per-project CAG layer** — compressed, structured project understanding (decisions, open items, people, themes) that updates as new content is ingested
- **Hybrid retrieval** — CAG orients, RAG grounds, LLM synthesizes
- **Team-scoped access** — enforced at the retrieval layer, not just the API
- **Configurable ingestion pipeline** — swap chunking strategy and models per project
- **Pluggable LLM backend** — OpenAI, Anthropic, local Ollama, anything LiteLLM supports
- **Self-hostable** — one command, Docker Compose, no external dependencies required
- **Immutable event log** — belief states are derived artifacts; raw data is never mutated

---

## Quick Start

### Prerequisites
- Docker and Docker Compose
- An LLM API key (or local Ollama)

### Run

```bash
git clone https://github.com/[your-handle]/[PROJECT_NAME].git
cd [PROJECT_NAME]
cp .env.example .env
# Edit .env with your LLM API key and preferred settings
docker compose -f docker-compose.dev.yml up
```

The app will be available at `http://localhost:3000`.
API docs at `http://localhost:8000/docs`.

### First Steps

1. Register a super admin account at `/register`
2. Create a team and invite members
3. Create a project and assign it to the team
4. Upload documents (markdown, txt, PDF)
5. Query your project

---

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 + FastAPI |
| Vector Store | Qdrant |
| Embeddings | FastEmbed |
| LLM Interface | LiteLLM |
| Database | PostgreSQL 16 |
| Task Queue | ARQ + Redis |
| Frontend | Next.js 16 |
| Auth | JWT + RBAC |
| Containerization | Docker Compose |

Full stack rationale in [`/docs/STACK.md`](/docs/STACK.md).

---

## Project Structure

```
/backend/app      FastAPI application
/frontend         Next.js application
/backend/tests    Pytest test suite
/backend/alembic  Database migrations
/docs             Architecture, stack, rules, roadmap
/tasks            Feature files and task breakdowns
docker-compose.dev.yml
.env.example
pyproject.toml
```

---

## Documentation

| Document | Description |
|---|---|
| [PROJECT.md](/docs/PROJECT.md) | Problem statement, goals, non-goals |
| [ARCHITECTURE.md](/docs/ARCHITECTURE.md) | System design, data flow, schema |
| [STACK.md](/docs/STACK.md) | Stack decisions with rationale |
| [RULES.md](/docs/RULES.md) | Coding conventions and patterns |
| [ROADMAP.md](/docs/ROADMAP.md) | Phase plan and v2 backlog |

---

## Contributing

See [`CONTRIBUTING.md`](/docs/CONTRIBUTING.md). The short version: open an issue before starting significant work, keep PRs focused, tests are required.

---

## License

MIT
