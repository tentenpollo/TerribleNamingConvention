# RULES.md

> Coding conventions, patterns to follow, patterns to avoid. These are not suggestions.

---

## General Principles

1. **Clarity over cleverness.** This project may have contributors who didn't write the original code. Write for them.
2. **Explicit over implicit.** No magic. If something happens, it should be obvious where and why.
3. **Fail loudly.** Raise exceptions with clear messages. Never silently swallow errors.
4. **Tests are not optional.** Every new function, service method, and route handler has a corresponding test.
5. **One thing per PR.** A PR that adds a feature should not also refactor unrelated code.

---

## Python Conventions

### Typing
- All function signatures must have type annotations — parameters and return types
- Use `from __future__ import annotations` at the top of every file
- Prefer `X | None` over `Optional[X]` (Python 3.10+ style)
- Use `TypeAlias` for complex types used in multiple places

```python
# Good
async def get_project(project_id: UUID, user: User) -> Project | None:
    ...

# Bad
async def get_project(project_id, user):
    ...
```

### Async
- All database calls, LLM calls, and Qdrant calls must be `async`/`await`
- Never use `asyncio.run()` inside a service or route handler
- Never use sync SQLAlchemy — always use `async with AsyncSession`
- Use `asyncio.gather()` for concurrent independent operations, not sequential awaits

```python
# Good — concurrent
belief_state, chunks = await asyncio.gather(
    cag_service.get_belief_state(project_id),
    retrieval_service.retrieve(query, project_id),
)

# Bad — sequential when they could be concurrent
belief_state = await cag_service.get_belief_state(project_id)
chunks = await retrieval_service.retrieve(query, project_id)
```

### Error Handling
- Define custom exceptions in `/backend/app/core/exceptions.py`
- Raise specific exceptions, catch specific exceptions
- Never use bare `except:` or `except Exception:` unless you're at the top-level error handler
- Always include context in exception messages

```python
# Good
raise ProjectNotFoundError(f"Project {project_id} does not exist")

# Bad
raise Exception("not found")
```

### Imports
- Standard library first, third-party second, local last — separated by blank lines
- Ruff handles this automatically — do not fight it
- Avoid wildcard imports (`from module import *`)
- Use absolute imports for cross-module references

### Naming
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Prefix private methods with `_` (single underscore)
- Never abbreviate unless the abbreviation is universally understood (e.g., `id`, `url`, `api`)

---

## FastAPI Conventions

### Route Handlers Are Thin
Route handlers do exactly three things: validate input, call a service, return output. No business logic.

```python
# Good
@router.post("/{project_id}/query", response_model=QueryResponse)
async def query_project(
    project_id: UUID,
    body: QueryRequest,
    user: User = Depends(get_current_user),
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    return await query_service.query(
        project_id=project_id,
        query=body.query,
        user=user,
    )

# Bad — logic in handler
@router.post("/{project_id}/query")
async def query_project(project_id: UUID, body: QueryRequest):
    # checking access here, building prompts here, calling qdrant here...
    ...
```

### Dependency Injection
- Auth: `get_current_user` — always required on protected routes
- RBAC: `require_role(Role.ADMIN)` — composable, declare at route level
- Services: injected via `Depends(get_X_service)` — never instantiated inside handlers
- DB session: `get_async_session` — never create sessions manually in handlers

### Schemas
- Request bodies: `XRequest` (e.g., `QueryRequest`)
- Response bodies: `XResponse` (e.g., `QueryResponse`)
- DB models live in `/app/models/`, Pydantic schemas in `/app/schemas/`
- Never return ORM model objects directly from route handlers — always use response schemas
- Use `model_config = ConfigDict(from_attributes=True)` on schemas that map from ORM models

### HTTP Status Codes
- `200` — successful GET, successful query
- `201` — successful POST that creates a resource
- `202` — accepted (async job queued)
- `400` — bad request (validation error)
- `401` — not authenticated
- `403` — authenticated but not authorized
- `404` — resource not found
- `422` — request body validation failure (FastAPI default for Pydantic errors)
- `500` — unexpected server error

---

## Service Layer Conventions

- Services are classes with `async` methods
- Receive dependencies (DB session, Qdrant client, LLM wrapper) via constructor, not globals
- Never import `Request` from FastAPI inside a service
- Services raise domain exceptions, not HTTP exceptions — the route handler converts them

```python
# Good — service raises domain exception
class ProjectService:
    async def get_project(self, project_id: UUID) -> Project:
        project = await self.db.get(Project, project_id)
        if not project:
            raise ProjectNotFoundError(f"Project {project_id} not found")
        return project

# Bad — service raises HTTP exception
class ProjectService:
    async def get_project(self, project_id: UUID) -> Project:
        project = await self.db.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="not found")  # wrong layer
        return project
```

---

## Retrieval Layer Conventions

- Every retrieval function must receive `accessible_project_ids: list[UUID]` as a parameter
- Never call `get_accessible_project_ids` inside the retrieval layer — it must be passed in
- This makes access scoping explicit and testable
- Never hardcode collection names — always derive from `project_{project_id}`

```python
# Good
async def query(
    query: str,
    project_id: UUID,
    accessible_project_ids: list[UUID],  # always explicit
) -> QueryResult:
    if project_id not in accessible_project_ids:
        raise AccessDeniedError(...)
    ...
```

---

## CAG / Belief State Conventions

- Never UPDATE a belief state row — always INSERT a new versioned row
- The current belief state is always the row with the highest `version` for a given `project_id`
- Never delete `document_summaries` rows — this is the immutable event log
- Belief state generation must be idempotent — running it twice with the same input produces equivalent output

---

## Database Conventions

### Alembic Migrations
- One migration per PR that touches the schema
- Migration file names must be descriptive: `add_belief_state_version_column.py` not `abc123.py`
- Always review autogenerated migrations before committing — Alembic sometimes generates unnecessary changes
- Never edit a migration that has already been applied to any environment

### SQLAlchemy Models
- All models inherit from a shared `Base` in `/backend/app/models/base.py`
- All tables have a UUID primary key using `gen_random_uuid()`
- All tables have a `created_at` TIMESTAMPTZ with server default `now()`
- Foreign keys always have explicit `ON DELETE` behavior declared

### Queries
- Use SQLAlchemy Core (`select()`, `insert()`) for simple queries
- Use ORM for complex relationships
- Always paginate list endpoints — never return unbounded lists to the API

---

## Testing Conventions

### Structure
- `tests/` mirrors `app/` structure
- `tests/api/` — route handler tests (uses HTTPX AsyncClient)
- `tests/services/` — service unit tests (mock external dependencies)
- `tests/retrieval/` — retrieval logic tests
- `tests/ingestion/` — ingestion pipeline tests
- `tests/integration/` — tests that use real Postgres and Qdrant

### Rules
- Every new function needs at least one test
- Every new route handler needs: success case, auth failure case, not-found case
- Every new service method needs: success case, failure case, edge cases
- Mock LLM calls in unit tests — never make real LLM API calls in tests
- Use `pytest.mark.integration` for tests that require external services
- Use fixtures for: creating test users, test teams, test projects, auth tokens

### Fixture Conventions
```python
# conftest.py — shared fixtures
@pytest.fixture
async def test_user(db_session) -> User:
    ...

@pytest.fixture
async def test_project(db_session, test_team) -> Project:
    ...

@pytest.fixture
def member_token(test_user) -> str:
    return create_jwt(test_user)

@pytest.fixture
def admin_token(test_admin_user) -> str:
    return create_jwt(test_admin_user)
```

---

## Git Conventions

### Branch Naming
```
feature/short-description     new feature
fix/short-description         bug fix
chore/short-description       non-functional (deps, config, docs)
refactor/short-description    code change with no behavior change
test/short-description        adding or fixing tests
```

### Commit Messages
Follow conventional commits:
```
feat: add contextual chunking pipeline
fix: scope qdrant query to accessible collections
chore: update fastapi to 0.115
refactor: extract belief state builder into separate module
test: add integration tests for ingestion job
docs: update architecture diagram with CAG flow
```

### PR Rules
- PRs must have a description explaining what and why
- All CI checks must pass before merge
- At least one review required (when working with collaborators)
- Keep PRs focused — one concern per PR
- Link to relevant issue if one exists

---

## Logging

Use the structured logger from `/backend/app/core/logging.py`. Never use `print()`.

```python
from app.core.logging import logger

logger.info("CAG rebuild triggered", project_id=str(project_id), doc_count=count)
logger.error("Ingestion failed", document_id=str(doc_id), error=str(e))
```

Log levels:
- `DEBUG` — detailed diagnostic info, not in production by default
- `INFO` — normal operational events (job started, job completed, query served)
- `WARNING` — unexpected but handled situations
- `ERROR` — failures that need attention

Never log: passwords, JWT tokens, raw document content, API keys.

---

## Environment and Config

- All config lives in `.env` and is loaded via Pydantic `BaseSettings` in `/backend/app/core/config.py`
- Never hardcode any config value in application code
- Never commit `.env` — only commit `.env.example`
- `.env.example` must be kept up to date — every new config key added to the app must be added to `.env.example` with a comment explaining it
