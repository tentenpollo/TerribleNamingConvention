# Phase 4 Summary — Query Layer Operational Hardening

> Completes Phase 4 of the roadmap: per-user rate limiting, LLM timeout/graceful failure,
> and token/cost telemetry for the query path.

---

## Sub-Phase Breakdown

### 4.1 Rate Limiting
- Added `app/core/ratelimit.py` with a Redis sorted-set sliding-window limiter.
- Key pattern: `rl:query:{user_id}`.
- Pipeline: `ZREMRANGEBYSCORE` (prune window), `ZADD` (record request), `ZCARD` (count),
  `PEXPIRE` (TTL).
- Configurable via `QUERY_RATE_LIMIT_PER_MINUTE` (default: 20).
- Added a dedicated `redis.asyncio` client in the app lifespan, separate from the ARQ
  broker pool, with `get_rate_limit_redis` dependency.
- Wired `rate_limit_query` dependency to both `POST /projects/{id}/query` and `POST /query`.
- Returns `429 Too Many Requests` with `Retry-After` (seconds until oldest window entry
  expires). Fails open with an error log if Redis is unreachable.

### 4.2 LLM Timeout and Graceful Failure
- Added `query_llm_timeout_seconds` setting (default: 60).
- Wrapped query LLM generation in `asyncio.timeout`.
- Added `QueryGenerationError`; handler returns `503 Service Unavailable` with
  `{"detail": "...", "retryable": true}`.
- On timeout or `LLMError`, retrieval context is preserved and logged at error level
  (`chunk_count`, `belief_state_version`) for diagnosis.

### 4.3 Token/Cost Telemetry
- Changed `llm_call` to return `LLMResult(text, prompt_tokens, completion_tokens, model)`.
- Migrated all call sites in the same PR: summarizer, CAG synthesis, and query service
  now read `.text`.
- `QueryService` emits one `query_completed` structlog event per query with token counts,
  model, duration, and grounding metadata. No question, chunk, or answer text is logged.

### 4.4 Config + Docs
- Added `query_rate_limit_per_minute` and `query_llm_timeout_seconds` to `Settings`.
- Added documented entries to `.env.example`.
- Updated `docs/ARCHITECTURE.md` with an "Operational Limits" section.

---

## Test Coverage

| Area | Tests |
|------|-------|
| Rate limiter unit | `backend/tests/core/test_ratelimit.py` — allows N, rejects N+1 with `Retry-After`, window slide, fail-open |
| LLMResult | `backend/tests/core/test_llm.py` — usage block and missing usage |
| Migrated call sites | `test_summarizer.py`, `test_workers/test_cag.py`, `test_services/test_query.py` |
| 503 handler | `backend/tests/api/test_query.py::test_query_generation_error_handler_returns_503_retryable` |
| Rate-limit dependency | `backend/tests/api/test_query.py::test_rate_limit_query_dependency_rejects_over_limit` |
| Log privacy | `backend/tests/services/test_query.py::test_query_completed_log_contains_no_user_content` |
| Red-team structural containment | `backend/tests/api/test_query.py::test_prompt_injection_containment_structural` (integration) |
| End-to-end rate limit | `backend/tests/api/test_query.py::test_query_rate_limit_integration_429_with_retry_after` (integration) |

Verification table:

| Check | Status |
|-------|--------|
| Unit tests pass | `pytest -m "not integration and not e2e"` → 249 passed |
| Lint clean | `ruff check backend/app backend/tests` → pass |
| Format clean | `ruff format backend/app backend/tests` → pass |
| Type clean | `mypy backend/app` → success |
| No bare-string llm_call consumers | Confirmed via grep of `app/` |

---

## Known Limitations

1. **Cross-collection RRF comparability.** Cross-project query merges scores from
   per-project Qdrant collections via RRF. Scores from different collections are not
   directly comparable; the current top-k merge is v1-simple and may under-rank sources
   from projects with sparser collections.
2. **No streaming.** Query responses are returned in a single completion. Server-sent
   events / streaming responses are out of scope for v1.
3. **Fail-open rate limiter.** Redis failures are logged and allowed through by design.
   This keeps querying available during broker outages but removes protection until Redis
   recovers.
4. **Timeout covers generation only.** The `query_llm_timeout_seconds` cap wraps the
   LLM generation call, not retrieval or belief-state loading. A hung Qdrant query or
   slow `get_latest` has no request-level ceiling in v1. A proper end-to-end request
   timeout belongs at the ASGI/proxy layer (Phase 6).

---

## Files Changed

- `backend/app/core/config.py`
- `backend/app/core/exceptions.py`
- `backend/app/core/llm.py`
- `backend/app/core/ratelimit.py` (new)
- `backend/app/core/dependencies.py`
- `backend/app/services/query.py`
- `backend/app/ingestion/summarizer.py`
- `backend/app/workers/cag.py`
- `backend/app/api/query.py`
- `backend/app/main.py`
- `.env.example`
- `docs/ARCHITECTURE.md`
- `backend/tests/core/test_llm.py` (new)
- `backend/tests/core/test_ratelimit.py` (new)
- `backend/tests/services/test_query.py`
- `backend/tests/ingestion/test_summarizer.py`
- `backend/tests/workers/test_cag.py`
- `backend/tests/services/test_cag_integration.py`
- `backend/tests/api/test_query.py`
