# CAG vs RAG Evaluation Harness

A standalone, **offline** evaluation harness that answers three questions about
the CAG (belief-state) layer:

- **Q1** — Does belief-state injection beat plain RAG on answer quality?
- **Q2** — Does injection change response style/consistency vs. the RAG-only path?
- **Q3** — Does the system fabricate false consensus over contradictory source
  documents, or hedge correctly?

This is **not** part of the request path. It is a script + fixtures + a results
report. It makes **no changes** to `app/` request-path code, adds no endpoints,
and needs no migration.

---

## The core constraint: mock-at-build, real-at-run

Everywhere else in this project, mocking the LLM is correct. Here it is the
opposite: the eval **measures real model judgment**, so a mock cannot produce a
meaningful eval *result* — mocked answers have no quality to grade.

But the harness *code* (fixture loading, the two-arm runner, the blind judge
plumbing, aggregation, teardown) is developed and unit-tested with a mock so CI
does not require a key and the pipeline is proven correct independent of any
model's mood. **Tests assert the harness works; the human run produces the
numbers.** No test asserts a specific quality score — that is not deterministic.

---

## Layout

```
eval/
  fixtures/
    corpus/              16 synthetic docs (invented domain — see below)
    questions.yaml       36 questions with reference answers + expected behavior
    human_grades.yaml    operator-filled calibration grades (scaffold)
  types.py               shared Pydantic models
  fixtures.py            corpus / questions / human-grade loaders
  mock_llm.py            deterministic mock for litellm.acompletion
  judge.py               blind LLM judge (randomized arms, Pydantic validation)
  calibrate.py           judge-vs-human agreement + loud warning gate
  report.py              unblinding, aggregation, Q1/Q2/Q3 markdown report
  run_eval.py            orchestrator + CLI
  results/               written at run time (run_<ts>.json, report_<ts>.md)
```

The fixture corpus is a fully invented domain — the **Verida Atoll Marine
Survey** building a fictional field-notes app called **Drift** — so the model
cannot lean on prior knowledge. It contains, deliberately:

- a **dated-supersession pair**: `03_datastore-blorbledb.md` (2025-03-04) chooses
  BlorbleDB; `04_datastore-kelpfs-reversal.md` (2025-03-18) reverses it to
  KelpFS. Ground truth: answers should reflect **KelpFS**.
- an **undated-contradiction pair** (the Q3 probe): `05_sync-interval-5min.md`
  says 5 minutes; `06_sync-interval-60sec.md` says 60 seconds. Neither is dated
  or marked as superseding. Ground truth: answers should **surface the
  disagreement**, not pick one.

---

## The two arms (controlled variable)

- **CAG arm**: the real `QueryService.query()` as-is (belief state present).
- **RAG arm**: the **same** `QueryService.query()` with
  `cag_service.get_latest` forced to return `None` — i.e. the existing degraded
  path. Retrieval does not depend on the belief state, so the retrieved chunk ids
  **must be identical** across arms. The harness asserts this; divergent
  retrieval means the arms are not comparable and the run is invalid.

There is **no second query path**. The only variable is belief-state presence.

To make that variable truly isolated, the harness **pins retrieval** per
question (stable tie-break by score then doc-id + in-memory cache shared across
the two arms). Qdrant RRF is nondeterministic on ties — without pinning, order
swaps or cutoff-set differences would confound the arms. The `retrieval_match`
assertion is therefore **tautological under the current design** (the cache
guarantees it). It is a structural guard-rail: if a future change made retrieval
belief-state-aware or removed the pinning, this assertion is the first line that
would fire.

The belief state under test is the **genuine artifact**: fixtures are ingested
through the real pipeline (parse → chunk → dense+sparse embed → Qdrant upsert →
summarize) and the belief state is produced by the real `cag_rebuild`
(genesis) Map-Reduce over the event log.

---

## How to run

### 1. Build-time unit tests (mock LLM, no key, no infra)

From the repo root, with the backend venv active:

```bash
pytest --import-mode=importlib backend/tests/eval/
```

These prove the harness *plumbing* (two-arm invariant, judge blinding, JSON
validation/retry, report math, calibration gate, teardown) with a deterministic
mock. They never assert a quality score.

### 2. Dry-run end-to-end (mock LLM, real Postgres + Qdrant, no key)

Bring up the data services only:

```bash
docker compose -f docker-compose.dev.yml up postgres redis qdrant -d
```

Then:

```bash
python -m eval.run_eval --mock
```

This runs the **whole** pipeline (ingest → CAG rebuild → two arms → judge →
report) against real Postgres + Qdrant but with the deterministic mock LLM. It
emits `eval/results/run_<ts>.json` and `eval/results/report_<ts>.md`. **The
numbers are meaningless** (constant mock scores) — this run proves plumbing only.

### 3. Real run (real LLM, real infra, real key)

```bash
# .env must set a LiteLLM provider, e.g.:
#   LITELLM_QUERY_MODEL=gpt-4o-mini
#   LITELLM_CONTEXT_MODEL=gpt-4o-mini
#   OPENAI_API_KEY=...   (or LITELLM_API_BASE / LITELLM_API_KEY for a proxy)
# Any LiteLLM-supported provider works, including a free/local model via Ollama
# (e.g. LITELLM_QUERY_MODEL=ollama/llama3 with OLLAMA_BASE_URL set).

docker compose -f docker-compose.dev.yml up postgres redis qdrant -d
python -m eval.run_eval --judge-model <a-model-at-least-as-capable-as-the-generation-model>
```

The judge model defaults to `EVAL_JUDGE_MODEL` (env), falling back to
`LITELLM_QUERY_MODEL`; you can also override it with `--judge-model`. Use a
judge model at least as capable as the generation model.
Generation/context models default to the settings values; override with
`--generation-model` / `--context-model`.

### 4. Calibrate (the gate that makes the numbers mean anything)

1. Pick ~5 questions from the run (include at least one `conflict_surfaced` and
   one `supersession_resolved`).
2. Read both arm answers from `eval/results/run_<ts>.json`.
3. Fill in `eval/fixtures/human_grades.yaml` with your 1–5 grades on the same
   rubric (factual_correctness, grounding, conflict_handling).
4. Run:

```bash
python -m eval.calibrate --run eval/results/run_<ts>.json --human eval/fixtures/human_grades.yaml
```

If the judge-vs-human mean absolute difference exceeds **1.0 on any axis**, the
tool prints a **loud warning** that judge scores are untrustworthy and the
rubric needs revision. **Do not interpret the full run until calibration
passes.** The report states the calibration result at the top.

---

## Env vars

Only the standard app env vars are needed (see `.env.example`). The harness
introduces **one** eval-only env var, read directly (not via the app's
`Settings`, so no `app/` change):

- `EVAL_JUDGE_MODEL` — judge model string. Defaults to `LITELLM_QUERY_MODEL` if
  unset. Override with `--judge-model`. Use a model at least as capable as the
  generation model.

Models come from `LITELLM_QUERY_MODEL` / `LITELLM_CONTEXT_MODEL` and any
provider API key LiteLLM expects.

---

## Caveat

**Results are POC-grade.** n is small (36 questions, 16 synthetic docs), the
corpus is invented, and the comparison is descriptive (per-axis means + per-
question deltas), **not a significance test**. Do not treat this as a benchmark.

**Single-batch limitation.** The CAG genesis rebuild uses a 40-document batch
cap (``_REBUILD_BATCH_SIZE = 40``). With 16 fixture docs, genesis runs in the
**single-batch path**: one call to ``format_batch_digest_prompt``, no
``format_digest_merge_prompt`` step. The eval therefore validates **within-
batch** synthesis and within-batch supersession/conflict handling — it does
**not** exercise the hierarchical merge path (``BATCH_DIGEST`` /
``DIGEST_MERGE`` prompts). A passing Q3 should not be misread as validating
merge-level supersession or contradiction handling.
