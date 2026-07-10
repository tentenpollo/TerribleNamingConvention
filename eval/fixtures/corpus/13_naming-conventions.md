# Drift — Naming Conventions

**Date:** 2025-02-28

Code and storage naming conventions for Drift:

- Observation ids are lowercase kebab-case: `obs-<sector>-<yyyymmdd>-<seq>`,
  e.g. `obs-north-20250304-007`.
- Species names use the common name as written by Dr. Wynne, e.g.
  `Verida bubblefish` (capitalised genus-style first word).
- Sectors are single lowercase words: `north`, `east`, `south`, `west`.
- Database tables are snake_case and prefixed by domain: `obs_observations`,
  `obs_dive_logs`, `sync_state`.

Do not abbreviate species names in stored data; the full common name is always
persisted.
