# Drift — Export Format

**Date:** 2025-03-05

Drift exports a diver's notes as a single JSON document per survey sector. The
export shape is:

```
{
  "sector": "north",
  "exported_at": "<ISO8601>",
  "observations": [ ... ]
}
```

Exports are generated on the sync server, not on the tablet, so they reflect the
synced state rather than a diver's local-only edits. Exports do not include
dive logs by default; a `?include_dive_logs=true` flag adds them.
