# Drift — Offline-First Principle

**Date:** 2025-02-20

Offline-first is a non-negotiable principle for Drift. The app must never
require network access to create, edit, or read a note. Sync is a background
best-effort process; failure to sync must never prevent a diver from logging an
observation.

If the local datastore and the sync server disagree, the local copy wins for
edits made within the last 24 hours; older conflicts are flagged for Dr. Wynne
to review.
