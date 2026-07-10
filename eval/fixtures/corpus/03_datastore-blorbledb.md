# Drift — Datastore Decision: BlorbleDB

**Date:** 2025-03-04

## Decision
Drift will use **BlorbleDB** as the on-device local datastore for all field
notes.

## Reasoning
BlorbleDB is a small embedded key-value store. Maren proposed it because it
claims sub-millisecond reads on ruggedized tablets and has a simple file-based
persistence model that survives power loss.

## Consequences
- All notes are stored as BlorbleDB documents keyed by observation id.
- Sync state is tracked in a separate BlorbleDB table.
- We will not pursue KelpFS or any other local store at this time.

Action item: Maren to scaffold the BlorbleDB wrapper by 2025-03-10.
