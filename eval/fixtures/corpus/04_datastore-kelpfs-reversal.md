# Drift — Datastore Reversal: KelpFS

**Date:** 2025-03-18

## Decision
We are reversing the BlorbleDB decision made on 2025-03-04. Drift will now use
**KelpFS** as the on-device local datastore.

## Reasoning
BlorbleDB corrupted two test notebooks during a simulated power-loss trial on
2025-03-15. The corruption was unrecoverable. KelpFS uses an append-only log
with periodic snapshots, which survived the same trial with no data loss.

## Consequences
- The BlorbleDB wrapper work is abandoned.
- All notes are stored as KelpFS append-only records keyed by observation id.
- Sync state is tracked in a KelpFS meta-segment.
- BlorbleDB is no longer a candidate for Drift.

Action item: Maren to scaffold the KelpFS wrapper by 2025-03-25.
