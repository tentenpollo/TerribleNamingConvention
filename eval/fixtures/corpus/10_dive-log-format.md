# Drift — Dive Log Format

**Date:** 2025-03-01

Each Drift observation can optionally attach a dive log entry. A dive log entry
has:

- `dive_number`: integer, sequential per diver per day.
- `bottom_time_min`: integer.
- `max_depth_m`: float.
- `gas`: one of `air`, `nitrox32`, `nitrox36`.

Dive logs are stored alongside the observation in the same local datastore and
sync as a single unit. A note without a dive log is still valid.
