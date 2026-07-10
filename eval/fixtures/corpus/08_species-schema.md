# Drift — Species Schema

**Date:** 2025-02-22

An observation record in Drift has this shape:

- `observation_id`: string, unique per note.
- `sector`: one of `north`, `east`, `south`, `west`.
- `species`: the fictional species name, e.g. `Verida bubblefish` or
  `kelp-glider eel`.
- `count`: integer, number of individuals sighted.
- `depth_m`: float, sighting depth in metres.
- `observer`: the diver's name.
- `notes`: free text, max 2000 characters.

All species in Drift are fictional Verida Atoll endemics created for survey
training. The full training list is: Verida bubblefish, kelp-glider eel,
atoll skipper crab, and reef lantern jelly.
