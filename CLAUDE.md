# GameOfLife

A persistent simulated voxel world inhabited by robot agents that live continuously,
interact with the world and each other, and learn lifelong from their own experience —
"a more real Game of Life." This is a research platform for emergent behavior, not a
training environment.

## Invariants (anti-drift rules — these killed the previous version of this project)

The predecessor (`~/simulator`) drifted from "persistent world of interacting robots"
into "episodic RL training pipeline." These rules exist so that never happens again:

1. **One persistent world.** No episodes, no `reset()`, no episode counters, no Gym API,
   no task registry, no train/deploy split. If a change introduces any of these
   concepts, it is wrong.
2. **No designer-assigned tasks or fitness functions.** Behavior comes only from
   intrinsic drives (curiosity + homeostasis). Reward shaping toward a task is drift.
3. **Every milestone ends with the world running and observable.** No
   infrastructure-only phases.
4. **World and brains checkpoint together** at the same tick, atomically. Resume must
   always be coherent.
5. **The sim never waits for learning.** Learner backpressure = skip updates, never
   stall the world.
6. **The obs/action contract** (`gol_world/interface.py`) changes only by deliberate
   versioned decision (`OBS_VERSION`).

## Research questions (why this exists)

1. Lifelong, non-episodic world-model learning — can a Dreamer-class agent learn in one
   unbroken life in a nonstationary world?
2. Intrinsic motivation in a shared world — what happens when curiosity-driven agents
   are each other's most unpredictable objects?
3. Emergent communication — does a free signal channel acquire meaning under survival
   pressure?
4. Cultural transmission — does warm-starting newborns from living agents' weights
   propagate behaviors?

## Layout

- `packages/world/gol_world` — voxels, terrain, physics, entities, sensing, persistence
- `packages/brains/gol_brains` — Brain interface, scripted baselines, `dreamer/`
- `packages/runtime/gol_runtime` — persistent loop, learner thread, checkpoints, CLI
- `packages/obs/gol_obs` — Rerun logging, metrics/events writers
- `configs/` — world/brain/run YAML (dataclass defaults → YAML → `--set k=v`)
- `saves/` — gitignored persistent worlds; a save dir is a reproducible experiment

## Commands

- `uv sync` — install
- `uv run gol-run --new saves/<name>` / `--resume saves/<name>` — run a world (add
  `--headless --ticks N` for unpaced headless)
- `uv run gol-ctl pause|resume|speed <x>|checkpoint` — control a running world
- `uv run gol-stats <save-dir>` — analyze metrics/events
- `uv run pytest` (`-m "not slow"` for quick), `uv run ruff check .`, `uv run mypy packages`

## Conventions

- Python ≥3.11, strict mypy, ruff line length 100.
- Sim hot paths are vectorized numpy across all agents (raycasts, physics); brains are
  the expensive part, not the world.
- Determinism: world stepping is deterministic given seed + actions; `--sync` mode
  exists for determinism tests.
- The full design plan lives in `docs/architecture.md`.
