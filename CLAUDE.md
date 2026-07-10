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
2. **Open-ended evolvability without a fixed objective.** No externally assigned task
   reward, goal label, behavioral script, demonstration, pretrained skill, or designer
   fitness score. Prediction, self-generated reachability/controllability, bodily
   interoception, physical mortality, and differential reproduction are legitimate
   mechanisms of the simulated reality. Shaping any of them toward a named behavior is
   drift.
3. **Every milestone ends with the world running and observable.** No
   infrastructure-only phases.
4. **World and brains checkpoint together** at the same tick, atomically. Resume must
   always be coherent.
5. **The sim never waits for learning.** Learner backpressure = skip updates, never
   stall the world.
6. **The obs/action contract** (`gol_world/interface.py`) changes only by deliberate
   versioned decision (`OBS_VERSION`).

## Research questions (why this exists)

1. Lifelong, non-episodic learning — can an agent learn in one unbroken life in a
   nonstationary world, and *which brain architectures can host that*? World-models
   (Dreamer-class) are the first bet, not the only one: model-free, predictive-coding,
   evolved-plasticity, and other families are all in scope, and the architecture itself
   is a research variable. The `Brain` interface is architecture-agnostic by design;
   any family that implements it lives in the same world beside the others. We do our
   own exploration here (params, learning rules, mechanisms), not just replicate
   existing results.
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
- `configs/` — world/brain/run YAML (dataclass defaults → YAML → `--set k=v`); round
  configs are save-name-prefixed (`beta_NN_*.yaml`) and freeze at launch — see
  `configs/README.md`. The prefix names a *track* (a brain-architecture family + the
  bet it's testing), not the whole project: `beta` is the world-model/Dreamer track; a
  new architecture branch gets its own prefix so tracks in the same world stay
  comparable and don't collide on one linear counter.
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
- Training speed/pacing, dreamer config flags, checkpoint compatibility, and the
  offline screening gym are in `docs/training-ops.md` — read it before staging a
  round; benchmark with `scripts/bench_learn.py` on the round's hardware before
  picking a world speed.
- The full design plan lives in `docs/architecture.md`.
