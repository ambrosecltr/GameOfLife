# GameOfLife

A persistent simulated 3D voxel world inhabited by little robots that live continuously,
interact with the world and each other, and learn — lifelong, from their own experience,
driven only by curiosity and survival. A more "real" Game of Life.

There are no episodes, no resets, no tasks, and no fitness functions. One world runs for
days; robots forage, dig, build, signal, hibernate, and die in it; each learning robot
carries its own DreamerV3-style world model trained online from its own life. The point
is watching what emerges.

## Quick start

```bash
uv sync
uv run gol-run --new saves/alpha          # create a world and watch it in Rerun
uv run gol-run --resume saves/alpha       # continue where it left off
uv run gol-ctl pause                      # control a running world
uv run gol-stats saves/alpha              # dig through metrics and events
```

## The shape of the thing

- **World**: finite Minecraft-like voxel world (terrain, water, ore, food bushes,
  day/night). Blocks are diggable and placeable. Physics is Minecraft-style
  AABB-vs-voxel — rich and modifiable, but cheap.
- **Robots**: wheeled bodies with raycast vision, a gripper, touch, an energy store, and
  a free 2-channel signal broadcast. Food restores energy; night stops regrowth;
  running out means hibernation, then death — which drops scrap back into the world.
- **Brains**: pluggable. Scripted baselines (random walker, forager) share the world
  with learning agents: per-robot DreamerV3-style agents (RSSM world model,
  imagination-trained actor-critic) rewarded only by Plan2Explore curiosity +
  homeostasis.
- **Observability**: [Rerun](https://rerun.io) — live 3D scene, per-agent charts
  (energy, curiosity, prediction error), scrubbable timelines, recordings.
- **Compute tiers**: 1–2 learning robots locally (Apple Silicon); populations of 8–16
  on a single rented cloud GPU. Worlds checkpoint atomically and resume anywhere.

## Docs

- [Architecture](docs/architecture.md) — full design: world, brains, runtime, milestones
- [Research questions](docs/research-questions.md) — what this platform is built to probe
- [Journal](docs/journal.md) — findings from long runs
