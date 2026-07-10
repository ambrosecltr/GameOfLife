# GameOfLife

A persistent simulated 3D voxel world inhabited by artificial organisms that live
continuously, interact with the world and each other, and learn lifelong from their own
experience. A more "real" Game of Life.

There are no episodes, no resets, no tasks, and no fitness functions. One world runs for
days; robots forage, dig, build, signal, hibernate, and die in it; each learning robot
carries its own predictive world model trained online from its own life. The brains can
develop reusable temporal skills while feeling only endogenous consequences of their
bodies and predictions. The point is watching what emerges.

## The research

Despite the name this is a research platform, not a game — though like Conway's
original it's a zero-player one: nothing to win, everything to watch. The overarching
question: **what behaviors, traits, and systems emerge in a world with no fixed
objective?** No designer assigns task rewards, skill labels, demonstrations, pretrained
behaviors, or a fitness score. There are bodies that need energy and can die, minds that
predict, feel, explore, and develop reusable skills, and a shared world that other minds
keep changing. Whatever shows up — survival, sociality, signaling conventions, mating,
culture, or something unanticipated — has to grow from those mechanisms.

Concrete questions structure the long runs
(in full: [research questions](docs/research-questions.md)). Four are about what an agent
*learns and does*:

1. **Lifelong, non-episodic world-model learning.** The world-model literature
   (DreamerV3, Plan2Explore, TD-MPC2) resets its agents thousands of times into a
   stationary environment. Can a Dreamer-class agent learn through *one unbroken life*
   in a world that never holds still? Watch for plasticity loss and forgetting of
   places not seen for a sim-day.
2. **Intrinsic motivation in a shared world.** To a curiosity-driven agent, the most
   unpredictable object around is another curiosity-driven agent — uncertainty about a
   moving target never fully resolves. Mutual fascination? Avoidance once modeled?
   Chasing? An ablation flag masks other agents out of the curiosity signal to compare.
3. **Emergent communication.** The 2-channel broadcast costs a little energy and means
   nothing by design. Does it acquire meaning under survival pressure — bursts at food
   discoveries, distress near death, silence at night?
4. **Cultural transmission.** When newborns are warm-started from living agents'
   weights, do behaviors — foraging routes, hoarding sites — propagate across
   "generations"?

Four more are about what the *system* becomes over time — these the long runs surfaced on
their own rather than being planned, and they answer the "no goals" question most directly:

5. **What keeps a mind motivated for a whole lifetime?** The project's central question,
   discovered in the runs: curiosity is self-extinguishing — master the world and there's
   nothing left to be surprised by — so how does motivation stay alive across one life,
   handing off between drives as each satisfies itself?
6. **Where does individuality come from?** Identical minds with identical drives, in a
   shared world, have already diverged into distinct and persistent personalities. Do they
   have to, and what carries the identity?
7. **Does the world itself evolve under the population?** A persistent world shaped by
   adaptive agents generates selection pressure back at them — an accidental plant-defense
   ecology already appeared under grazing pressure, with no ecology designed in.
8. **Can natural selection arise without a fitness function?** If reproduction becomes a
   bodily process funded by an agent's own energy surplus, does a population under *no*
   fitness function develop real selection and a cross-generational competence ratchet?

Every long run is attributable to at least one of these, and every round of runs gets a
written-up finding in the [research journal](docs/research_journal/) — negative and
confounded results included, since the entries are the durable record after a save is
pruned. The early rounds already produced surprises: an accidental plant-defense
ecology under grazing pressure, curiosity collapsing once the world became predictable,
and behavioral individuality arriving before survival competence.

## Quick start

```bash
uv sync
uv run gol-run saves/alpha --new          # create a world and watch it in Rerun
uv run gol-run saves/alpha --resume       # continue where it left off
uv run gol-ctl pause                      # control a running world (speed/checkpoint too)
uv run gol-stats saves/alpha              # dig through metrics and events
uv run gol-stats saves/alpha --compare    # are the dreamers pulling ahead of chance?
uv run gol-stats saves/alpha --interests  # do agents differ, and stay themselves?
scripts/soak.sh saves/soak_001            # overnight run, restart-on-crash
scripts/provision_runpod.sh root@gpu-box saves/alpha   # ship a world to a cloud GPU
```

## The shape of the thing

- **World**: finite Minecraft-like voxel world (terrain, water, ore, food bushes,
  day/night). Blocks are diggable and placeable. Physics is Minecraft-style
  AABB-vs-voxel — rich and modifiable, but cheap.
- **Robots**: wheeled bodies with color raycast vision (depth + shaded RGB — what a
  block *is* has to be read from how it looks) and a steerable gaze, plus a gripper,
  touch, an energy store, and a free 2-channel signal broadcast. Food restores energy;
  night stops regrowth; running out means hibernation, then death — which drops scrap
  back into the world.
- **Brains**: pluggable. Scripted baselines (random walker, forager) share the world
  with learning agents: per-robot Dreamer-style world models with imagination-trained
  critics, endogenous affect (interoception, curiosity, boredom, predicted mortality),
  and an optional learned temporal-skill manager/worker layer. No task reward, named
  skill, demonstration, pretrained behavior, or fitness score exists in the learning
  path.
- **Observability**: [Rerun](https://rerun.io) — live 3D scene, per-agent charts
  (energy, curiosity, prediction error), scrubbable timelines, recordings.
- **Compute tiers**: 1–2 learning robots locally (Apple Silicon); populations of 8–16
  on a single rented cloud GPU. Worlds checkpoint atomically and resume anywhere.
- **Minds outlive bodies**: with `inherit_weights: lineage` (what the standard run
  configs use), a learning brain's weights and memory carry over to its respawned
  body — death is costly, but the lineage keeps learning. `none` and `random_living`
  exist for the cultural-transmission experiments.
- **Experiments**: `configs/run/exp_*.yaml` are pre-registered protocols for the
  research questions (social curiosity with/without agent-masked curiosity,
  cultural transmission across inheritance modes). Long-run rounds get their own
  save-name-prefixed configs (`beta_NN_*.yaml`) that freeze at launch, so every
  save dir is a reproducible experiment.

## Docs

- [Architecture](docs/architecture.md) — full design: world, brains, runtime, milestones
- [Research questions](docs/research-questions.md) — what this platform is built to probe
- [Research journal](docs/research_journal/) — findings from long runs, one entry per round
- [Research proposals](docs/research_proposals/) — designs argued out before they land
- [Configs](configs/README.md) — how config layering and frozen round configs work
