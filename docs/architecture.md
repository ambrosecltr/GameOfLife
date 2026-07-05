# GameOfLife — A Persistent World of Learning Robots (research-grade)

## Context

Restart of `~/simulator`, realigned to the original vision: a persistent simulated 3D world inhabited by robot agents that live continuously, interact with the world and each other, and learn lifelong from their own experience — "a more real Game of Life." The old project drifted into an episodic RL training pipeline because bipedal locomotion consumed everything. The new design makes bodies cheap and the world rich, and treats the project as **a research platform, not just a simulation**: state-of-the-art world-model brains, no cut corners, cloud GPU compute when the vision needs it.

**The research questions this platform is built to probe** (stated in docs so the project never loses them again):
1. **Lifelong, non-episodic world-model learning** — can a Dreamer-class agent learn continuously in one unbroken life (no resets, no episodes, nonstationary world)? Most world-model literature is episodic; this is genuinely underexplored.
2. **Intrinsic motivation in a shared world** — what happens when curiosity-driven agents are each other's most unpredictable objects? (Social curiosity, chasing, avoidance.)
3. **Emergent communication** — does a free 2-channel signal broadcast acquire meaning under survival pressure?
4. **Cultural transmission** — does warm-starting newborns from living agents' weights propagate behaviors?

## Decisions (user-confirmed + revised after review)

- **World**: rich Minecraft-like 3D voxel world — diggable/placeable blocks, resources, terrain, water, day/night. Minecraft-style AABB-vs-voxel physics (Minecraft itself uses no rigid-body engine) keeps a *rich, modifiable* world computationally cheap.
- **Bodies**: wheeled robots (drive + gripper + raycast vision + touch/proprioception + signal channel). Controllable from step one — no locomotion learning trap.
- **Learning**: in-world, lifelong, online. ONE persistent world. No episodes, no resets, no Gym API, no task registry. World + brains checkpoint together; stop/resume across days.
- **Brains**: **full DreamerV3-style agent per robot** (see below). RSSM included — the DreamerV3 recipe is specifically engineered for out-of-the-box robustness, which is exactly what unsupervised lifelong learning needs. Intrinsic drives only: **Plan2Explore ensemble-disagreement curiosity + homeostasis**.
- **Compute tiers**: develop and run 1–2 learning agents locally on the M1 Pro; rent a single cloud GPU (RTX 4090-class, ~$0.35–0.70/hr on RunPod/Vast) for populations of 8–16. A 24–72h soak costs roughly $10–50. Whole stack runs on one machine per run (sim is CPU-cheap; brains use the GPU) — no distributed systems complexity.
- **Observability**: **Rerun** (rerun.io) as the primary viewer — a purpose-built research visualization tool (3D scene + time-series + timeline scrubbing + remote streaming + recording files), not a hand-rolled HTML viewer. A tiny control API (`gol-ctl`) handles interactivity Rerun doesn't (pause/speed/checkpoint-now).

### Stack evaluation (considered fresh, not inherited)
- **Sim language**: Python + numpy. Considered: Rust core (premature — sim is not the bottleneck, brains are; revisit only if profiling says so), JAX end-to-end à la Craftax (wrong shape — that's for thousands of vectorized episodic envs; we have ONE persistent world with dynamic entities), Luanti/Minetest via Craftium (episodic Gym framing, poor determinism/persistence control), game engines (hard coupling to training). A custom ~2k-line numpy voxel sim gives full determinism, persistence, and multi-agent control.
- **Brain framework**: **PyTorch** (works on MPS locally and CUDA in cloud; `torch.compile` on CUDA; JAX-on-Metal is poor, which would kill the local tier). Official DreamerV3 (danijar/dreamerv3, JAX) and NM512/dreamerv3-torch serve as reference implementations; we implement in-repo, sized to our observations.
- **Tooling**: Python ≥3.11, uv, ruff, mypy, pytest — carried over because it was the one thing the old project got right.

## Repo layout

```
GameOfLife/
├── pyproject.toml              # uv, packages/* auto-discovery
├── README.md / CLAUDE.md       # CLAUDE.md: anti-drift invariants + research questions
├── configs/
│   ├── world/default.yaml      # size, seed, terrain, resources, day length, economy
│   ├── brain/dreamer.yaml      # model sizes, curiosity weights, ablation flags
│   ├── run/local_m1.yaml       # 2 learners + 6 scripted, small model, CPU/MPS
│   └── run/cloud_gpu.yaml      # 12–16 learners, CUDA, bigger model
├── packages/
│   ├── world/gol_world/        # voxels, terrain, physics, entities, sensing, persistence
│   ├── brains/gol_brains/      # Brain interface, scripted brains, dreamer/
│   ├── runtime/gol_runtime/    # persistent loop, scheduler, checkpointing, CLI, control API
│   └── obs/gol_obs/            # Rerun logging, metrics/events writers, replay export
├── scripts/                    # soak.sh, cloud provisioning (provision_runpod.sh)
├── docs/                       # architecture.md, research-questions.md, journal.md
└── saves/                      # gitignored persistent worlds
```

CLI: `gol-run` (create/resume + run a world), `gol-ctl` (pause/speed/checkpoint/spawn via local HTTP), `gol-stats` (analyze a save's metrics). Config layering: frozen dataclasses → YAML → `--set key=value`. Every save dir has `manifest.json` (config snapshot, seed, git commit) — every long run is a reproducible experiment.

**Anti-drift invariants (go in CLAUDE.md):** every milestone ends with the persistent world running and observable; the codebase contains no `reset()`, no episode counter, no Gym API, no task registry, no fitness function.

## World — `packages/world/gol_world/`

Files: `blocks.py`, `grid.py`, `terrain.py`, `physics.py`, `entities.py`, `sensing.py`, `interface.py`, `world.py`, `persistence.py`.

- **Grid**: finite world, default 256×256×64, one dense `np.uint8` array (~4 MB). Chunks (16×16 columns) are generation + dirty-tracking units. Unbreakable border.
- **Blocks**: `Block(IntEnum)`: AIR, BEDROCK, ROCK, SOIL, GRASS, SAND, WATER, BUSH_EMPTY, BUSH_RIPE, ORE, SCRAP. Numpy lookup tables SOLID/DIGGABLE/COLOR (single palette source, shared with Rerun logging).
- **Terrain**: seeded numpy value-noise fBm heightmap → layered blocks, water basins, ore pockets, bushes on grass. Deterministic per seed (tested).
- **Ecology**: bushes = food economy; eat flips RIPE→EMPTY; regrowth min-heap flips back **daytime only** — night scarcity is the core survival pressure. `light_level` in observations.
- **Persistence**: save dir = a world's life. Atomic checkpoints (tmp + rename, keep last 3): `world.npz` (blocks, regrowth heap, tick, rng states) + `entities.json` + `brains/agent_*/` (weights, optimizer, normalizers, replay buffer). Append-only `events.ndjson` / `metrics.ndjson`. Crash recovery = resume `latest`. Dead agents' brains are archived, not deleted (research artifact).

## Entities & physics

- **Robot**: `id, pos, yaw, vel, energy, integrity, held, dormant, age_ticks, brain_name, rng_seed`. Body = 0.8×0.8×0.9-block AABB; `BodySpec` dataclass carries tunables (rays, FOV, speed, energy caps) so variants are config, not code.
- **Physics**: gravity + axis-by-axis AABB-vs-voxel sweep (`move_and_collide`). Auto-climb 1-block steps (energy surcharge); 2+ blocked; falls >3 blocks damage integrity; water halves speed, triples drain. Touch flags feed proprioception.
- **Energy**: basal drain + movement/dig/climb costs; eating restores. Below `brownout_threshold`, actuation (speed, turn) fades linearly to `brownout_floor` at zero — a starving body sags, so depletion is felt in the body's own dynamics before stasis; costs still charge the commanded effort, so a browned-out robot pays full price for less motion. Energy ≤ 0 → **hibernate** (dormant, slow integrity decay). Two ways back: a solar trickle recharges dormant bodies in daylight (wake at `wake_energy`), or a peer feeds them — `place` while holding food and facing a dormant body transfers the meal (`feed` event; the world's first prosocial affordance — and it works with toxic food too, so rescue and murder share a verb). Integrity 0 → **death**, drops SCRAP (death feeds the world).
- **Poison**: `toxic_fraction` of bushes are BUSH_TOXIC (purple; a distinct ray class). Eating one gives reduced energy but costs integrity, fires `took_damage`, and emits a hurt cry — avoidance must be learned from consequence. `ecology.toxic_mimic` ablation makes toxic bushes visually identical to ripe ones (consequence + place memory only).
- **Fatigue**: 0..1 homeostat in proprio. Builds while driving, clears while still (or dormant); past `exhaustion_threshold` energy costs multiply and integrity bleeds. No hardcoded sleep — night scarcity plus fatigue should make resting at night *emerge*, or not (that's the experiment).
- **Involuntary sounds**: death leaves a loud transient cry at the spot (~2 s, pattern (-1,-1) on the signal channel); fall damage a quieter distinct one. World physics, not vocabulary — agents can mimic them, and witnesses get cause-and-effect material (sound → body stops → scrap). Transient sounds checkpoint with the world.
- **Population**: respawn budget, not evolution. `target_population`; on death, delayed respawn at a spawn pad with a fresh brain; `inherit_weights: random_living` flag enables cultural-transmission experiments (research question 4). No fitness scoring anywhere.

## Sensing/action contract — `interface.py` (the stable wall between world and brains)

```python
Observation (TypedDict):                # OBS_VERSION 3: color vision + gaze
  rays:    float32 (R, 8)    # depth + shaded RGB + 4-way hit-kind one-hot (block/robot/dormant/none).
                             # Block identity is carried only by color (palette × face shade ×
                             # per-voxel grain × daylight); misses see the sky. Default R=144
                             # (6 pitch rows +30..-50° × 24 over 160°, range 32). Stage 2 option:
                             # full RGB-D camera image + CNN encoder, behind a flag, at cloud scale.
  proprio: float32 (17)      # body-frame vel, yaw sin/cos, energy, integrity, held, touch(4),
                             # light, fatigue, gaze pitch/yaw
  sound:   float32 (4)       # distance-weighted neighbor signals + world cries (r=12), bearing of loudest
  events:  float32 (4)       # ate, took_damage, dig_success, bumped_robot

Action (frozen dataclass):
  drive:   float32 (2)       # forward ∈[-1,1], turn ∈[-1,1]
  gripper: int               # 0 noop | 1 dig/grab | 2 place/drop | 3 eat/use
  signal:  float32 (2)       # broadcast on sound channel
  gaze:    float32 (2)       # head pitch/yaw targets ∈[-1,1] × body gaze range; None = straight.
                             # Eyes look, arms reach: the gripper stays on the body heading.
```

Appearance is one definition (`blocks.py`): palette color, per-face shade, deterministic
per-voxel luminance grain (a pure hash of position — texture, not noise), and a daylight
factor, shared by ray sensing and the Rerun mesher, so the viewer sees what agents see.

Raycasting: Amanatides–Woo voxel DDA, vectorized across all rays of all agents in one numpy call; entity hits via neighbor pass. Contract versioned (`OBS_VERSION` checked on brain checkpoint load).

## Brains — `packages/brains/gol_brains/`

**Interface** (`base.py`): `act(obs) -> Action` (records transition), `learn() -> metrics|None` (one bounded step), `introspect() -> dict` (curiosity, pred error, value, reward components → observability), `state_dict/load_state_dict`. `registry.py` maps YAML `kind:` → constructor; mixed populations are one config list.

**Scripted baselines** (`scripted.py`): `RandomWalkerBrain`, `ScriptedForagerBrain` (seek ripe-red rays by nearest palette chroma — brightness-invariant, so shading/grain/daylight don't fool it —, eat, avoid water, idle at night). In-world control group + economy calibration probes (tune so forager thrives, walker dies).

**DreamerBrain** (`dreamer/{networks,rssm,buffer,brain}.py`) — full DreamerV3 recipe, implemented in-repo (reference: danijar/dreamerv3, NM512/dreamerv3-torch), sized to our low-dim obs:

- **Encoder**: rays flattened + proprio/sound/events → MLP → embedding. (No CNN needed for ray obs; the ray-grid camera option reuses a small CNN encoder behind a flag.)
- **RSSM** (yes, the real thing): deterministic GRU path + **categorical stochastic latents** (default 24×24 one-hots, scalable), KL balancing with free bits, unimix. Model sizes as named presets `nano/small/base` (~4M / ~12M / ~30M params) — `local_m1.yaml` uses nano, cloud uses small/base.
- **Heads**: decoder (ray depths via symlog MSE, ray classes CE, proprio symlog MSE — raw-ray reconstruction keeps errors visualizable), reward head (twohot symlog), continue head (predicts hibernation risk, not episode end — there are no episodes), value + actor heads.
- **Actor-critic trained in imagination** (DreamerV3 standard: horizon 15, λ-returns, twohot critic with EMA regularizer, entropy bonus, return normalization by percentile). With the full RSSM + stabilizer stack, imagination is the proven path. A `replay_ac` ablation flag keeps the model-free-on-latents alternative available for research comparison.
- **Curiosity — Plan2Explore, not raw prediction error**: ensemble of K=8 small MLPs predicting the next stochastic latent; intrinsic reward = ensemble disagreement (variance). This is the established fix for the noisy-TV trap (raw prediction error makes agents stare at unpredictable things — like other robots — forever). Total reward = `w_c · disagreement + w_h · homeostasis` (ate/damage/low-energy from `events`/`proprio`), both terms normalized (RunningMeanStd) with configurable weights. Curiosity-target masking of other-agent ray classes stays available as an ablation for research question 2.
- **Replay buffer**: per-agent ring, ~500k steps quantized (~50–100 MB at ray-fan size); samples sequences (batch 16 × length 64). Long buffer = ballast against nonstationarity.
- **Lifelong specifics**: train on one unbroken sequence stream (no episode boundaries — DreamerV3 is already off-policy sequence-chunk training, which suits this perfectly); LayerNorm everywhere per the recipe; monitor for plasticity loss (pred-error trend per agent is a first-class logged metric).
- **Cadence**: configurable train-ratio (updates per act-step), throttled by the learner thread. VRAM check: 16 agents × ~12M params × Adam ≈ ~4 GB + activations — fits a 4090 with room; batched multi-agent training via `torch.func.functional_call` + vmap over stacked per-agent params is the M4 perf lever.

## Runtime — `packages/runtime/gol_runtime/`

Files: `run.py`, `loop.py` (SimLoop), `scheduler.py` (LearnerThread), `checkpoint.py`, `config.py`, `control.py` (HTTP control API), `inspect.py`. Single process, three threads:

1. **Sim thread**: fixed timestep, 20 ticks/s sim time, wall-paced × speed multiplier (`--headless --ticks N` unpaced). Brains act every 5 ticks (4 Hz): one vectorized raycast for all agents, then per-agent `act()` (RSSM posterior step) under that brain's lock. Inference device per config (CPU for nano local; CUDA batched in cloud).
2. **Learner thread**: round-robins `learn()` across brains on the training device (MPS locally — benchmark fallback CPU; CUDA in cloud). **Backpressure rule: the learner skips, the sim never waits.**
3. **Observability thread**: Rerun logging + control HTTP endpoint (asyncio).

Checkpoint every 30k ticks and on SIGINT, at a tick boundary — world + all brains saved together atomically so resume is coherent. `--sync` mode (learning inline) for determinism tests.

**Cloud workflow** (`scripts/provision_runpod.sh` + `docs/architecture.md` section): rsync repo + save dir to a RunPod/Vast box, `uv sync`, `gol-run --resume saves/alpha` under tmux; Rerun streams to the laptop (`rr.connect_grpc()` over SSH tunnel) or records `.rrd` files synced back. Checkpoints rsync home on cadence — a killed spot instance costs at most one checkpoint interval.

## Observability — `packages/obs/gol_obs/`

Files: `rerun_log.py`, `metrics.py`, `events.py`, `export.py`.

- **Rerun scene** (~10 Hz + on-change): voxel terrain as per-chunk meshes (greedy-meshed, re-logged only for dirty chunks), robots as colored boxes with labels/energy bars, held items, selected agent's ray fan as line strips colored by hit class, spatial visit heatmap as an image overlay, day/night as ambient light scalar.
- **Rerun time-series per agent**: energy, integrity, curiosity (disagreement), pred error, value, reward components, action histogram — the "is it alive and learning" dashboard, scrubbable over the whole run.
- **Recording**: `.rrd` files rotated per N sim-hours — replays for free, no custom replay format.
- **Analysis logs** (source of truth, tool-agnostic): `metrics.ndjson` (per-agent every 100 ticks), `events.ndjson` (eat/dig/place/death/spawn/contact/signal-burst with positions) → `gol-stats` for emergence hunting (interaction graphs, congregation detection, signal-usage stats). Optional W&B/TensorBoard export behind a flag for long cloud runs.
- **Control**: `gol-ctl pause|resume|speed 4|checkpoint|population 12` → HTTP to the run's control endpoint.

## Milestones (each ends with the world running + observable; no infrastructure-only phases)

- **M0 — World renders.** pyproject, configs, blocks/grid/terrain/persistence, minimal loop (regrowth + day/night), Rerun terrain + timeline. *Demo: `gol-run --new saves/alpha` → world in Rerun, light cycles, bushes regrow, save/resume works.*
- **M1 — A body lives in it.** Physics, raycasting, obs/action contract, RandomWalkerBrain, one robot, `gol-ctl` pause/speed, determinism tests. *Demo: robot wanders, climbs, falls, splashes; its ray fan renders.*
- **M2 — An ecology.** Energy economy, gripper, hibernate/death/scrap, respawn, ScriptedForagerBrain, 8 scripted bots, metrics/events + per-agent Rerun charts. *Demo: foragers survive, walkers starve, population churns. Economy calibrated here.*
- **M3 — One mind awakens (local).** Full DreamerBrain (RSSM + Plan2Explore + imagination AC) for 1–2 agents among scripted peers on the M1 Pro (nano preset); sync mode first, then LearnerThread; joint checkpointing; introspection charts. *Demo: kill and resume — learning visibly continues; pred error trends down; behavior differs from the random walker within hours.*
- **M4 — A living population (cloud).** Provisioning script, CUDA path, batched/vmapped multi-agent training, 12–16 learners, perf floor test, soak.sh, `.rrd` recording rotation. *Demo: multi-day cloud run streamed to the laptop; scrub a day of world-time in Rerun.*
- **M5 — Social pressure & research runs.** Signal channel live, ore/dig-place material loop, night-scarcity tuning, `inherit_weights`, curiosity-masking ablation configs, `docs/journal.md` findings log. *Demo: designed experiment runs targeting research questions 2–4; evidence (or absence) of signaling/congregation/hoarding in events + heatmaps.*

## Verification

- **Unit tests** per package: terrain determinism; table-driven physics (wall stop, 1-block climb, 2-block block, fall damage, water); raycasts vs hand-built 8×8×8 scenes; energy accounting; save→load→save round-trip equality; Brain contract test against every registered brain; buffer quantization bounds.
- **Dreamer component tests**: RSSM shape/KL sanity, symlog/twohot invertibility, imagination rollout shapes, ensemble disagreement > 0 on random data and → 0 on constant data, one overfit-a-tiny-sequence convergence test.
- **Determinism**: `--sync` world hash equal across two 2k-tick same-seed runs.
- **Smoke** (pytest `slow`): headless 5k ticks, mixed population — no exceptions, population > 0, logs written, checkpoint completes, resume runs 500 more.
- **Learning sanity probe** (observational, in-world — not a benchmark): after N wall-hours, a Dreamer agent's pred error declines and its food-finding rate exceeds RandomWalker's. Run via `gol-stats compare`.
- **Perf floor**: headless 1k ticks with 16 agents beats a configured ticks/s.

## Risks & mitigations

1. **DreamerV3 implementation complexity** (the big one). Mitigate: follow the published recipe exactly (it's engineered for fixed-hyperparameter robustness), port against dreamerv3-torch as reference, component unit tests, M3 gate = single agent visibly learning before any scale-up.
2. **Lifelong/nonstationary collapse** (plasticity loss, forgetting over multi-day runs). Mitigate: long replay ballast, LayerNorm-everywhere recipe, pred-error trend as a first-class monitored metric, scripted control group distinguishes "world broken" from "brain broken".
3. **Curiosity pathologies**. Mitigate: Plan2Explore disagreement (epistemic, noise-robust) instead of raw prediction error; normalized+clipped reward mixing; masking ablation flag; per-agent curiosity time-series to spot fixation.
4. **MPS gaps / local tier too slow**. Mitigate: nano preset sized for it, CPU fallback benchmarked, and the tier plan already assumes local = 1–2 agents only.
5. **Cloud cost creep**. Mitigate: spot instances + checkpoint-rsync discipline (max loss = one interval), $/run estimates in docs, population/model-size presets per budget, soaks are resumable so nothing is wasted.
6. **Drift back to task-training** (killed v1). Mitigate: milestone rule, no episodic concepts in code, invariants + research questions in CLAUDE.md.

## Implementation scope for this session

Initialize the repo (git, uv, pyproject, tooling, CLAUDE.md with invariants + research questions, docs/architecture.md + docs/research-questions.md distilled from this plan) and build **M0 end-to-end**, then continue into M1 as far as the session allows. Each milestone is verified by its demo + tests before moving on.

---

## Implementation deviations (as built, 2026-07-05)

The build followed this plan with four evidence-driven changes:

1. **Plan2Explore ensemble predicts the next observation *embedding***, not the
   next stochastic latent — closer to the original paper, and it enables the
   `curiosity_mask_agents` ablation (other robots erased from the curiosity
   target).
2. **`inherit_weights: lineage` added and made the default**: a learning brain
   (weights + replay) survives its body's death and continues in the respawn;
   only the recurrent state resets. Motivated by newborn dreamers starving
   before learning could accumulate. `none` / `random_living` remain for the
   cultural-transmission experiments.
3. **Warmup cut 2000 → 500 act-steps** (same starvation reason).
4. **Devices by benchmark (M1 Pro)**: nano learns fastest on cpu
   (474 vs 568 ms/update); small+ wins on mps (655 vs 1009). Learning brains
   live wholly on `devices.learning`. Batched multi-agent training via
   torch.func/vmap over stacked per-agent params remains the future perf lever;
   sequential round-robin is comfortably within budget on a 4090 at 16 agents.
