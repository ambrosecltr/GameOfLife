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
- **Brains**: a Dreamer-style learned world model and critic per organism, with
  endogenous affect and an optional learned temporal manager/worker controller. No
  task reward, named skill, demonstration, pretrained behavior, or designer fitness
  score. The architecture is a research variable; the closed anima track established
  that a learned expectation/critic is constitutive for valence over a mortal life.
- **Compute tiers**: develop and validate locally; select the long-run pod only after
  the synchronized five-minute gate on RTX 4090, RTX 5090, and H100 SXM. The 5090 is
  the first Aion candidate when 32 GB fits; H100 bounds throughput. L40S/A100 enter
  only when measured memory pressure justifies them. Pod prices are benchmark inputs,
  not hard-coded assumptions. Multi-GPU runs assign independent brains per device and
  need no distributed gradient system.
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
│   ├── brains/gol_brains/      # Brain interface; scripted, dreamer/, aion/, plastic/
│   ├── runtime/gol_runtime/    # persistent loop, scheduler, checkpointing, CLI, control API
│   └── obs/gol_obs/            # Rerun logging, metrics/events writers, replay export
├── scripts/                    # soak.sh, cloud provisioning (provision_runpod.sh)
├── docs/                       # architecture.md, research-questions.md, research_journal/
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
- **Physics**: gravity + axis-by-axis AABB-vs-voxel sweep (`move_and_collide`). Auto-climb 1-block steps (energy surcharge); 2+ blocked; falls >3 blocks damage integrity. Water speed and drain are independently configurable; the measured substrate uses 0.5× speed and 1.75× drain. Touch and submersion feed proprioception.
- **Energy**: basal drain + separately measured forward, turn, signal, dig, climb, water, repair, and reproduction costs; eating restores. Below `brownout_threshold`, actuation (speed, turn) fades linearly to `brownout_floor` at zero — a starving body sags, so depletion is felt in the body's own dynamics before stasis; costs still charge commanded effort. Energy ≤ 0 → **hibernate** (dormant, slow integrity decay). Solar recharges a dormant body only to a functional wake floor below the repair threshold, while a peer can feed it directly. Integrity 0 → **death**, drops SCRAP (death feeds the world).
- **Poison**: `toxic_fraction` of bushes are BUSH_TOXIC (purple; a distinct ray class). Eating one gives reduced energy but costs integrity, fires `took_damage`, and emits a hurt cry — avoidance must be learned from consequence. `ecology.toxic_mimic` ablation makes toxic bushes visually identical to ripe ones (consequence + place memory only).
- **Fatigue**: 0..1 homeostat in proprio. Builds while driving, clears while still (or dormant); past `exhaustion_threshold` energy costs multiply and integrity bleeds. No hardcoded sleep — night scarcity plus fatigue should make resting at night *emerge*, or not (that's the experiment).
- **Involuntary sounds**: death leaves a loud transient cry at the spot (~2 s, pattern (-1,-1) on the signal channel); fall damage a quieter distinct one. World physics, not vocabulary — agents can mimic them, and witnesses get cause-and-effect material (sound → body stops → scrap). Transient sounds checkpoint with the world.
- **Population**: developmental lineage continuity, distinct learned descendants,
  or earned budding. `lineage` reincarnates one learning object across bodies;
  `descendant` copies the dead parent's learned substrate into a distinct newborn.
  In budding mode a physiologically thriving body pays energy/integrity to create
  a child carrying mutated heritable state; no fitness score ranks organisms. A
  small extinction floor is an experimental safeguard, not selection.

## Sensing/action contract — `interface.py` (the stable wall between world and brains)

```python
Observation (TypedDict):                # OBS_VERSION 5: color, gaze, senescence, water
  rays:    float32 (R, 8)    # depth + shaded RGB + 4-way hit-kind one-hot (block/robot/dormant/none).
                             # Block identity is carried only by color (palette × face shade ×
                             # per-voxel grain × daylight); misses see the sky. Default R=144
                             # (6 pitch rows +30..-50° × 24 over 160°, range 32). Stage 2 option:
                             # full RGB-D camera image + CNN encoder, behind a flag, at cloud scale.
  proprio: float32 (19)      # body-frame vel, yaw sin/cos, energy, integrity, held, touch(4),
                             # light, fatigue, gaze pitch/yaw, senescence, in-water
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
- **Heads**: decoder (ray depth/RGB/class and proprioception), a diagnostic twohot
  affect head, and a continuation head trained on real terminal states delivered by the
  runtime. There are no episode ends; continuation represents physical mortality.
- **Actor-critic trained in imagination**: horizon rollouts, λ-returns, twohot value
  distributions with EMA targets, entropy, and percentile return scaling. The organism
  path uses six value channels (comfort, viability, curiosity, boredom, predicted
  mortality, controllability) so unlike signals remain learnable until the actor combines
  them. Historical configs contain an inert `replay_ac` key; replay actor-critic was never
  implemented and is not a supported ablation.
- **Temporal skills** (`temporal_skills:`): a manager selects an unnamed latent intent
  every N actions; a worker turns intent plus world-model state into motor commands. A
  displacement discriminator supplies a self-generated controllability signal. No skill
  has a label, demonstration, or pretrained behavior.
- **Curiosity — learning progress over self-organized regions** (`reward.curiosity: lp`, Oudeyer-style; `dreamer/interest.py`): the Plan2Explore ensemble still measures per-sample model error, but the reward is the *rate that error falls* per region — fast/slow error EMAs per region, their gap (relative to the slow level) as LP. Regions are online k-means centroids in RSSM feature space (`lp.partition: latent`; what counts as "an activity" is carved by the agent's own representation) or animate-presence buckets (`kind`, ablation), and LP is a pure function of region index so imagination can query it. Mastered regions (error low, flat) and unlearnable ones (noise, other minds — the noisy-TV trap) both yield zero progress, so attention lives at the learnable frontier and moves on; path-dependence turns tiny early differences into individual interests. A `mix_disagreement` trickle keeps newborns moving before regions have history. Legacy disagreement-level curiosity remains as `curiosity: disagreement` (ablation), and curiosity-target masking of other-agent rays stays available for research question 2. Total reward = `w_c · curiosity + w_h · homeostasis − boredom`, terms normalized (RunningMeanStd) with configurable weights.
- **Homeostasis and viability**: the world model predicts proprioception, then the
  organism path directly evaluates drive reduction, standing deficit, a log barrier to
  bodily boundaries, and predicted cessation. A learned reward head remains a diagnostic
  ablation. These equations describe bodily consequence, never a desired action.
- **Boredom** (`reward.boredom`): a slow mood charged by chronologically lived calm and
  low stimulation. Replay updates its stimulation estimate but cannot advance mood out of
  order. Imagined dull/safe futures are then priced by the current pressure.
- **Temperament** (`temperament:` in the brain config): innate, heritable individuality — log-normal multipliers over the abstract drive knobs only (curiosity/homeostasis weights, per-drive weights, boredom weight, entropy scale), sampled at birth, persisted in checkpoints, and mutated when a newborn warm-starts from a living donor (`Brain.inherit`). Never object-specific: an agent may be born motion-hungry or damage-averse, but *what* it comes to like must emerge from its own history. Observability: `gol-stats --interests` computes per-agent activity profiles (rest/social/forage/dormant + eat/dig/place rates from `near_robots`/`near_bushes`/`resting` metrics fields), between-agent divergence (individuality) and within-agent stability (interests) — the measurement side of research questions 2 and 4.
- **Replay buffer**: per-agent ring, ~500k steps quantized (~50–100 MB at ray-fan size); samples sequences (batch 16 × length 64). Long buffer = ballast against nonstationarity.
- **Lifelong specifics**: train on one unbroken sequence stream (no episode boundaries — DreamerV3 is already off-policy sequence-chunk training, which suits this perfectly); LayerNorm everywhere per the recipe; monitor for plasticity loss (pred-error trend per agent is a first-class logged metric).
- **Cadence**: configurable train ratio is checkpointed learning credit per lived
  act-step. A measured wall-clock governor slows world execution before causal lag
  exceeds its configured bound. Credit dropping is an explicit, metered ablation.
  Independent learners can be assigned round-robin to `devices.learning: [cuda:0,
  cuda:1, ...]`; there is no gradient all-reduce because organisms share no model,
  optimizer, replay, or statistics. Cross-brain batching/vmap remains a profiler-gated
  option, not an assumed optimization.

**AionBrain** (`aion/{brain,s5}.py`) is a separate checkpointed world-model
lineage. It retains the organism mechanisms above but replaces the GRU transition
with stable continuous-time S5 blocks. Live action and imagination use one recurrent
transition per subjective step; replay uses an associative scan over 1,024-step
contexts. A wake is distinct from a new life: fast sensorimotor modes reset while
slow modes persist and decay across the runtime-measured blackout duration. See
[proposal 005](research_proposals/005-aion-s5.md) for the architecture boundary,
falsification gates, and the limits of treating predictive S5 state as memory.

S5 is stored and evaluated as paired FP32 real channels. The transition is the exact
diagonal complex algebra (`a·real − b·imag`, `b·real + a·imag`) without native
complex kernels. Decay/frequency/log-step, discretization, wake powers, persistent
state, scan, and B/C projections are protected from BF16 autocast. This matters at
the configured slow edge: `exp(-0.5 × 0.0001) ≈ 0.9999500012` rounds to 1.0 in
both FP16 and BF16, erasing the intended `A^1024 ≈ 0.95009` decay. Dense encoder,
heads, gates, actor, critic, and ensemble may use BF16 AMP; protected FP32 projection
GEMMs may use TF32. Parameters and optimizer states stay FP32.

## Runtime — `packages/runtime/gol_runtime/`

Files: `run.py`, `loop.py` (SimLoop), `scheduler.py` (LearnerThread),
`governor.py`, `config.py`, `control.py` (HTTP control API), and `inspect.py`.
Single process, one sim thread plus one serial learner worker per learning brain:

1. **Sim thread**: physics always advances by the same fixed 1/20-second timestep,
   and brains act every five ticks. Fixed pacing targets the configured wall rate;
   adaptive pacing targets the minimum measured learner, inference, world, and
   configured capacity with headroom/hysteresis. `--headless` removes viewer pacing,
   not causal backpressure.
2. **Learner workers**: each pays checkpoint-derived update credit serially while
   sibling brains run independently. At the causal-lag ceiling the sim waits in wall
   time; virtual state does not advance and credit is not lost. Async controllers read
   immutable snapshots published after a fixed number of completed updates.
3. **Observability thread**: Rerun logging + control HTTP endpoint (asyncio).

When every embodied organism is truly dormant, required whole-update credit is paid
first. Ordinary stepping then runs unpaced. Settled, event-free intervals can jump to
the tick immediately before the earliest ecology, spoilage, sound, lifecycle, wake,
death, metrics, checkpoint, or run-end boundary; the boundary itself uses scalar
`World.step()`. Bulk accounting advances solar charge, hibernation damage, fatigue,
age, heatmap visits, and Aion's missed opportunities without consuming RNG.

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
- **M4 — A living population (cloud).** Provisioning script, explicit CUDA
  precision, measured causal governor, independent multi-GPU assignment, profiler-gated
  batching/vmap, perf floor tests, soak.sh, and `.rrd` rotation. *Demo: multi-day
  cloud run streamed to the laptop; scrub a day of world-time in Rerun.*
- **M5 — Social pressure & research runs.** Signal channel live, ore/dig-place material loop, night-scarcity tuning, `inherit_weights`, curiosity-masking ablation configs, `docs/research_journal/` findings log. *Demo: designed experiment runs targeting research questions 2–4; evidence (or absence) of signaling/congregation/hoarding in events + heatmaps.*

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
   Proposal 006 later added `descendant`: the same learned substrate can cross
   a bodily generation without treating the newborn as the same mind.
3. **Warmup cut 2000 → 500 act-steps** (same starvation reason).
4. **Devices by benchmark (M1 Pro)**: nano learns fastest on cpu
   (474 vs 568 ms/update); small+ wins on mps (655 vs 1009). Learning brains
   live wholly on `devices.learning`. Batched multi-agent training via
   torch.func/vmap over stacked per-agent params remains the future perf lever;
   sequential round-robin is comfortably within budget on a 4090 at 16 agents.
