# Training ops: speed, pacing, and the offline screen

How to budget, pace, and instrument a round. Written for the agent staging the
next run, local or cloud. The learning code this describes is the post-swift
core on `main` (journal 010 has its provenance; this doc is only how to use it).

## The pacing model

`training.train_ratio` is a fixed scientific budget of optimizer updates per
eligible lived act-step. Warmup experience creates no update credit. Thereafter
credit is derived from checkpointed lived steps, completed updates, the target
ratio, and any explicitly dropped credit. Restarting the process cannot erase or
duplicate debt.

The adaptive controller uses the measured identity:

    safe ticks / wall second =
      awake-brain learner updates / wall second × act_every
      ----------------------------------------------------
          sum(train_ratio for those awake learning brains)

`pacing.headroom` is then applied, and measured world/inference capacity can
become the tighter limit. Because optimizer capacity is not transferable
between independent organisms, the slowest per-brain causal rate is also a
ceiling; a fast sibling cannot hide another brain's debt. Physics remains one
fixed 1/20-second transition per tick; brains still receive one action
opportunity every `act_every` ticks;
replay shape, optimizer-step meaning, publication cadence, and train ratio do
not change. Faster hardware shortens wall time.

`pacing.debt_policy: backpressure` is the research default. At
`max_debt_updates` the world pauses in wall time until debt falls to
`resume_debt_updates`; no virtual event occurs during the wait. `drop` is an
explicit throughput-ablation mode: excess credit is discarded and both
`dropped_update_credit` and per-brain totals are logged. It is never a hidden
fallback.

Runtime metrics include `safe_ticks_per_second`,
`actual_virtual_ticks_per_second`, measured learner capacity, maximum debt,
inference deadline misses, precision, and the limiting subsystem. Per-brain
metrics include `pending_update_credit`, `dropped_update_credit`,
`inference_lag_updates`, `learn_seconds`, and `train_ratio_eff`. Async inference
publishes every fixed `publish_every` completed updates; the exact published
snapshot and its update number are checkpointed.

`--headless` removes viewer pacing, but causal backpressure still applies.
Adaptive mode still targets the measured safe rate. `--sync` remains the
deterministic inline-learning path and also consumes only earned update credit.
In adaptive mode `gol-ctl speed` may slow the controller below safe capacity;
values above 1 cannot override the measured safety ceiling.

## Precision policy

`training.precision` has three fail-closed modes:

- `ieee_fp32`: FP32 reference, TF32 disabled;
- `tf32`: FP32 tensors and optimizer state, TF32 enabled for eligible CUDA real
  GEMMs;
- `amp_bf16`: BF16 autocast for eligible forward/loss dense computation, FP32
  master parameters and optimizer state, plus TF32 for remaining eligible FP32
  GEMMs. Backward is outside autocast and BF16 uses no GradScaler.

TF32 controls are process-global, so a runtime validates every CUDA learning
brain before construction. Mixed brains may use `tf32` and `amp_bf16` together
because both request the same TF32 posture; mixing either with `ieee_fp32` in
one process is rejected. Requested CUDA BF16/TF32 modes fail if the device lacks
the capability; there is no silent FP32 fallback.

Aion's recurrence is a protected exception. Decay, frequency, log step,
continuous eigenvalues, transition `exp`/elapsed-time power, paired-real state,
associative scan, and B/C projections stay FP32. Eligible projection GEMMs may
use TF32. `world_model.s5.projection_precision` currently accepts only `fp32`;
BF16 projection is blocked until target-GPU long-retention and learning-update
parity pass.

## Five-minute target-GPU gate

Run the same checkout on RTX 4090, RTX 5090, and H100 SXM. Add L40S or A100 only
if the 32 GB units do not fit or profiler evidence shows memory capacity, not
compute, is limiting. Supply the current pod price:

```bash
scripts/bench_aion_preflight.sh /tmp/aion-preflight <gpu-hourly-price>
```

The script records GPU/driver/Torch capability, then runs one learner and the
real three-brain contention case in `ieee_fp32`, `tf32`, and `amp_bf16` with
batch 8, 1,024 graded steps, and 256 burn-in. Timers synchronize the device.
The RunPod image must provide GNU `timeout`; the script hard-stops and rejects
the unit when the shared five-minute deadline expires, including mid-command.
Outputs include learn mean/p50/min, action p50/p95/max under concurrent learning,
deadline misses, allocated/reserved VRAM, host peak memory, graded timepoints/s,
cost per million timepoints, and raw plus 0.85-headroom sustainable ticks/s.
The BF16 case also writes a Chrome trace with named replay-transfer, model, S5/scan kernels,
optimizer, inference, world, log, and checkpoint regions where present.

Do not launch a 24-hour run unless all of these hold:

1. BF16 capability checks pass and the reported precision is the requested one.
2. Full Aion replay fits with at least 15% reserved-VRAM headroom.
3. Three-brain sustainable ticks/s exceeds the intended operating rate after
   the configured 0.85 controller headroom.
4. Concurrent action p95 is below the selected act deadline with zero misses
   after warmup.
5. Losses and gradients remain finite in every mode; FP32 is the reference.
6. The profiler identifies no unexplained host transfer, serialization, or scan
   stall. Do not add vmap, streams, pinned transfer, or compilation without this
   evidence.

No CUDA performance result is claimed by the local test suite. Preserve every
pod output with the round journal before choosing the long-run unit.

### Aion 01 two-GPU arm

The two-lineage arm assigns one independent Aion to each GPU through
`devices.learning: [cuda:0, cuda:1]`; it does not shard either organism or
all-reduce gradients. Gate the exact pod with its total hourly cost:

```bash
scripts/bench_aion_2gpu.sh /workspace/aion-2gpu-preflight <pod-hourly-price>
```

The gate runs both full-shape BF16 learners concurrently and requires at least
25 safe ticks/s after 0.85 headroom, zero 250 ms action-deadline misses, action
p95 below the deadline, and at least 15% reserved-VRAM headroom on each card.
The round config is `configs/run/aion_01_2gpu.yaml`; it keeps six bodies as two
Aions plus four scripted foragers. Launch only through
`scripts/start_aion_01_2gpu.sh`, which fails closed unless exactly two
BF16-capable CUDA devices are visible and refuses to overwrite an existing
world.

### What may adapt and what may not

Semantically transparent runtime changes are wall-clock pacing, universal-dormant
event-free jumps, device assignment of independent brains, synchronized
measurement, and the paired-real expression of the same FP32 diagonal algebra.
They preserve the configured virtual schedule and causal work.

Precision mode, S5 projection precision, batch size, sequence/burn-in length,
train ratio, replay sampling, optimizer, publication cadence, imagination
horizon, and any dropped credit are deliberate research variables. The governor
and benchmark never rewrite them in response to a faster GPU. Compilation,
pinned transfer, CUDA streams, inference batching, or vmap may be adopted only
after profiler evidence and parity tests; they are not hidden fallback paths.

## Universal-dormancy acceleration

This optimization applies only when every embodied robot is dormant, including
scripted organisms. An awake resting body still senses and acts and therefore
disables it. Two stages are configured under `dormancy_acceleration`:

1. `exact_unpaced`: once every whole owed learning update is paid, ordinary
   `World.step()` continues without wall-clock sleep. Falling bodies and all
   ordinary event ordering remain scalar and exact.
2. `event_fast_forward`: once all dormant bodies are scalar-normalized, settled,
   non-overlapping, interaction-free, and have no pending grip, bulk-integrate only the interval
   before the next causal boundary. Solar charge, hibernation damage, fatigue,
   held-item age, body age, heatmap occupancy, and missed act opportunities are
   advanced in virtual time.

Jumps stop before wake, integrity death, falling/settling, held-food spoilage,
regrowth/wither/sprout heap work, transient expiration, lifecycle/respawn work,
metrics, checkpoints, the requested run end, or any pending world event. The
boundary itself runs through ordinary scalar stepping, preserving RNG
consumption and same-tick event order. Aion receives the exact count of skipped
act opportunities, then performs its FP32 elapsed-time transition at wake.

If any safety predicate is false, the runtime simply uses exact ordinary steps;
there is no approximate fallback. `metrics.ndjson`, events, and checkpoints
retain virtual tick timestamps. Population death reconciliation runs after every
world tick, and checkpoint serialization waits for any in-flight terminal record,
so a death between act boundaries cannot orphan its lineage or replay evidence.

## Dreamer config flags (brain YAML), semantics and status

All default to legacy behavior; enabling any is a round decision to note in
the journal entry's "What changed."

`replay:`
- `burn_in: N` — each sampled sequence gets N gradient-free prefix steps that
  warm the recurrent state, so `seq_len` can shrink at equal samples/update
  (e.g. batch 32 × seq 32 + burn_in 8 ≈ batch 16 × seq 64). Cuts backward
  cost ~40% and removes the zero-init state at window starts. Used by
  swift_01; not yet validated at base scale.
- `recent: N` — N batch rows pinned to the newest experience window
  (online-queue mixing); the rest sample uniformly over the whole life.
- `prioritize: none | reward` — reward-aware replay (round 009's reachability
  fix). `reward` draws `prioritize_rows` (default batch/4) rows from windows
  containing a *salient* step. Salience = |realized drive reduction| recorded
  at every act() — **not** event flags: under HRRL a meal at satiety is an ate
  event worth exactly zero (measured), and a future priced blackout is a
  salience spike with no event at all. Threshold `prioritize_threshold`
  (default 0.1, same bar as `homeo_spike_frac`). No salient steps lived yet →
  rows fall back uniform. Changes what is learned from, never what is
  rewarded. Screen it offline before spending pod hours (below).
- Salience is recorded even when prioritization is off, rides checkpoints,
  and is backfilled from stored proprio when loading blobs that predate it.
- `prioritize_rows` gets meals into batches; whether the twohot head then
  *fits* rare spikes is open — in the only screen so far the spike error
  barely moved with spikes in every batch (tiny buffer, weak evidence). If
  that replicates on a long life, the follow-up knob is spike-weighted loss.

`reward:`
- `imagined_homeostasis: head | proprio` — `head` preserves the historical
  learned twohot affect head. `proprio` applies the configured comfort and
  viability equations directly to predicted before/after proprioception during
  imagination; the head is still trained and logged as a diagnostic. Use
  `proprio` when the body function is already known and the research question is
  behavior, not whether another network can rediscover that function.
- `blackout: cut | priced` — how the dormancy gap enters the learned stream
  (round 011's reachability fix). `cut` (legacy) severs it: wake resets the
  live state and the salience chain, and near-zero energy trains the
  continuation head to 0, so imagination discount-terminates at the crash.
  `priced` makes the blackout one visible transition — the pre-collapse step
  stays the predecessor of the wake observation, the gap's real
  energy/integrity delta lands as wake-step salience (so reward-aware replay
  can find every blackout), the live recurrent state still resets (the mind
  was off), and nothing observable terminates (cont trains to 1: a crash the
  actor can't plan across can't be avoided). Requires `homeostasis: drive`.
  Respawn into a new body remains a hard cut in both modes; death stays
  unexperienced.
- `spike_loss_weight: w` — multiplies the twohot reward-head loss by
  (1 + w) on |reward| > `prioritize_threshold` samples. The pre-registered
  follow-up if prioritized replay gets spikes into batches but the head
  still can't fit them. 0 = legacy unweighted loss.
- `viability:` (round 012, proposal 003) — an *additive* second homeostatic
  term: a log-barrier on distance to the LETHAL floor (energy→dormancy,
  integrity→death), where the comfort drive is convex distance to a comfort
  SETPOINT. `scale` is its HRRL weight (0 = off, recovers beta_10 exactly);
  `floor` a standing danger-zone tax on `V`; `{energy,integrity}_safe` the
  margins above the floor where `V` is 0; `{energy,integrity}_lethal` the
  floors (default 0); `barrier_cap` clamps each −log component and optional
  `total_cap` clamps their weighted sum. Priced through the same reward head as the comfort drive (their
  sum is what the actor maximizes) but logged separately: `reward_viability`,
  `viability_level`, `viability_max`, and per-life `life_return_via` /
  `life_return_homeo` (exact realized return, accumulated in `act()`).
  Requires `homeostasis: drive`. Near-death moments also become salient to
  prioritized replay: the barrier's salience is `|scale·ΔV| + floor·V`, so the
  standing tax carries the priority in the floor-only form (deltas are small
  when the drift toward the floor is slow). Motivation: the comfort drive
  telescopes to a *negative*
  return over any mortal life, so the reward gives no positive stake in
  survival; the barrier's marginal value explodes toward the floor without
  telescoping to a loss.
- `death_terminal: true` — true death (integrity → `integrity_lethal`)
  terminates the imagined stream (cont target = integrity above the floor), so
  its absorbing ~0 return backs up through the critic: a functional fear of
  death from prediction. Recoverable dormancy stays non-terminal. Independent
  of `blackout`/`viability` (ablatable alone). false = beta_10. The terminal
  targets come from the runtime death hook: a dying body is never observable
  from inside (dormant bodies don't act; the death tick removes the robot
  before sensing), so on death the scheduler hands the body's last observation
  to its brain (`Brain.record_death`, non-blocking — the sim never waits) and
  the brain records it with vitals at the floor. Lineage runs only: the record
  needs a replay buffer that outlives the body.
- `terminal_loss_weight: w` — multiplies continuation BCE on physical terminal
  rows. Round 012 had only 1–15 terminal rows among thousands and learned
  `cont=1` at death; use `cont_terminal`, `cont_alive`, and `terminal_frac` to
  verify separation. Values below 1 are rejected.
- `fear_weight: w` — adds `w · log P(continue_next)` to imagined affect. It is
  zero when continuation is certain and increasingly negative when the learned
  world predicts cessation; it names no action or task.
- `boredom.gate: drive | viability` — what "calm enough to be bored" reads.
  `drive` (beta_10) reads the comfort drive, so any deficit below setpoint
  shuts the boredom gate (couples boredom to hunger — the round-011 concern).
  `viability` reads the barrier, so an agent far from the lethal floor can be
  bored while merely peckish; only true danger shuts the gate.

`training:`
- `precision: ieee_fp32 | tf32 | amp_bf16` — explicit policy described above.
  Precision is recorded in checkpoints, benchmark output, runtime status, and
  metrics. Changing it is a deliberate research-variable change.
- `optimizer: adam | muon` — Muon (vendored, `dreamer/optim.py`) on the
  world model's 2D matrices, Adam elsewhere. `muon_lr` (default 0.02).
  Staged ablation; not yet run in any round.
- `l2_init: w` — L2-toward-init on the world model (plasticity maintenance
  for one unbroken life). Staged ablation; not yet run in any round.
- `compile: true` — torch.compile on the RSSM step functions. Measured
  neutral on cpu; untested on cuda. Adds one-time compile stalls at brain
  construction (act path is pre-warmed; learn shapes compile on the learner
  thread; checkpointed credit remains owed during the stall).
- `async_inference: true` — publish immutable encoder/RSSM/controller snapshots
  so `act()` never shares parameters with the optimizer and the learner need not
  hold the population action lock. `publish_every` controls update cadence.
  Checkpoints serialize both training weights and the exact published snapshot
  against the brain's internal learning lock. This
  mode is intentionally incompatible with `compile` until compiled-module
  snapshots have their own verified path.

`actor_critic:`
- `vector_critic: true` — learn separate twohot values for comfort, viability,
  curiosity, boredom, mortality risk, and skill controllability, then sum only
  for the actor's normalized advantage. Metrics expose `value_*`, `return_*`,
  and `affect_*` for every channel.

`temporal_skills:`
- `enabled: true` replaces the flat actor with an unnamed manager/worker
  hierarchy. `num_skills` is the latent vocabulary, `duration` the number of
  real and imagined actions each selection persists, `intrinsic_weight` the
  variational controllability signal, and `manager_entropy` its exploration
  pressure. The discriminator reads latent displacement, not absolute place.
  Monitor usage entropy, discriminator accuracy/loss, manager entropy, switches,
  and per-skill action/consequence profiles. No index has built-in semantics.

New metrics these produce: `loss_reward` (always), `reward_head_spike_err`
(|predicted − realized| on spike samples — the reachability gauge),
`spike_row_frac` (what prioritization feeds the head), `l2_init_dist`,
`lp_mix_eff` (0 = trickle annealed, imagination skip active), continuation
separation, per-affect value/return/instantaneous channels, temporal-skill
health, and `stimulation_online`.

## Checkpoint compatibility

Brain blobs from before the speed core load fine: the Plan2Explore ensemble
is migrated to the batched layout, salience is backfilled, and the **model
optimizer's Adam moments reset** (param layout changed; actor/critic moments
survive). For offline analysis that's harmless. For *resuming a live world*
across the boundary, expect a brief model-loss wobble while Adam re-estimates
its moments — note it in the journal if the round's data spans the resume.

Architecture flags must match the checkpoint: enabling temporal skills changes
the actor parameter layout, and enabling the vector critic changes the critic
output shape. Start a fresh brain or use a deliberate migration; do not resume a
flat/scalar brain under those flags.

The original Aion foundation stored S5 B/C matrices as native `complex64`.
Loading that checkpoint into the paired-real implementation performs a one-way,
key-validated split into FP32 real/imaginary parameters. The flat live state is
already compatible. The world-model optimizer restarts because its complex
moments cannot map unambiguously to the new parameter list; actor, critic,
replay, lineage identity, wake state, and published inference state survive.
New paired-real checkpoints carry `aion_s5_format: paired_real_v1`; missing or
mixed format markers fail rather than guessing.

Checkpointed precision must match the resume config. A different precision mode
is a new experimental arm or an explicit future migration, not a transparent
resume setting. The checkpoint also validates train ratio, batch/sequence/burn-in,
warmup, replay sampling, and publication cadence as one learning contract, so
hardware selection cannot silently rewrite the organism's consolidation budget.

## The offline gym: screen learn-side knobs before pod hours

Anything that lives in `learn()`-side statistics (normalizers, LP, boredom
constants, replay sampling, reward-head behavior) can be screened against a
*recorded* life in minutes:

```
uv run python scripts/conditioning_gym.py <blob.pt | save-dir> \
    --brain configs/brain/<candidate>.yaml --fresh-model \
    --updates 2000 --set replay.prioritize=reward --out /tmp/arm_a.ndjson
```

- `--fresh-model` keeps only the stored replay buffer and grows a fresh brain
  over it; the candidate config may be a different preset than the recorded
  brain (the obs contract is shared), so a nano can train on a base life.
  Without the flag it continues the stored mind (config must match preset).
- Run arms with identical seeds and diff the metric trajectories.
- **Open-loop caveat:** the actor never changes what was lived, so
  closed-loop effects are invisible. The gym earns a knob a live round; it
  never replaces one. And a screen is only as good as its data — check the
  buffer actually contains the phenomenon (e.g. salient meals) before
  reading a null result as "knob does nothing."
- It needs `checkpoints/ckpt_*/brains/*.pt`. Local saves have them; **the
  cloud sync scripts exclude brains by default** — pull at least one dreamer
  blob (~561MB at base preset) off the pod while it's alive if the round's
  data should stay screenable.

## Scale caveat for cross-track reasoning

Nano-scale results (swift track) bound *performance*, not *emergence*: a
behavior appearing at base may simply not fit in a ~4M-param brain, so a nano
null is never evidence against a base-scale mechanism. Use local runs to
validate machinery, pacing, and signal plumbing; use the beta track to claim
anything about behavior.
