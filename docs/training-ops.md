# Training ops: speed, pacing, and the offline screen

How to budget, pace, and instrument a round. Written for the agent staging the
next run, local or cloud. The learning code this describes is the post-swift
core on `main` (journal 010 has its provenance; this doc is only how to use it).

## The pacing model (read this before picking a world speed)

`training.train_ratio` is a *target* of updates per lived act-step. The learner
runs one worker per brain; each worker accrues debt as its brain acts and pays
it one `learn()` at a time. Debt is capped (`LearnerThread.MAX_DEBT = 1024`) —
a world outrunning its learners sheds updates rather than banking stale ones,
and hibernating brains let workers pay down the day's debt (sleep-learning).

The whole budget is one identity:

    updates/s needed  =  awake_brains × (tick_rate / act_every) × train_ratio
    updates/s available = benchmark it (below); never assume

Verification live: `updates == act_steps − warmup_steps` (exactly, ±debt cap)
means the target ratio is truly held. `train_ratio_eff` = updates ÷ *all* acts
including warmup, so it climbs toward the target over life instead of sitting
at it — don't read its early value as starvation. Both are in each brain's
metrics, with `learn_seconds` (EMA of update wall-time; first values include
one-time warmup, trust it after ~20 updates).

Run rounds **paced** (no `--headless`). Unpaced headless sprints collapse the
achieved ratio and test starvation, not the brain. `--sync` holds ratio by
stalling the sim — determinism tests only.

## Benchmarking: do this before every round on new hardware

```
uv run python scripts/bench_learn.py --brain configs/brain/<round>_dreamer.yaml \
    --devices cuda --updates 30 --brains 3
```

- It fills a buffer, times `learn()` and `act()`, and prints the sustainable
  tick_rate for the requested ratio/brain-count. Treat that as a starting
  point; confirm with the live identity above and adjust `gol-ctl speed`.
- **Benchmark contention, not just solo speed.** Sibling learner workers share
  one device. Measured on M1: solo mps beats cpu (188 vs 209 ms/update, nano),
  but at 3 workers cpu aggregates 7.6 updates/s vs mps 4.9 — one GPU queue
  serializes siblings, cores share. CUDA contention is **unmeasured**; time a
  3-thread probe on the pod before trusting the solo number.
- Reference points (M1 Pro, nano, 1024 samples/update): ~0.21s cpu / ~0.19s
  mps solo; ~0.25s under 3-worker cpu contention. LP-mode brains speed up
  once the trickle anneals (the imagination ensemble pass is skipped when
  `lp_mix_eff` reaches 0) — expect `learn_seconds` to drop at that point.
- CUDA gains vs older base-preset numbers (0.25–0.74s on a 3090) are real but
  smaller than the local speedup — the biggest local win fixed a CPU-specific
  pathology. Measure; don't extrapolate.

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
  floors (default 0); `barrier_cap` clamps −log so `V` is finite at the
  boundary. Priced through the same reward head as the comfort drive (their
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
- `boredom.gate: drive | viability` — what "calm enough to be bored" reads.
  `drive` (beta_10) reads the comfort drive, so any deficit below setpoint
  shuts the boredom gate (couples boredom to hunger — the round-011 concern).
  `viability` reads the barrier, so an agent far from the lethal floor can be
  bored while merely peckish; only true danger shuts the gate.

`training:`
- `optimizer: adam | muon` — Muon (vendored, `dreamer/optim.py`) on the
  world model's 2D matrices, Adam elsewhere. `muon_lr` (default 0.02).
  Staged ablation; not yet run in any round.
- `l2_init: w` — L2-toward-init on the world model (plasticity maintenance
  for one unbroken life). Staged ablation; not yet run in any round.
- `compile: true` — torch.compile on the RSSM step functions. Measured
  neutral on cpu; untested on cuda. Adds one-time compile stalls at brain
  construction (act path is pre-warmed; learn shapes compile on the learner
  thread where a stall is just a skipped update).

New metrics these produce: `loss_reward` (always), `reward_head_spike_err`
(|predicted − realized| on spike samples — the reachability gauge),
`spike_row_frac` (what prioritization feeds the head), `l2_init_dist`,
`lp_mix_eff` (0 = trickle annealed, imagination skip active).

## Checkpoint compatibility

Brain blobs from before the speed core load fine: the Plan2Explore ensemble
is migrated to the batched layout, salience is backfilled, and the **model
optimizer's Adam moments reset** (param layout changed; actor/critic moments
survive). For offline analysis that's harmless. For *resuming a live world*
across the boundary, expect a brief model-loss wobble while Adam re-estimates
its moments — note it in the journal if the round's data spans the resume.

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
