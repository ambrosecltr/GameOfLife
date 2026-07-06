---
round: 009
title: the conditioning round — does boredom bite once the yardstick stops shrinking?
date: 2026-07-07            # pre-registered at build; results pending (~12 h GPU, ≥2.5M ticks)
status: planned
question: With capacity held at the beta_08 bundle and only signal conditioning changed — anchored normalization, an annealed cold-start trickle, and boredom as integrated pressure — does the curiosity→hunger handoff finally happen (boredom dwells, eating rises, lifespans decouple from the hibernation clock)?
headline: pending
runs:
  - save: saves/beta_09
    config: configs/run/beta_09_conditioning.yaml
    brain: configs/brain/beta_09_dreamer.yaml
    commit: pending         # set at launch
    ticks: 0                # target ≥2.5M (~12 h paced at speed 3)
    role: experiment
baselines: [008, 007]
tags: [motivation, reward, normalization, boredom]
---

# 009 — the conditioning round

## Why this round

Round 008 proved capacity necessary but not sufficient: the model converged
(loss 29→4.2), curiosity decayed (stimulation 3.7→~0.5), boredom fired for the
first time — and the cascade still stalled, for three instrumented reasons:

1. **The normalizer treadmill.** Both curiosity normalizers divide by a
   lifetime running std, so a decaying signal shrinks its own yardstick:
   `curiosity_scaled` rose 0.09→1.86 (~20×) while raw LP fell.
2. **The immortal trickle.** The newborn cold-start mix
   (`0.1 × normalized disagreement`) rode that re-inflated channel to a
   standing ~0.19 floor — ~40% of the 0.5 boredom stim gate, forever.
3. **No accumulator.** Boredom was an instantaneous gate product; stimulation
   sat ON the gate for 650k ticks and boredom never exceeded 1.6e-3 because
   there was no state for pressure to build in.

## What changed vs beta_08 (the only knobs)

All in `beta_09_dreamer.yaml`, all config-flagged with legacy code defaults:

- `reward.norm_anchor_samples: 1_000_000` — both normalizers calibrate on
  early life (~1000 updates) then freeze, so convergence reads as satisfaction.
- `lp.mix_anneal_steps: 1500` — the trickle anneals to zero over ~2.5 awake
  sim-days. Inherited newborns carry donor act-steps and skip it (they are
  not cold); only true-fresh founders get the full subsidy.
- `boredom.pressure: true` (`rise 0.002`, `decay 0.0002`) — boredom is now a
  leaky-integrated mood charged by calm×dull real experience and drained by
  lived relief; imagination pays `weight × pressure × gates`, so the actor
  can plan its way out. Newborns reset pressure (not born jaded); pressure
  rides `state_dict` so checkpoints resume the mood exactly.

Capacity bundle (base preset, train_ratio 1.0, cuda), world, seed protocol,
population, HRRL drive stack, temperament: byte-identical to beta_08.

## Predictions (written before launch)

- **P1 — the treadmill dies.** `curiosity_scaled` flattens after the anchor
  freezes (~1000 updates) instead of climbing all run; `stimulation` falls
  *through* the gate rather than hovering on it, since the trickle floor
  anneals away (`lp_mix_eff` → 0 by ~1500 act-steps).
- **P2 — boredom dwells.** `boredom_pressure` charges toward saturation over
  sustained dull safety (008 had 650k ticks of it); the `boredom` penalty
  reaches O(weight × gates) = O(0.01) — four orders above 008's flickers —
  and stays there until behavior changes it.
- **P3 — the handoff.** With curiosity genuinely quiet and boredom pushing,
  homeostasis (~0.005) is no longer drowned: eats/100k rises within-run for
  the first time (008: 8→~2 collapse), some dreamer lifespans decouple from
  the ~347k hibernation clock, and the poisoned-meal fraction (008: 38%)
  finally has selection pressure to fall.
- **P4 — the falsification branch.** If pressure charges (P2) but behavior
  still doesn't move (P3 fails), conditioning was not the binding constraint
  either: the suspects become the homeostasis reward scale itself and the
  actor's ability to cash sparse meal gradients — i.e. back to reward
  *semantics* or policy capacity, and the entry must say which the data
  favors (watch whether the actor at least *reduces boredom* — dithering vs
  purposive escape distinguishes them).
- Free riders to watch: dreamer-only terraforming (318/287 in 008) under
  boredom pressure — does "play" increase digging? Lineage styles under a
  working gratification balance.

## Operations

Same box class and pacing rule as 008 (RTX 3090, ≥16 vCPU; run paced, watch
`train_ratio_eff`, speed 3 sustainable while dreamers hibernate heavily —
recheck if P3 wakes them up, which is the point). Rerun stays OFF (008 disk
lesson). Budget: 12 h ≈ 2.6M ticks ≈ $2.60. Mirror home on a loop from the
laptop — NOTE `scripts/sync_back.sh` mirrors with `--delete-after`: never
park laptop-only artifacts inside the synced save dir (008 lost its local
.rrd copy exactly this way).

## Results

*(pending)*

## Interpretation

*(pending)*

## Caveats

- Three conditioning knobs move together (deliberate: 008 indicted them as a
  bundle; the per-knob ablations are `norm_anchor_samples: 0`,
  `mix_anneal_steps: 0`, `pressure: false` if attribution is needed).
- Anchor and pressure constants were sized from beta_08 telemetry, not tuned:
  anchor ≈ 1000 updates (early adulthood), pressure saturation ≈ a sim-day of
  sustained dull safety. If P2 charges too fast/slow the constants are wrong
  before the design is.
- Single run; round 006 measured 40% forager variance between identical-config
  runs.

## Next

*(pending close)*
