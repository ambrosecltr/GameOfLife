---
round: anima-005
title: the economy-and-senses pass — a fair, sensable world, same brain
date: 2026-07-09
status: planned
question: with the water hazard softened to a drag AND made interoceptive, and sleep restoring to a functional floor in one rest cycle, does the SAME anima_03 brain finally forage — and does plastic beat frozen on eating-while-hungry?
headline: "(staged, not launched) anima_04 showed the per-action economy is already forager-equal; the plastic overhead was screaming (fixed) + swimming an unsensed hazard. anima_05 holds the brain FIXED and changes only the world: water_drain_mult 3.0→1.75 (drag, not a 3× tax — speed_mult 0.5 already double-charges crossing), a new OBS_VERSION-5 in-water proprio channel so the hazard is felt, and wake_energy 65→38 so one rest cycle restores a functional body (guard: still < repair_threshold, so sleep never funds repair). Both arms run — the plastic-vs-frozen verdict anima_03/04 deferred."
runs:
  - save: saves/anima_05 (planned)
    config: configs/run/anima_05.yaml
    brain: configs/brain/anima_05_plastic.yaml (verbatim anima_03)
    commit: tbd
    ticks: 0
    role: experiment
  - save: saves/anima_05_frozen (planned)
    config: configs/run/anima_05_frozen.yaml
    brain: configs/brain/anima_05_frozen.yaml
    commit: tbd
    ticks: 0
    role: control
baselines: [anima-003, anima-004]
tags: [economy, senses, water, communication, plasticity, obs-version]
---

# anima 005 — the economy-and-senses pass (planned)

## Why this round

anima_04 (calibration) established two things: (1) per-ACTION the world's
economy is already life-like — a plastic body costs what a competent forager
body costs, once you remove signaling and water; (2) the plastic overhead is
therefore behavioural/perceptual, not a pricing problem — they scream (fixed:
signal_cost 0.001) and they wade through an almost-unsensed water hazard that
is the single biggest drain in the world (26% of plastic spend). This round
removes the two genuine world-side obstacles and holds the brain fixed, so the
next result is a clean read on the brain.

## What changed (world only; brain is verbatim anima_03)

- **water_drain_mult 3.0 → 1.75.** Water is a thicker medium (drag), not a
  metabolic 3× penalty; and water_speed_mult 0.5 already doubles the ticks
  (hence energy) to cross a given distance, so 3× double-counted. ~1.75 total
  surcharge on top of the speed halving is "thick, not lethal".
- **OBS_VERSION 4 → 5: in-water proprio channel (index 18).** The felt half of
  the hazard — 1.0 while submerged, else 0.0. Water was only a blue ray tint;
  now avoiding it is learnable with a clean interoceptive signal. Old
  checkpoints won't load (fresh founder population — free). The drain still
  teaches; the channel only informs.
- **wake_energy 65 → 38.** Sleep restores to FUNCTIONAL (just above brownout
  25), not full, in ~one day/night rest cycle (measured solar ~+38–51/cycle).
  Guard preserved: 38 < repair_threshold 60, dormant bodies never repair and
  keep losing integrity — sleep buys tomorrow, only foraging→surplus→repair
  buys next month, so the mortality gradient is intact.
- Reproduction thrive_energy stays 75 (> wake 38; one meal from a 38 wake =
  78 > 75, so a bud still costs an earned meal).

## Pre-registered questions

- P1: does plastic awake water-spend fall toward the forager's (0.00041) once
  water is sensable — i.e. do they learn/evolve to avoid it?
- P2: does post-wake eat-before-rehibernate finally rise well above anima_03's
  11%, now that a fed body has ~0.4 sim-day of runway and food is affordable?
- P3 (the deferred verdict): plastic vs frozen on eating-while-hungry. BOTH
  ARMS RUN. If plastic ≈ frozen again, the within-life Hebbian rule adds
  nothing even in a fair world — a strong statement about the family.
- P4: with signaling ~free (RQ3), does signal usage acquire any structure —
  amplitude, event-correlation, spatial clustering?
- P5: does the budding channel un-starve (thriving-pass rate ≫ anima_03's
  2.6%), giving selection real differential reproduction to work on?

## Method notes

- Read the energy ledger via `anima_stats` (ENERGY BUDGET section) and the
  scratch calibration script; compare plastic vs the forager anchor every run.
- Watch the in-water channel's effect as a *learning* signal, not a reward:
  the question is whether behaviour organizes around it, never whether we
  rewarded avoidance (we didn't).

## Next

(to be written when the round closes)
