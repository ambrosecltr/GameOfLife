---
round: 004
title: competence killed motivation
date: 2026-07-05
status: complete
question: With a stable ecology, does lifelong learning across generations actually accumulate — and what does a learned world feel like to live in?
headline: Cross-lifetime learning works (model loss 84→19 across generations) — and that's the problem; curiosity collapsed 20× as the world became predictable, homeostasis was ~1000× too quiet to take over, and behavior decayed to aimless wandering in a food-rich world.
runs:
  - save: saves/beta_04
    config: configs/run/local_social.yaml
    brain: configs/brain/dreamer.yaml
    commit: "4668138"
    ticks: 2670000 (last checkpoint; journal write-up at 2.2M while still running)
    role: experiment
baselines: [003]
tags: [motivation, curiosity, lifelong-learning]
---

# 004 — competence killed motivation

## Why this round

Validate the round-003 ecology fixes over a long soak, with the wear/repair economy
(4668138) live — and watch what cross-generation learning does over ~90+ sim-days.

## Results

**2.2M+ ticks ≈ 93 sim-days at write-up (ran to ~2.67M), 33 deaths.**

**The ecology fixes hold.** Standing bush stock oscillates 374–461 with no drift across
93 sim-days, withers ≈ sprouts (7.5k each), toxic share breathes between 12–23% around
its 15% baseline instead of ratcheting.

**The control arm works.** Foragers (fixed policy) eat 7–47/day and live 16–22 days with
heavy repair use (one repaired 74 integrity through four poisonings).

**The real finding: the dreamers are learning to predict and forgetting to live.**
Across generations (lineage inheritance carrying weights and buffer):

- World-model loss fell 84 → 36 → 19; prediction error 4×'d down — cross-lifetime
  learning, plainly.
- Curiosity (Plan2Explore disagreement, the dominant reward) collapsed 20× as the world
  became predictable; the homeostatic term (ate − damage − 0.02·low) was always ~1000×
  smaller than early curiosity.
- The reward landscape flattened; actor loss fell 17 → ~0.5; behavior drifted to aimless
  wandering.
- Dreamer eats/day: 1.9 → 0.5. Awake fraction: 16% → 11%. Median lifespan: 14.4 → 13.0
  days — in a *food-rich, stable* world. The decline is motivational, not ecological.

## Interpretation

Pure curiosity is a bootstrapping drive, not a lifelong one. Competence killed
motivation, and hunger was never loud enough to take over. A world you simply exist in
becomes boring.

## Caveats

Single run, single seed; the generational learning curve is confounded with the world's
own settling. The control arm's stability is the reason to believe the dreamer decline
is motivational.

## Next

Round 005 (beta_05) rebalances the homeostat so survival pressure remains a first-class
gradient once the world is learned: `low_energy_threshold` 0.25 → 0.4 (hunger felt
before brownout), `low_energy_penalty` 0.02 → 0.25 (dense, learnable, rivals residual
curiosity), `w_homeostasis` 1 → 2 (meals and damage count double). Curiosity untouched —
the question is whether a louder body sustains purposeful behavior after the mind has
mastered its world.
