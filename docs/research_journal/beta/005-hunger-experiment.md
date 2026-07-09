---
round: 005
title: the hunger experiment
date: 2026-07-05
status: complete
question: Does a louder body (rebalanced homeostat) sustain purposeful behavior after curiosity fades?
headline: A louder body softened the motivational decay but couldn't stop it — hunger reward held steady and lifespans stabilized, but the policy never learned to cash the gradient in; ~1 meal/day in replay is too sparse at nano/CPU capacity.
runs:
  - save: saves/beta_05
    config: configs/run/local_hunger.yaml
    brain: configs/brain/dreamer_hungry.yaml
    commit: d993912
    ticks: 3474200
    role: experiment
baselines: [004]
tags: [motivation, homeostasis, capacity]
---

# 005 — the hunger experiment

## Why this round

Round 004's diagnosis: curiosity collapses once the world is learned and homeostasis was
~1000× too quiet to take over. This round makes the body louder without touching
curiosity.

## What changed

`dreamer_hungry.yaml` vs `dreamer.yaml` (the only variable; world and population
identical to round 004):

- `low_energy_threshold` 0.25 → 0.4 — hunger felt before brownout sags the body.
- `low_energy_penalty` 0.02 → 0.25 — dense, learnable pressure rivaling residual curiosity.
- `w_homeostasis` 1 → 2 — meals and damage count double.

## Results

**3.45M ticks ≈ 144 sim-days.** The louder body softened the decay but did not stop it:

- Homeostatic reward held rock-steady at −0.07..−0.08 for the whole run (~25% of the
  reward stream after curiosity faded) — the flat-landscape failure of round 004 is fixed.
- In the window where beta_04 collapsed (1.7M+), hungry dreamers ate 1.1/day vs the
  baseline's 0.6–0.7, with lifespans *stabilizing* (14.0 → 14.4 → 14.5 days) instead of
  declining.
- But the within-run trend still pointed down (final era: 0.6 eats/day, awake 11%), and
  the tell is actor entropy: it *rose* all run (6.0 → 6.37).

## Interpretation

The hunger gradient is present; the policy cannot cash it in. With meals this rare in
replay (~1/day per agent), the reward head barely learns what eating is worth, so
imagination can't find the payoff — and a nano model on CPU at train_ratio 0.25 gives
each lineage only a few hundred updates per lifetime to break that chicken-and-egg.
Motivation was necessary but not sufficient; the binding constraint has moved from
reward design to learning capacity and experience density.

## Caveats

The 1.1 vs 0.6–0.7 eats/day comparison is cross-run against beta_04 (different commit:
d993912 adds the bush lifecycle). Round 006 later showed forager (control-arm) eat rates
vary ~40% between identical-config runs, so treat sub-0.5 eats/day deltas as noise;
the *stabilization pattern* and the entropy trend are the robust evidence.

## Next

- The case for the cloud round: small/base preset, higher train_ratio, same paired
  configs (local_social vs local_hunger) so the motivation ablation carries over.
- beta_05 paused at ckpt 3450000 (verified resumable) for later continuation.
