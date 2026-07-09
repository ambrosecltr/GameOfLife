---
round: 003
title: bush ecology and the toxic ratchet
date: 2026-07-05
status: complete
question: With working bodies and a wear economy, does a long soak produce a stable ecology and behavior-dependent lifespans?
headline: First emergent ecology — lifespans became behavior-dependent, but the regrow rule was a one-way toxic ratchet; the better the population avoided poison, the more poisoned the world became. Plants effectively evolved defenses under grazing pressure.
runs:
  - save: saves/beta_03
    config: configs/run/local_social.yaml
    brain: configs/brain/dreamer.yaml
    commit: 9cd18bf
    ticks: 10500000
    role: experiment
baselines: [002]
tags: [ecology, emergence, death-ledger]
---

# 003 — bush ecology and the toxic ratchet

## Why this round

Long soak with the round-002 body/economy fixes in place: do stakes now differentiate
behavior, and does the food ecology hold over hundreds of sim-days?

## Results

**10.5M ticks ≈ 437 sim-days, 189 deaths.** First emergent ecology result:

- **Lifespans became behavior-dependent** — foragers ~16 days, dreamers ~10–12, with
  death ledgers cleanly attributing poison vs hibernation vs wear.
- **The commons quietly degraded**: ripe bushes 290 → ~115 while toxic climbed 45 → ~210.
  Mechanism: a one-way ratchet in the regrow rule. An eaten ripe bush regrows toxic 15%
  of the time, but a toxic bush stays toxic until *someone eats it*. The better the
  population avoids poison, the more poisoned the world becomes; the only recycling
  agents were desperate or still-learning dreamers (374 poisonings). Plants effectively
  evolved defenses under grazing pressure.
- **Proto-provisioning observed**: dreamers dig up bushes and carry them. But hand-eaten
  bushes never scheduled a regrow, permanently deleting food sites.

## Interpretation

The ecology produced a genuine tragedy-of-the-commons dynamic from a two-line regrow
rule — a preview of how much emergent structure cheap asymmetries can generate. But an
unbounded ratchet eventually starves everyone regardless of behavior, which destroys the
long-horizon experiments this world exists for.

## Caveats

The ripe→toxic drift means late-run scarcity is an artifact of the regrow rule, not of
population behavior; don't read beta_03's late-era survival stats as an economy
calibration.

## Next

Fixes (landed in d993912, bush lifecycle ecology):

- **Bush senescence**: `bush_lifespan_ticks` ≈ 5 sim-days; withered bushes are replaced
  by sprouts with toxicity *re-rolled*, biased toward existing patches by
  `sprout_clump_bias` — breaks the ratchet.
- **Conserved bush budget**: standing + held + queued sprouts is invariant; hand-eaten,
  spoiled, and died-carrying bushes all return their slot.
- `held_spoil_ticks` so carried food perishes.
- Placement/caching/feeding deliberately kept alive for the cultural-transmission
  questions.
- beta_03 preserved as the "before" dataset.
