---
round: 002
title: survival pressures, broken bodies
date: 2026-07-05
status: complete
question: Do the new survival pressures (wake economy, brownout, hibernation) produce livable-but-real stakes?
headline: The wake economy was a death ratchet (wake below brownout) and the calibration probe was wedged from spawn (eat resolved only at eye height) — when the probe fails you can't distinguish "economy too harsh" from "bodies broken."
runs:
  - save: saves/beta_02
    config: configs/run/local_social.yaml
    brain: configs/brain/dreamer.yaml
    commit: 490b492
    ticks: 229000
    role: experiment
  - save: saves/beta_01
    config: configs/run/local_social.yaml
    brain: configs/brain/dreamer.yaml
    commit: 490b492
    ticks: (short)
    role: precursor
baselines: [001]
tags: [economy, calibration, embodiment]
---

# 002 — survival pressures, broken bodies

## Why this round

First beta soak after the M4+M5 build-out (490b492): 3 dreamers + 3 scripted foragers in
the standard world, testing whether the survival economy gives real stakes without being
a meat grinder. (beta_01 was a short precursor at the same commit, superseded by beta_02
and never analyzed.)

## What changed

Baseline beta configuration — this round tested the build-out as shipped.

## Results

**The wake economy was a death ratchet.** `wake_energy` (15) sat *below* the brownout
threshold (25), so robots woke pre-crippled, paying full commanded cost for reduced
motion, with ~1000 ticks of budget before collapsing again.

**The calibration probe was wedged from spawn.** `GRIP_EAT` resolved targets only at eye
height, so a bush one block downhill (plainly visible to the −30° ray row) could never be
eaten. All three foragers stood in front of food, spamming eat at exactly basal drain,
until integrity death at ~209k ticks.

## Interpretation

When the calibration probe fails, you cannot distinguish "economy too harsh" from "bodies
broken." The scripted foragers exist precisely to separate world problems from learning
problems; this round they measured neither.

## Caveats

No learning conclusions can be drawn from beta_02 — the bodies were broken for every
policy, learned or scripted.

## Next

Fixes (landed in 9cd18bf and 4668138):

- `_faced_edible`: gaze scan ±1 block vertically so eat resolves what the eyes see.
- A forager eat stall-breaker.
- Wake at 40 (above the brownout knee) with faster solar trickle.
- The wear/repair/senescence economy: repair funded by energy surplus, efficiency halving
  per `senescence_halflife`, per-robot integrity ledger for death-cause attribution.
