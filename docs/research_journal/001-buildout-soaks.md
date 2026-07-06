---
round: 001
title: build-out soaks
date: 2026-07-05
status: complete
question: Does the basic economy support life, does one mind learn, and where should brains run?
headline: Economy calibrated (bush clumps + softened costs), first mind confirmed lifelong learning across a body death (lineage inheritance), devices benchmarked (nano→CPU on M1).
runs:
  - save: not retained
    config: (pre-beta ad-hoc)
    brain: mixed scripted + configs/brain/dreamer.yaml
    commit: (during M0–M5 build-out, ≤490b492)
    ticks: 100000 (economy) / 40000 (first mind)
    role: experiment
baselines: []
tags: [calibration, economy, lineage, devices]
---

# 001 — build-out soaks

## Why this round

Local soaks (M1 Pro) during the M0–M5 build-out: validate the energy economy, confirm the
first learning brain actually learns, and pick devices before any long runs.

## What changed

These runs *were* the calibration; changes fell out of them (see Next).

## Results

**Economy calibration (100k ticks ≈ 4 sim-days, 5 foragers + 3 walkers).** Foragers
thrive indefinitely (energy 94–99 at age 100k); walkers starve within a sim-day and churn
through hibernate → death → scrap → respawn (9 walker deaths).

**First mind (40k ticks, 1 nano dreamer + 7 scripted, lineage on).** Model loss 115 → 83,
ray-depth prediction error 0.46 → 0.16, ensemble disagreement (curiosity) 0.118 → 0.079 —
the model is learning its world and the world is becoming less surprising, exactly the
expected signature. The original body (bot_000) starved during motor babbling and died;
its mind continued in bot_008 with the loss curve unbroken across the death.

**Device benchmark (M1 Pro).** nano: cpu 474 ms/update vs mps 568 (cpu wins, and act is
0.6 ms vs 9.1). small: mps 655 vs cpu 1009 (mps wins).

## Interpretation

The economy is livable for a competent policy and lethal for a random one — the intended
gradient. A lineage learns even though bodies die: lifelong learning across deaths is
real from the first soak.

## Caveats

Saves not retained; numbers above are the record. Pre-beta world configs differ from the
later beta series in many economy constants.

## Next

- Bushes must generate in **clumps** — a single 1-block bush slips between the 9°-spaced
  sensor rays and is invisible at range.
- Costs softened to basal 0.0015 / move 0.005 / eat 40.
- Warmup cut 2000 → 500 act-steps (newborns were starving mid-babble).
- `inherit_weights: lineage` made the default.
- Device policy: learning brains live on `devices.learning`; local nano default is cpu.
- First cloud soak should target the social-curiosity experiment
  (`configs/run/exp_social_curiosity.yaml` and its masked twin) — still pending.
