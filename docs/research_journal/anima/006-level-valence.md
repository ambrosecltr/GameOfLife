---
round: anima-006
title: the level-valence fix — read the feeling, not its change
date: 2026-07-09
status: planned
question: when the plastic modulator reads the felt LEVEL (fed feels good, hungry/near-death feel bad) instead of its tick-to-tick change, does foraging finally emerge — and does plastic beat frozen on eating-while-hungry now that the teaching signal points toward survival?
headline: "(staged, not launched) anima_05's forensic read located the wall in the modulator's geometry, not the world: change-based M was negative on 99.9% of steps and telescoped net-negative over a mortal life, and the rectified viability gate was inert (m_viability ≡ 0 across 1.4M ticks). A plastic brain has no critic to integrate reductions into a value, so a change-based signal teaches ~nothing — being fed earns zero because it is not a transition. anima_06 makes ONE change: M reads the level. Comfort = comfort_gain·(d_ref − d) (fed → +, hungry → −); viability = a standing danger tax −standing_gain·V (fires near the floor where the gate was dead). Offline-screened on the anima_05 saves: level return correlates +0.70 with mean energy / +0.4 with eating, where reduction correlated −0.80 / −0.41 (it rewarded the opposite). d_ref 0.40, standing tax 0.5, tau 60 → 120 (credit window 600). Both arms run; the dreamer is untouched."
runs:
  - save: saves/anima_06 (planned)
    config: configs/run/anima_06.yaml
    brain: configs/brain/anima_06_plastic.yaml
    commit: tbd
    ticks: 0
    role: experiment
  - save: saves/anima_06_frozen (planned)
    config: configs/run/anima_06_frozen.yaml
    brain: configs/brain/anima_06_frozen.yaml
    commit: tbd
    ticks: 0
    role: control
baselines: [anima-003, anima-005]
tags: [valence, level, homeostasis, plasticity, credit-assignment]
---

# anima 006 — the level-valence fix (planned)

## Why this round

anima_05 removed the last world-side excuses (water softened + felt, sleep
restores to a functional floor) and STILL produced no foraging: the population
collapsed to the floor, dying of hibernation, and frozen ate more than plastic
for the third round running. The forensic read (journal anima/005) relocated the
binding constraint off the world and onto the **modulator geometry**:

- change-based comfort M was negative on 99.9% of act-steps and telescoped
  net-negative over any mortal life (life_return ≈ −1.2);
- the rectified viability gate was inert — `m_viability ≡ 0.000` across 1.4M
  ticks — it can only fire on in-stream recovery from below-safe, which never
  happens (hibernate → stream reset);
- offline, the reduction return is *anti*-correlated with survival (−0.80 with
  mean energy, −0.41 with eating): it rewards volatile near-death cycling.

Root cause: the plastic family has no critic to integrate a stream of reductions
into a value. The dreamer can be taught with reductions because its critic
reconstructs a level; a no-critic Hebbian learner must read the level directly,
or "being fed" — a state, not a transition — teaches nothing.

## The change (brain-only; world/pop/repro held fixed from anima_05)

The modulator reads the felt **level**:

    M_comfort = comfort_gain · (d_ref − d)      # fed (d<d_ref) → +, hungry → −
    M_via     = − standing_gain · V             # standing danger tax near the floor

- **d_ref 0.40** — the neutral hunger level. Below it (fed) valence is positive;
  above it (hungry) negative. 0.40 puts neutral at energy ≈31, just above the
  brownout floor (25), and is the peak of corr(return, energy) in the screen.
- **standing_gain 0.0 → 0.5** — viability now enters M as a continuous danger
  tax (gene-scaled via the existing `standing_gain` gene, so selection tunes the
  survival weight). It fires on 64.5% of awake steps in the screen, where the
  rectified gate was ≡0.
- **tau 60 → 120 (credit window 300 → 600 ticks)** — a level signal is on every
  step, so the eligibility trace should span a full approach-then-feed, not just
  the terminal grip.
- `viability_gain` / `rectified` are now unused (kept in the yaml for config
  parity). Change is entirely in `plastic/brain.py` `act()`; the dreamer keeps
  `feeling.reduction()` and is byte-for-byte unaffected.

## Pre-registered questions

- P1: does foraging emerge — plastic eats/100k rise and hold, not collapse?
- P2 (the deferred verdict, 4th attempt): plastic > frozen on eating-while-
  hungry? With the teaching signal finally pointed the right way, a null here
  would be a strong statement about the Hebbian rule's ceiling, not the signal.
- P3: does `life_return` cross positive for the best-fed agents? (Health metric,
  not success — see caveat.) And does the standing viability tax, now live,
  produce any death-avoidance in behaviour?
- P4: with the credit window at 600 ticks, is there any sign of *approach*
  learning (heading toward food while hungry), or only close-range grip
  conditioning as before?
- P5: does the budding channel finally un-starve (thriving-pass rate ≫ anima_05's
  ~0%), giving selection real differential reproduction?

## Method notes

- Watch `life_return` sign as the primary health read. The offline screen's
  caveat: on the *starving* anima_05 population no d_ref makes a fed life
  net-positive, because those agents lived at median awake energy 26. That is a
  world-harshness signal, not a valence bug. If fed agents emerge, life_return
  should go positive; if it stays uniformly negative across a healthy-looking
  population, the NEXT lever is the world (recharge / food density), not the
  feeling — do not re-tune the valence to paper over a harsh world.
- Re-run `scripts/anima_valence_screen.py` on the anima_06 saves mid-round to
  confirm the live level return tracks the offline prediction.
- Both arms paced/headless; compare plastic vs frozen every check-in via
  `scripts/anima_stats.py`.

## Next

(to be written when the round closes)
