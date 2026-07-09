---
round: anima-006
title: the level-valence fix — read the feeling, not its change
date: 2026-07-09
status: closed
question: when the plastic modulator reads the felt LEVEL (fed feels good, hungry/near-death feel bad) instead of its tick-to-tick change, does foraging finally emerge — and does plastic beat frozen on eating-while-hungry now that the teaching signal points toward survival?
headline: "NO, and it broke in a NEW, worse way that is itself the key lesson. The direction was right (fed → +M) but the level form is UNCENTERED: because the population still never forages, it lives chronically below the neutral d_ref, so M became a permanent negative SLAB — comfort negative 99% of steps, the viability standing tax negative 100%, combined mean −2.46, negative ~100% of the time. And it was mis-scaled ~1000× for the rule's alpha, so it PINNED the fast weights to the clip (w_fast_norm 0.02 → ~1.2 of max 2.0). The Hebbian rule can only teach from a ~zero-mean signal (it reads a constant M as 'always anti-consolidate' → saturation); reduction was implicitly centered (why it is the textbook form) but misdirected, level is directed but uncentered. Evolution confirmed the harm: viability_gain −31%, alpha −16% (selection trying to switch the apparatus off). Still the same attractor (83% dormant, 39/61 hibernation, eating-while-hungry 5 vs 649 sated). Fix screened → anima_07: CENTER the signal (level minus a running baseline = a prediction error)."
runs:
  - save: saves/anima_06
    config: configs/run/anima_06.yaml
    brain: configs/brain/anima_06_plastic.yaml
    commit: ee8bb1a
    ticks: 1298000
    role: experiment
  - save: saves/anima_06_frozen (not run)
    config: configs/run/anima_06_frozen.yaml
    brain: configs/brain/anima_06_frozen.yaml
    commit: ee8bb1a
    ticks: 0
    role: control
baselines: [anima-003, anima-005]
tags: [valence, level, homeostasis, plasticity, credit-assignment, saturation]
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

## What happened (flagship 1.30M ticks; frozen arm never run)

The level form failed, and the *way* it failed is the result. Same behavioural
attractor as every prior round — 83% dormant, 39/61 hibernation deaths, eating-
while-hungry 5 vs 649 sated — but the modulator and the weights went somewhere
new and diagnostic:

- **M became a permanent negative slab.** The population never forages, so it
  lives chronically below the `d_ref = 0.40` neutral (median awake energy 26, in
  the danger band 64% of the time). So the level comfort term was negative on
  99% of steps (mean −1.22) and the viability standing tax negative on **100%**
  (mean −1.73, the larger term). Combined **M mean −2.46, negative ~100% of the
  time** — the exact opposite of a signed teacher.
- **It saturated the fast weights.** The rule's `alpha`/`decay`/`clip` were tuned
  for the reduction form's |M| ≈ 0.002; the level M is ~1000× larger, so
  `ΔW = α·M·trace` slammed `w_fast_norm` from a healthy ~0.02 to **~1.2 of the
  clip 2.0**. The net was being uniformly hammered flat, not taught.
- **Evolution tried to switch it off.** Steepest gene drift was the harmful
  terms: `viability_gain −31%`, `via_integrity_weight −29%`, `alpha −16%`.

## Why — the lesson (this is the real output)

The three-factor Hebbian rule can only teach from a **~zero-mean modulator**: it
reads a constant M as "always (anti-)consolidate," which just saturates, and
learns only from M's *variation* around zero. So:

- **reduction** (Δfeeling) is implicitly centered — changes average to ~0 — which
  is *why* it is the textbook form; but it is blind to sustained state and
  telescopes negative over a mortal life (anima_05).
- **level** (raw feeling) is correctly directed but **uncentered**, and is only
  centered if the agent actually visits *both sides* of neutral. A never-foraging
  population never visits the fed side, so level valence degenerates into a
  constant punishment slab that saturates the weights.

Neither pure form works. The right neuromodulator for a three-factor rule is a
**prediction error** — level minus an expectation — which is what dopamine
actually is. That is anima_07.

## Next → anima_07 (centered / prediction-error valence)

Screened this session (`scripts/anima_valence_screen.py`, new ROAD-1 section) on
the anima_06 saves: `M = comfort_gain·(baseline_d − d) − standing_gain·(V −
baseline_V)`, where each baseline is a running EMA of the feeling. This is the
interpolation between the two failed extremes — reduction ≈ baseline half-life→0,
level ≈ half-life→∞. At a **short baseline (~300 ticks)** it is ~zero-mean
(−0.08 vs level's −2.46), swings both ways (34% positive), and points the right
way — **corr(M, ΔE) = +0.42** (M rises when energy rises). Longer baselines
collapse back toward the level slab.

Honest caveat carried into 007: on this never-recovering population there is NO
baseline length that is both centered AND senses sustained hunger — a short
baseline works only by *forgetting* the chronic decline (edging toward
reduction's weakness). So centering yields the best-conditioned signal available
(unsaturated, right-signed) but cannot *prove* it bootstraps foraging; that needs
a live run. If it still can't start, the triangulation onto the critic/world fork
is then very strong. Watch **w_fast_norm** (must stay ≪ clip, unlike here) and
**frac(M>0)** (must be well off 0) as the mechanical health reads.
