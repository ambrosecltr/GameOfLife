---
round: anima-007
title: centered valence — the modulator as a prediction error
date: 2026-07-09
status: planned
question: when the plastic modulator is centered (a prediction error — the felt level minus a running baseline — instead of the raw level), does it stop saturating the fast weights, and does foraging finally bootstrap?
headline: "(staged, not launched) anima_06's uncentered level form pinned W_fast to the clip (w_fast_norm 0.02 → ~1.2) because a never-foraging population sits chronically below neutral, turning M into a permanent −2.46 slab — and a three-factor Hebbian rule reads a constant M as 'always anti-consolidate' and just saturates. reduction (Δfeeling) was implicitly centered but misdirected (telescopes net-negative over a mortal life, anima_05); level is directed but uncentered. anima_07 makes M a PREDICTION ERROR: M_comfort = comfort_gain·(base_d − d), M_via = −standing_gain·(V − base_v), with base a running EMA (half-life 60 act-steps ≈ 300 ticks). reduction ≈ baseline→0, level ≈ baseline→∞. Offline-screened on the anima_06 saves: at a short baseline M is ~zero-mean (−0.08 vs level's −2.46), swings both ways (34% positive), corr(M, Δenergy) ≈ +0.42. Watch w_fast_norm (must stay ≪ clip) and frac(M>0) (≈35-45%)."
runs:
  - save: saves/anima_07 (planned)
    config: configs/run/anima_07.yaml
    brain: configs/brain/anima_07_plastic.yaml
    commit: tbd
    ticks: 0
    role: experiment
  - save: saves/anima_07_frozen (planned)
    config: configs/run/anima_07_frozen.yaml
    brain: configs/brain/anima_07_frozen.yaml
    commit: tbd
    ticks: 0
    role: control
baselines: [anima-005, anima-006]
tags: [valence, prediction-error, centering, homeostasis, plasticity, saturation]
---

# anima 007 — centered valence (planned)

## Why this round

anima_06 proved the level modulator is directionally right but mechanically
broken: uncentered, it becomes a permanent negative slab on a population that
never forages (mean M −2.46, negative ~100% of steps), which pinned the fast
weights to the clip (w_fast_norm 0.02 → ~1.2) — the net was hammered flat, not
taught. The lesson from stacking the two failures:

- **reduction** (Δfeeling; anima_05) is implicitly centered — changes average to
  ~0 — but blind to sustained state and telescopes net-negative over a mortal
  life.
- **level** (raw feeling; anima_06) is correctly directed but uncentered, so it
  only stays zero-mean if the agent visits both sides of neutral, which a
  starving population never does → saturation.

A three-factor Hebbian rule can only teach from a **~zero-mean** modulator (it
reads a constant M as "always anti-consolidate"). The neuromodulator that is
both centered and correctly signed is a **prediction error** — the felt level
minus an expectation — which is what dopamine is.

## The change (brain-only; world/pop/repro held fixed from anima_05/06)

    M_comfort = comfort_gain · (base_d − d)        # less hungry than my normal → +
    M_via     = − standing_gain · (V − base_v)     # more endangered than normal → −
    base_x   ← EMA of the felt level (half-life `baseline_halflife` act-steps)

- **baseline_halflife 60 act-steps (~300 ticks at act_every 5).** reduction ≈
  half-life→0, level ≈ →∞; the screen picked a short baseline as the centered
  sweet spot. The baseline seeds on the first step of a stream (so that step's
  M is 0) and re-seeds on every stream break (wake/reset).
- `standing_gain 0.5` stays but is now a **centered** danger error (escaping
  danger reads positive, unlike anima_06's always-negative tax); gene-scaled.
- `d_ref` is now unused (the baseline is the moving neutral). Change is in
  `plastic/brain.py` (`act()` + baseline state in checkpoint/reset); the dreamer
  is untouched.

## Pre-registered questions

- P1 (the mechanical fix): does w_fast_norm stay ≪ clip 2.0 (anima_06 pinned it
  at ~1.2), and is frac(m_comfort > 0) well off 0 (≈35-45%, vs anima_06's ~1%)?
- P2 (the behaviour): does foraging bootstrap — plastic eats/100k rise and hold?
- P3 (the deferred verdict, 5th attempt): plastic > frozen on eating-while-
  hungry, now that the signal is centered AND correctly signed?
- P4: does escaping danger (now a positive M under centering) leave any
  behavioural trace of death-avoidance?
- P5: does the budding channel un-starve (thriving-pass rate ≫ ~0%)?

## Method notes

- **Primary reads are mechanical this round** (P1): w_fast_norm and frac(M>0).
  The prior two rounds each failed at the signal level before behaviour could
  even be assessed; confirm the signal is healthy first.
- Re-run `scripts/anima_valence_screen.py` (ROAD-1 section) on the anima_07 saves
  mid-round to confirm the live centered M tracks the offline prediction.
- **Honest caveat carried from the 006 close:** on a never-recovering population
  there is no baseline length that is both centered AND sensitive to sustained
  hunger — a short baseline works only by forgetting the chronic decline. So this
  is the best-conditioned signal available, not a guarantee. If foraging still
  does not start with a demonstrably healthy signal, the triangulation onto the
  critic (prediction/value) or a world-harshness lever is then very strong — do
  NOT respond by tuning the valence further.
- Both arms paced/headless; compare via `scripts/anima_stats.py`.

## Next

(to be written when the round closes)
