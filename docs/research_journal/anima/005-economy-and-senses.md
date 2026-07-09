---
round: anima-005
title: the economy-and-senses pass — a fair, sensable world, same brain
date: 2026-07-09
status: closed
question: with the water hazard softened to a drag AND made interoceptive, and sleep restoring to a functional floor in one rest cycle, does the SAME anima_03 brain finally forage — and does plastic beat frozen on eating-while-hungry?
headline: "NO on both, and the forensic read relocated the binding constraint off the world entirely. A fair, sensable world (water softened + felt, wake 38) did not produce foraging: plastic eats collapsed to single digits and the population fell to the floor (4/42, 5/42) dying of hibernation (43/58). Frozen ATE MORE than plastic (17,998 vs 9,570 energy; 844 vs 446 meals) — the 3rd straight null on within-life plasticity. The cause is the MODULATOR GEOMETRY, not the world: change-based comfort M was negative on 99.9% of act-steps and telescoped net-negative over any mortal life (life_return ≈ −1.2), and the rectified viability gate was INERT (m_viability ≡ 0.000 across 1.4M ticks). A plastic brain has no critic to integrate reductions into a value, so a change-based signal teaches ~nothing — being fed earns zero because it is not a transition. Fix built + offline-screened for anima_06: read the LEVEL, not the change."
runs:
  - save: saves/anima_05
    config: configs/run/anima_05.yaml
    brain: configs/brain/anima_05_plastic.yaml (verbatim anima_03)
    commit: 3d2d5f9
    ticks: 1436700
    role: experiment
  - save: saves/anima_05_frozen
    config: configs/run/anima_05_frozen.yaml
    brain: configs/brain/anima_05_frozen.yaml
    commit: 3d2d5f9
    ticks: 1334300
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

## What happened

Both arms ran ~1.4M / 1.3M ticks and collapsed to the population floor (plastic
4/42 alive, frozen 5/42), dying overwhelmingly of hibernation (43 of 58 plastic
deaths). Against the pre-registered questions:

- **P1 (water avoidance): no.** Awake water spend stayed ~22% of the plastic
  budget; the new in-water proprio channel did not organize avoidance
  (confounded — they barely learn anything, so it's not a clean read on the
  channel).
- **P2 (eat-before-rehibernate rises): no.** Foraging is tiny and *falling*
  (plastic eats/100k 219 → single digits) vs the forager anchor's hundreds.
- **P3 (plastic vs frozen — the deferred verdict): plastic ≠ better; frozen is
  if anything better.** Frozen ate more total energy (17,998 vs 9,570) over more
  meals (844 vs 446). Third straight null on within-life W_fast.
- **P5 (budding un-starves): no.** thriving-now 0/4, zero buds in the last 100k.
  Selection still has almost no differential reproduction (N≈4 alive — the gene
  drift is noise at this population).

## Why — the forensic read (this is the real result)

The world is not the binding constraint; the modulator's *geometry* is. Measured
on the run's own telemetry + an offline replay of the recorded lives:

1. **Change-based comfort M is net-negative over every life.** m_comfort is
   negative on **99.9%** of act-steps (every draining tick is a tiny "ow"; a meal
   is a rare spike) and `life_return_comfort` averages **−1.2**, negative in
   every window. This is beta's telescoping-negative-return wall wearing the
   neuromodulator's clothes: over a mortal life the reduction sum ≈ (feeling at
   birth − feeling at death) < 0, because they die starving. **Being fed earns
   nothing** — it is a state, not a transition.
2. **The rectified viability gate is inert.** `m_viability ≡ 0.000` across 1.4M
   ticks in *both* arms. It only fires on in-stream recovery from below-safe,
   which never happens: agents hibernate at the floor and `reset_stream()` wipes
   the trace. Everything proposal 003 built to teach survival contributes
   literally zero to plastic learning. (anima_03 rectified the gate to stop it
   *anti*-consolidating; that just swapped "harmful" for "dead".)
3. **The eat-energy surface is a red herring but the hunger is real.** 80% of
   eat *events* log at energy >85 — a feeding-bout artifact (eating raises
   energy; the tail of a bout reads high). The awake population is in fact
   chronically hungry: **median awake energy 26**, 74% of awake-time below 40,
   in the viability danger band 64% of the time.
4. **Offline the reduction return is anti-correlated with survival.** Replaying
   the recorded lives (`scripts/anima_valence_screen.py`), reduction life-return
   correlates **−0.80** with mean energy and **−0.41** with eating — it rewards
   volatile near-death cycling, the *opposite* of staying fed. The level form
   flips both to **+0.70 / +0.4**.

Root cause: a plastic brain has **no critic** to integrate a stream of reductions
back into a value (the dreamer can use reductions precisely because its critic
reconstructs the level). Feed a no-critic learner changes and it learns ~nothing.

## Next → anima_06 (the level-valence fix)

Built and offline-screened this session; both arms staged, not launched. The
modulator now reads the felt **level**, not its change:
`M_comfort = comfort_gain·(d_ref − d)` (fed → +, hungry → −) and viability as a
standing danger tax `−standing_gain·V` (fires near the floor, where the gate was
≡0). Screen picked `d_ref = 0.40` (neutral ≈ energy 31, just above brownout;
peak corr(return, energy)), viability standing weight 0.5, and the credit window
widened tau 60 → 120 (600 ticks) since a level signal is on every step and wants
to span a full approach-then-feed. Change is anima-local (plastic/brain.py); the
dreamer is untouched (it keeps `feeling.reduction()`; anima never called it).
See `anima/006-level-valence.md`.

**Health metric to watch in 006:** `life_return` sign. The screen's caveat is
that on *this* starving population no d_ref makes a fed life net-positive — that
is a world-harshness signal (median awake energy 26), not a valence bug. If fed
agents emerge under level valence, life_return should cross positive; if it
stays uniformly negative, the next lever is the world config, not the feeling.
