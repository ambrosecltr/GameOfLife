---
round: anima-003
title: the audibility package — appetite works, the corridor didn't
date: 2026-07-09
status: complete
question: with all four measured hunger-deafness blockers removed as a package, does the hunger feeling finally become audible — do hungry meals happen, and does plasticity consolidate them?
headline: "The package failed for a measurable reason: the corridor was sized against an ASSUMED drain (basal+move 0.0065/tick) but real awake drain is 0.012–0.023/tick, so wake 55 bought ~750 ticks not 2.3k and post-wake eat-first fell to 11% — while appetite itself worked (arousal 0.32→0.55 with hunger, gene_appetite_gain the top eat-rate correlate at r=+0.21) and the rectified gate went perfectly silent (no escapes ever happen), leaving M a net-negative whisper that teaches nothing. Calibrate against measurement, not config arithmetic."
runs:
  - save: saves/anima_03
    config: configs/run/anima_03.yaml
    brain: configs/brain/anima_03_plastic.yaml
    commit: (uncommitted at launch; staged post-anima_02 review)
    ticks: 4454900
    role: experiment
  - save: (never launched)
    config: configs/run/anima_03_frozen.yaml
    brain: configs/brain/anima_03_frozen.yaml
    commit: same
    ticks: 0
    role: control (planned, not run — see Caveats)
baselines: [anima-002]
tags: [motivation, valence, plasticity, calibration, appetite, economy]
---

# anima 003 — the audibility package: appetite works, the corridor didn't

## Why this round

anima_02's inversion probe proved comfort valence behaviorally inert and diagnosed
four stacked blockers (see anima-002). This round removed all four as a package,
pre-registered as: if it works, ablate later for attribution; if it fails with
every measured blocker gone, that is a strong negative for the three-factor
Hebbian family as configured. The verdict is neither — the package *didn't
actually remove blocker #1*, so the family wasn't yet tested.

## What changed

- **World** (`configs/world/anima_03.yaml`): wake_energy 40 → 55 (intended to
  double the awake search corridor).
- **Brain** (`configs/brain/anima_03_plastic.yaml`): `appetite_gain 2.0` — new
  heritable gene, arousal = clip(restlessness·(1 + appetite·drive), 0, 0.6)
  scales motor noise + the discrete ε-floor; `plasticity.decay 1e-4` (engram
  half-life ~35k ticks) + `tau 60` (credit window ~300 ticks);
  `valence.viability.rectified true` (consolidate escapes only).
- **Reproduction**: thrive_energy 65 (must clear wake_energy or waking is free
  bud-eligibility — caught in pre-launch review; the thrive check is
  instantaneous, `scheduler._is_thriving`).
- Checkpoint-compat: missing gene keys fill with 1.0 on load.

## Results

**Appetite worked mechanically and is selection-visible.** Awake arousal by
energy: 0.316 (E 80–100) → 0.418 (40–60) → 0.545 (0–20), a clean monotonic
gradient. `gene_appetite_gain` is the strongest gene correlate of per-robot eat
rate (Pearson r = +0.21, n = 116 robots ≥50k ticks) — ahead of restlessness
(+0.04), comfort_gain (+0.09), viability_gain (−0.12).

**Eat-when-hungry did not emerge.** Near-food eat rate per 100k awake
robot-ticks: 1.8 (E 20–40) vs 119 (E 80–100) — the same anti-gradient as
anima_02. 12 hungry (E<40) eats vs 958 sated; median energy at eat 97.8.

**The corridor calibration was wrong — the round's operative finding.**
Measured awake drain (energy deltas between consecutive awake samples,
eat-jumps excluded): **0.012–0.023/tick** across energy bands, 2–3× the
basal+move arithmetic (0.0065) used to size wake_energy. Hibernation is forced
at E ≤ 0 (`world.py _account_energy`); median wake→re-hibernate was **2,114
ticks** — wake 55 bought ~750 extra ticks, not ~2,300. Post-wake
eat-before-rehibernate **fell to 11%** (anima_02 era: 18–24%; n = 1,243
wakes). The 138 successes ate at median 1,128 ticks post-wake, i.e. around
E 30–50 — genuinely drive-relevant meals; there just weren't enough. Dig/place
rates were *lower* than the anima_02-era baseline (240–427 vs 278–594 per 100k
awake robot-ticks), so action spam is not the drain; the unexplained residual
(suspect: climb at 0.15/block on hilly terrain, exhaustion/water multipliers)
is what anima_04 instruments.

**The rectified gate went perfectly silent.** |m_viability| p99 = 0.0003 —
zero escapes ever consolidated, because an escape requires eating below E ~25,
which requires the eat to happen (the same chicken-and-egg). With the negative
half rectified away and no positive events, M collapsed to the comfort whisper:
|M| mean 0.002 (clip 5), life_return_comfort mean −0.98, **0% of robots
positive**. anima_02's gate punished awake life; anima_03's says nothing.
Silencing the punisher created no teacher.

**Population and selection.** 42 → 4 plastic by tick 500k, pinned at floor 4
thereafter. 32 buds (bursts at 106–212k and 826–900k, last at 3.98M) vs 174
spawns. Thrive-leg analysis over awake adult (age ≥ 20k) samples: **2.6% pass
both legs; energy < 65 fails 87%, integrity < 70 fails 74%** — both binding.
113 deaths, 87 hibernation-dominant, median death age 311k (~3.5 senescence
half-lives). Dormant fraction 0.75; awake fraction ~0.16 throughout.

## Interpretation

- The round did not test "can the Hebbian rule learn eating" — the teaching
  event stayed at ~1 per agent-lifetime because the corridor fix undershot 3×.
  Config arithmetic is not a substitute for measurement: the world's real
  cost surface (terrain climbing, multipliers) was never observable per-cause.
- Appetite raises *search* but cannot conjure *reach*: the sated-vs-hungry eat
  gap is largely a reach gap (campers sit on clumps with hundreds of in-reach
  tries; hungry wanderers pass within sensing radius 8 but not within grip
  reach). Digs at E 0–20 (240/100k) prove grips work under brownout.
- The two M-gate forms are now both measured at scale: unrectified =
  net-suppressing (anima_02), rectified = silent (anima_03). Neither teaches
  while escapes don't occur. The gate design is downstream of event rate;
  fixing the gate before the event rate was order-of-operations backwards.

## Caveats

- **The frozen control never launched** — no plastic-vs-frozen claim can be
  made for this round. Appetite's r = +0.21 is within-run, cross-lineage, and
  stands on its own; any "plasticity helped/didn't" statement does not.
- **Founder-genome confound vs anima_02**: measured founder gene means differ
  substantially (restlessness 1.10 vs 1.88, viability_gain 0.59 vs 1.02;
  cause unclear — possibly survivorship-weighted sampling in the stats
  windows). Cross-round eat-level comparisons are polluted; within-run
  correlations are the clean signals.
- Single arm, one seed; population sat at the floor (n=4) for 90% of the run,
  so late-run statistics describe floor-respawn dynamics more than lineage
  evolution.

## Next

- **anima_04 — the calibration round (measurement before mechanism).** Add a
  per-cause ENERGY LEDGER to the world (mirror of the integrity ledger:
  basal/move/climb/signal/exhaustion/water/dig/place/repair/bud spend,
  eaten/solar income, banked-not-nominal meals so overflow is visible), run a
  diagnostic, and re-derive the whole economy coherence table — corridor
  length, meals/day required, travel radius per meal, lifespan in sim-days —
  from measured numbers. Then size wake_energy (or trim the dominant cost) so
  the corridor is genuinely ~5k+ ticks. No brain changes in this round.
- **Deferred fork (explicitly not taken yet): "salivation"** — hunger-tilted
  ε-floor toward EAT. It would manufacture the event but installs the reflex
  the track is asking whether the brain can learn; only to be considered if
  hungry meals stay rare under honest physics, and then gene-scaled from ~0 so
  selection, not design, turns it up.
- Rule adopted: **affordance calibration must use ledger-measured rates, not
  config arithmetic** (this round's lesson, now written into the energy-ledger
  docstring).
