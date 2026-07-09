---
round: anima-007
title: centered valence — the modulator as a prediction error (TRACK CLOSE)
date: 2026-07-09/10
status: closed
question: when the plastic modulator is centered (a prediction error — the felt level minus a running baseline — instead of the raw level), does it stop saturating the fast weights, and does foraging finally bootstrap?
headline: "NO, and the way it failed closes the track. In vivo the prediction-error M was NOT zero-mean: frac(M>0) was 3.2% overall / 12.1% awake (screen predicted ~34%), mean M −0.51 drifting to −1.14 as the population aged — because a trailing EMA can only center a STATIONARY feeling, and a mortal, never-recovering life is a monotone downward trend (felt senescence rose to ~0.47, energy pinned ~20 near brownout). The baseline lags the decline and the 'error' becomes a standing negative bias: Σ over a life of (level − EMA) ≈ net felt decline / λ — anima_05's telescoping negativity, re-derived for the third and last geometry. Mechanics half-improved (w_fast plateau ~0.82 vs 006's pinned ~1.2 of clip 2.0; awake corr(M,ΔE)=+0.27, right sign) but foraging DECAYED (eats/100k 269→36; hungry 3 vs sated 742), the population hit the floor (4/42; 32/54 hibernation + 22 poison deaths), budding starved (0 buds in the last 100k). Reduction (Δ), level, and level−EMA are one family, and the whole family integrates negative over a declining life; the only escapes are actual thriving (chicken-and-egg) or a LEARNED expectation — a critic; dopamine is a TD error, not level-minus-EMA. A learned predictor is exactly what the anima charter excludes → the bet ('feeling alone can teach') is falsified. TRACK CLOSED after 7 rounds."
runs:
  - save: saves/archive/anima_07
    config: configs/run/anima_07.yaml
    brain: configs/brain/anima_07_plastic.yaml
    commit: 079c6c6
    ticks: 783340
    role: experiment
  - save: saves/anima_07_frozen (never run — moot once P1 failed)
    config: configs/run/anima_07_frozen.yaml
    brain: configs/brain/anima_07_frozen.yaml
    commit: 079c6c6
    ticks: 0
    role: control
baselines: [anima-005, anima-006]
tags: [valence, prediction-error, centering, homeostasis, plasticity, saturation, track-close]
---

# anima 007 — centered valence (closed; closes the anima track)

## Why this round

anima_06 proved the level modulator is directionally right but mechanically
broken: uncentered, it becomes a permanent negative slab on a population that
never forages (mean M −2.46, negative ~100% of steps), which pinned the fast
weights to the clip (w_fast_norm 0.02 → ~1.2 of 2.0) — the net was hammered flat,
not taught. The lesson from stacking the two failures:

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

Offline screen (on the anima_06 saves): mean M −0.08, 34% positive,
corr(M, ΔE) ≈ +0.42 at the short baseline. Pre-registered health reads:
w_fast_norm ≪ clip 2.0, frac(M>0) ≈ 35–45%.

## Pre-registered questions → answers

- **P1 (the mechanical fix) — FAILED, diagnostically.** frac(M>0) was **3.2%**
  of all samples, **12.1%** awake-only — nowhere near the screened ~34%. Mean M
  ran **−0.51 in the first 50k and worsened monotonically to −1.14** by 750k.
  w_fast_norm did *not* pin like 006 (window means 0.36 → plateau **~0.82**,
  per-sample max 1.5–1.7, vs 006's mean ~1.2 of the same clip 2.0) — the
  re-seeding on every wake plus the smaller |M| halved the hammering — but a
  plateau at ~40% of clip held by a one-signed slab is still saturation
  dynamics, not teaching.
- **P2 (foraging bootstrap) — NO, it decayed.** Plastic eats/100k:
  **269, 202, 83, 26, 61, 51, 17, 36** (foragers in the same world: 226–1149).
  Eating-while-hungry: **3 hungry vs 742 sated** — the sated-snacking pattern of
  every round since anima_01, now at lower volume.
- **P3 (plastic > frozen, 5th deferral) — moot, control never launched.** The
  primary read was mechanical (P1) by pre-registration; when the signal itself
  failed in the first check-in there was nothing for a frozen arm to control.
  The verdict stands on rounds 01/05/06: frozen ties or beats plastic, 3 for 3.
- **P4 (death-avoidance from positive danger-escape M) — NO.** frac(m_via>0)
  never exceeded ~5% of samples; m_viability was itself the dominant negative
  term (see below). Deaths 54: **32 hibernation, 22 poison** (41% poison — the
  highest of any anima round; what eating survived was undiscriminating).
  Median death age 246,950 (p10 150k, p90 327k) — still the hibernation clock.
- **P5 (budding un-starves) — NO.** 9 bud events in 783k ticks, **0 in the last
  100k**, thriving-now 0/4. Population ended at the floor: **4/42 plastic**
  (71 respawn spawns over the run). Stopped at 783,340 with a clean final
  checkpoint; with the population at floor and no buds, there was nothing left
  for a longer run to measure.

## What happened — the signal, in numbers

Per-50k-window means over all plastic brains (metrics every 100 ticks):

| window | mean M | mean M_c | frac(M_c>0) | mean M_via | w_fast (mean/max) | dormant |
|---|---|---|---|---|---|---|
| 0–50k | −0.51 | −0.15 | 2.6% | −0.47 | 0.36 / 1.48 | 0.68 |
| 100–150k | −0.72 | −0.20 | 3.5% | −0.68 | 0.79 / 1.54 | 0.76 |
| 300–350k | −0.78 | −0.17 | 1.4% | −0.76 | 0.82 / 1.38 | 0.74 |
| 500–550k | −0.93 | −0.21 | 3.4% | −0.73 | 0.77 / 1.60 | 0.76 |
| 750k+ | −1.14 | −0.18 | 1.9% | −0.94 | 0.71 / 1.30 | ~0.75 |

- **M_viability, not M_comfort, is the slab this time** — and it *grows with
  population age* (−0.47 → −0.94), tracking felt senescence (mean ≈ 0.47 by the
  end; senescence half-life 90k, median death age ≈ 2.7 half-lives). The comfort
  error stayed a roughly constant −0.15..−0.21 (energy pinned at 19–28, right at
  the brownout floor 25, so hunger *trends* rather than oscillates).
- **Live corr(M, ΔE/100t): +0.27 awake-only** (screen said +0.42) — right sign,
  weaker; **−0.06 over all samples**, because 68–81% of plastic robot-time is
  dormant, where M is pure slab and energy is solar trickle.
- **Life-integrated returns stayed deeply negative for the third geometry in a
  row:** life_return_comfort −52..−115, life_return_via −40..−77 per life.
- **Evolution leaned against the apparatus again**, milder than 006:
  comfort_gain −17%, via_integrity_weight −22%, viability_gain −5%
  (006 was viability_gain −31%).

## Why — the lesson (this is the real output)

**A trailing average can only center a stationary signal.** Write the EMA update
as `base ← base + λ·(x − base)`; then over a life, `Σ(x − base) ≈
(x_end − x_start)/λ`. The prediction error is zero-mean **iff the felt level
ends where it began.** A mortal agent that never learns to forage is a monotone
downward trend — hunger ratchets, integrity wears, and (new since OBS v4)
senescence *rises by construction* — so the baseline chronically lags reality
and the "error" carries a standing negative bias proportional to the
deterioration rate. That is exactly what the live data shows: M_via growing
with population age, M never zero-mean, life-returns negative.

This is anima_05's telescoping identity again, one level up. All three
canonical geometries of an *unlearned* felt-state modulator are one family —
reduction is `λ→1`, level is `λ→0`, the EMA error interpolates — and the whole
family obeys the same integral: **over a declining life, any causal centering
of felt state sums to the net felt decline, which for a mortal that never
thrives is negative.** The offline screen missed it because it replayed
recorded anima_06 streams; the live population aged past anything in those
recordings — the trend the signal needed to center is *created by the very
failure the signal was supposed to fix* (distribution shift by dying).

Two exits exist, and only two:

1. **Actually thrive**, so the felt level mean-reverts and the error is
   genuinely zero-mean. Chicken-and-egg: that requires the foraging this signal
   was supposed to teach.
2. **Center against a learned expectation** — a predictor of the felt level
   conditioned on state, i.e. a value function. This is what dopamine actually
   is: a **TD error against a learned critic**, not level-minus-trailing-average.
   A learned baseline adapts to the trend (its forecast *includes* the decline),
   so surprise stays zero-mean even on a dying trajectory.

Exit 2 is machinery the anima charter deliberately excluded: the founding bet
(proposal 002) was that *feeling alone* — no world model, no critic, no gradient
descent — could gate Hebbian consolidation into survival competence. Seven
rounds falsified that bet at every layer we could isolate: the attractor is not
architecture (01), the valence is behaviorally inert when misdirected (02), the
corridor was real but fixing it didn't help (03/04), the world is exonerated
(05), and the modulator geometry fails in all three canonical forms for one
provable reason (05/06/07). Adding a TD-learned baseline would fix the signal —
a linear TD head is even backprop-free — but a brain with a learned value
predictor is a *different family with a critic*, not anima. The honest
conclusion is that the family, as chartered, cannot generate its own teaching
signal in a mortal world.

## Track verdict — anima CLOSED after 7 rounds

**Answer to the track question:** No. A backprop-free, world-model-free
recurrent net whose fast weights are gated by an evolved homeostatic valence
signal did not keep itself alive in beta's world, and within-life plasticity
never beat the frozen control (0 for 3 with 2 deferrals). The binding
constraint is constitutive, not parametric: a three-factor rule needs a
zero-mean modulator, a zero-mean modulator over a mortal life needs a learned
expectation, and a learned expectation is the critic the family was defined by
not having. Do not iterate valence further (rule pre-registered in 006/007 and
now triggered twice).

**Fairness note vs beta:** the dreamer has not cracked foraging either (012:
first live mortality gradient, behaviour did not convert) — but beta's wall is
downstream, in converting a *healthy learned* gradient into policy, while anima
never obtained a usable teaching signal at all. The dreamer's learned critic is
precisely the component that centers the signal; that comparison is the track's
scientific value.

**What the track leaves behind (all merged, all live in beta's world):**

- **The architecture-independence result** (anima_01): the hibernation
  attractor is reward geometry, not a planning/gradient artifact — this
  reframed beta's 012 mortality work.
- **The zero-mean theorem for neuromodulated Hebbian learning** (05/06/07):
  reduction/level/error tried and measured; the constraint and its proof are
  the design spec for any future plasticity family.
- **World instrumentation:** per-cause energy ledger (anima_04), OBS_VERSION 4
  senescence channel, OBS_VERSION 5 in-water channel, the calibrated water/wake
  economy (anima_05) — beta inherits all of it.
- **Tooling:** `scripts/anima_stats.py`, `scripts/anima_valence_screen.py`,
  `scripts/anima_viability_screen.py`.
- **The felt-state rule** (kept from the charter, validated): felt STATE as
  intrinsic signal is legitimate; ACTION/fitness reward is scripting. The
  missing piece is that the *centering* must be learned, not designed.

## Next

- **No anima_08.** Save archived intact to `saves/archive/anima_07`; this
  entry carries the durable numbers per journal policy.
- If a plasticity family is ever revisited, it starts a **new track** with a
  TD-learned baseline (a micro-critic over the felt level) as a founding design
  element — per the pre-registered "critic fork," which is outside anima's
  charter by definition.
- Primary line of work returns to **beta** (012 follow-ups: mortality gradient
  → behaviour conversion) and the staged **swift** nano round.
