---
round: anima-004
title: the measured economy — a calibration round
date: 2026-07-09
status: complete
question: what does the world's energy economy actually cost, per cause, and does it scale coherently against a life — corridor, meals/day, recharge, lifespan?
headline: "A pure calibration round (no brain experiment): the per-cause energy ledger overturned the cost model twice. First, SIGNALING was the #1 plastic drain — 36.6% of spend, ~the price of walking — silently taxing the charter's 'free signal channel' (RQ3); foragers don't signal, which is why their drain (0.0065) matched the old estimate while plastic (0.0129) ran 2×. Cut signal to near-free and WATER took the #1 slot at 25.9% (vs foragers' 7.5%) — and it's an almost-unsensed hazard (blue ray tint only, no interoceptive 'I'm submerged' channel). Net finding: per-ACTION the economy is already forager-equal; the plastic overhead was screaming + swimming, both brain/sense problems, not price problems. Deliverable = the calibrated economy for anima_05."
runs:
  - save: saves/anima_04_diag (archived, brains dropped)
    config: configs/run/anima_03.yaml (+ energy ledger instrumentation)
    brain: configs/brain/anima_03_plastic.yaml
    commit: (staged, uncommitted)
    ticks: 24400
    role: precursor (calibration diagnostic, one sim-day, pre-signal-cut)
  - save: saves/anima_04_diag_verify (archived, brains dropped)
    config: configs/run/anima_04.yaml
    brain: configs/brain/anima_04_plastic.yaml
    commit: (staged, uncommitted)
    ticks: 100000
    role: precursor (calibration verify, post-signal-cut economy, ~4 sim-days)
baselines: [anima-003]
tags: [calibration, economy, communication, instrumentation, water, senses]
---

# anima 004 — the measured economy (a calibration round)

**Scope note:** anima_04 ran no brain experiment. It built the energy ledger,
measured the economy, and produced the calibrated world that anima_05 tests
brains on. The plasticity verdict (plastic vs frozen) that was pre-registered
here moves to anima_05, where hungry meals are actually reachable AND water is
sensable. The planned anima_04/anima_04_frozen experiment runs were folded into
anima_05 rather than launched.

## Why this round

anima_03's lesson, now a rule: **affordance calibration must use ledger-measured
rates, not config arithmetic.** Its corridor was sized against an assumed
0.0065/tick awake drain; reality was 0.012–0.023, and the round's central
intervention silently never happened. anima_04 changes *nothing in the brain*
— it instruments the world, measures, and re-derives the economy so that
energy costs, levels, integrity, and lifespan are scaled coherently against
the world the brains actually live in.

## What changed

- **Energy ledger** (instrumentation, all kinds): per-cause cumulative spend
  (basal / move / climb / signal / exhaustion-surcharge / water-surcharge /
  dig / place / repair / bud) and income (eaten / solar) on every robot,
  mirroring the integrity ledger; in metrics rows, death events, and
  checkpoints. Meals record energy *banked*, not nominal — overflow (sated
  eating) is now directly visible. A balance test pins ledger ≡ energy delta.
- **World** (`configs/world/anima_04.yaml`): signal_cost 0.01 → **0.001**
  (measured 36.6% of plastic spend; charter RQ3 says the signal channel should
  be ~free — 0.01/tick was drift, discovered only by measurement);
  wake_energy 55 → **65** (sized from measured post-cut drain).
- **Reproduction**: thrive_energy 65 → **75** (stays 10 above wake_energy —
  waking is never bud-eligibility; one meal from wake clears it; 75 − 40 bud
  cost = 35 > brownout 25).
- Brain configs are verbatim copies of anima_03's (appetite 2.0, decay 1e-4,
  tau 60, rectified gate), frozen per round convention.

## The calibration measurement (saves/anima_04_diag, 24.4k ticks = 1 sim-day)

Per-cause awake spend, plastic (42 robots, 434k awake / 591k dormant robot-ticks):

| cause | /awake-tick | share |
|---|---|---|
| signal | 0.00473 | 36.6% |
| move | 0.00203 | 15.7% |
| water (surcharge) | 0.00180 | 13.9% |
| basal | 0.00149 | 11.5% |
| dig | 0.00126 | 9.7% |
| climb | 0.00102 | 7.9% |
| repair | 0.00035 | 2.7% |
| place | 0.00025 | 1.9% |
| **total** | **0.01293** | |

Foragers (no signaling, competent navigation): 0.00655/awake-tick — move 36%,
water 25%, basal 19%, climb 15%. The scripted body's drain equals the old
assumption; the plastic overhead was babble + water-blundering + random digs.
Climb — anima_03's prime suspect — is only 7.9%. Income: plastic banked 1,460
energy from meals (0.00337/awake-tick) vs 0.01293 spend — a structural ~4×
awake deficit; solar 0.00097/dormant-tick.

**World coherence (measured, pre-cut):** wake 55 corridor to brownout 2,321
ticks (matches anima_03's observed 2,114 median wake→re-hibernate); a meal
buys 3,094 awake ticks (0.13 sim-day); break-even 7.8 meals/sim-day of
wakefulness; solar coma 0→55 in 56.5k ticks (2.35 sim-days) costing ~17
integrity; senescence halflife 3.8 sim-days; a 300k-tick life = 12.5 sim-days.

## The verify measurement (saves/anima_04_diag_verify, 100k ticks ≈ 4 sim-days, signal cut applied)

Per-cause awake spend, plastic (42 robots, 1,183k awake / 3,017k dormant robot-ticks):

| cause | /awake-tick | share | forager |
|---|---|---|---|
| **water (surcharge)** | **0.00248** | **25.9%** | 0.00041 (7.5%) |
| move | 0.00217 | 22.7% | 0.00210 (38.4%) |
| basal | 0.00148 | 15.5% | 0.00120 |
| dig | 0.00129 | 13.5% | — |
| climb | 0.00108 | 11.3% | 0.00116 |
| signal | 0.00050 | 5.2% | 0 |
| repair | 0.00030 | 3.1% | 0.00059 |
| place | 0.00025 | 2.7% | — |
| **total** | **0.00956** | | **0.00546** |

Three findings:

1. **The signal cut worked** — 36.6% → 5.2% (still ~half-amplitude babble, now
   nearly free). Total plastic drain 0.01293 → 0.00956.
2. **Water is the new #1 cost and it is the ENTIRE plastic-vs-forager gap.**
   Plastic spends 0.00248/tick in water vs foragers' 0.00041 — a 6× difference
   that alone (~0.00207) accounts for essentially the whole drain gap
   (0.00956 − 0.00546 = 0.00410, the rest being dig/signal the forager doesn't
   do). Move is dead even (0.00217 vs 0.00210). **Per-action, plastic ≈ forager:
   the economy is not overpriced; the plastic brains scream and swim.**
3. **Water is almost unsensed.** Checked the obs contract: water is a normal
   block, so it shows only as a blue tint in the ray RGB; there is NO
   interoceptive "I'm submerged / being drained" proprio channel (proprio has
   velocity, energy, integrity, touch=solid-contact, light, fatigue,
   senescence — nothing for water). The biggest drain in the world is a hazard
   the brain gets no clean feedback about. Climb — anima_03's prime suspect —
   is 11.3% and correctly charged **up-only** (descent is gravity, no cost;
   fall damage only past 3 blocks, waived in water). No climb fix needed.

**World coherence (measured, post-cut):** wake 65 corridor to brownout **4,186
ticks** (was 2,114 in anima_03 — the corridor genuinely ~doubled this time); a
meal buys 4,186 awake ticks (0.17 sim-day); break-even 5.7 meals/sim-day at the
plastic rate but **only ~1.15 at the forager rate** (forager funds 87% of its
awake burn from food: 0.00475 eaten / 0.00546 spent). At the forager rate a
full tank lasts **0.76 sim-day**, usable band ~0.57 — i.e. "a tank gets a
competent body through most of a day," the intended target, already met. Solar
0.00212/dormant-tick → coma 0→65 in 1.28 sim-days (~+51 energy per full rest
cycle). Plastic funds only 21% of its burn from food (0.00197/0.00956): they
starve because they don't forage, not because food is dear.

## Interpretation

- **The per-action economy is already life-like.** Once signaling and water are
  removed, a plastic body costs what a competent forager body costs. The
  "misalignment" that looked like a pricing problem is two specific things: one
  unsensed hazard (water) and a coma-length recharge (wake 65 needs ~1.7 rest
  cycles).
- **Tune against the forager, not the plastic brains** (adopted rule): the
  forager is the achievable-competence anchor and it lives comfortably with
  margin. Any residual plastic failure is therefore a brain/sense result — the
  thing the track is trying to isolate — not an economy result.
- The measured coherence is sound: tank ≈ most-of-a-day, meal ≈ 0.17 day,
  lifespan ≈ 12.5 sim-days bounded by senescence (a ceiling, extendable within
  the envelope by foraging→repair, shiftable across generations by evolved
  genes). Lifespan needs no retuning; competent foraging needs to become
  achievable so a well-fed life can even be measured.

## Caveats

- Both diagnostics are on hibernation-attractor populations (hyperactive,
  always-signaling, water-blundering). The forager is the clean competence
  anchor; plastic rates describe a broken policy, not the economy's intent.
- Founder-genome distributions differ from anima_02 (see anima-003).
- No brain experiment ran; no plasticity claim is made here (it moves to
  anima_05, with both arms).

## Next

- **anima_05 — the economy-and-senses pass** (this round's deliverable applied,
  then the brain re-test). Three changes, all following from the measurements:
  1. **`in_water` proprio channel** (OBS_VERSION 4 → 5): make the biggest
     hazard in the world *felt*, so avoiding it is learnable — a real body
     knows it's submerged. Honest per invariant 6 (deliberate versioned
     contract change); old checkpoints won't load, cost-free on a fresh
     founder population.
  2. **water_drain_mult 3.0 → 1.75**: water is a thicker medium (drag), not a
     3× metabolic penalty — and water_speed_mult 0.5 already doubles the ticks
     (hence energy) to cross it, so the multiplier double-counts. ~1.75 total
     surcharge on top of the speed penalty is the "thick, not lethal" target.
  3. **Recharge reframe**: wake_energy 65 → ~38 (functional floor just above
     brownout 25), leaving solar as-is (already ~+38–51/rest-cycle) so **one
     day/night of rest wakes you**, life-like. Guard preserved: solar never
     reaches the repair threshold (60) and dormant integrity keeps draining +
     no dormant repair, so sleep buys tomorrow but only foraging→surplus→repair
     buys next month — the mortality gradient stays intact. thrive_energy stays
     75 (still > wake, one meal from a 38 wake).
  Then run BOTH arms (plastic + frozen) — the plasticity verdict anima_03 and
  anima_04 both deferred.
- Energy ledger is permanent instrumentation; `anima_stats` reads it.
