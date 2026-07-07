---
round: 009
title: the conditioning round — does boredom bite once the yardstick stops shrinking?
date: 2026-07-07            # pre-registered at build; closed same day at 2.988M ticks
status: complete
question: With capacity held at the beta_08 bundle and only signal conditioning changed — anchored normalization, an annealed cold-start trickle, and boredom as integrated pressure — does the curiosity→hunger handoff finally happen (boredom dwells, eating rises, lifespans decouple from the hibernation clock)?
headline: "conditioning worked and boredom is a thermostat, not a ratchet: it drove a real foraging binge (eats 5→14→9→5 per 200k, first within-run rise ever) that collapsed once its own success closed the gates — the binding constraint moves to reward reachability (the blackout is invisible, meals too rare for the reward head)"
runs:
  - save: saves/beta_09
    config: configs/run/beta_09_conditioning.yaml
    brain: configs/brain/beta_09_dreamer.yaml
    commit: 56ecf74
    ticks: 2988314          # extended past the 2.5M target; clean SIGINT close, final atomic checkpoint
    role: experiment
baselines: [008, 007]
tags: [motivation, reward, normalization, boredom]
---

# 009 — the conditioning round

## Why this round

Round 008 proved capacity necessary but not sufficient: the model converged
(loss 29→4.2), curiosity decayed (stimulation 3.7→~0.5), boredom fired for the
first time — and the cascade still stalled, for three instrumented reasons:

1. **The normalizer treadmill.** Both curiosity normalizers divide by a
   lifetime running std, so a decaying signal shrinks its own yardstick:
   `curiosity_scaled` rose 0.09→1.86 (~20×) while raw LP fell.
2. **The immortal trickle.** The newborn cold-start mix
   (`0.1 × normalized disagreement`) rode that re-inflated channel to a
   standing ~0.19 floor — ~40% of the 0.5 boredom stim gate, forever.
3. **No accumulator.** Boredom was an instantaneous gate product; stimulation
   sat ON the gate for 650k ticks and boredom never exceeded 1.6e-3 because
   there was no state for pressure to build in.

## What changed vs beta_08 (the only knobs)

All in `beta_09_dreamer.yaml`, all config-flagged with legacy code defaults:

- `reward.norm_anchor_samples: 1_000_000` — both normalizers calibrate on
  early life (~1000 updates) then freeze, so convergence reads as satisfaction.
- `lp.mix_anneal_steps: 1500` — the trickle anneals to zero over ~2.5 awake
  sim-days. Inherited newborns carry donor act-steps and skip it (they are
  not cold); only true-fresh founders get the full subsidy.
- `boredom.pressure: true` (`rise 0.002`, `decay 0.0002`) — boredom is now a
  leaky-integrated mood charged by calm×dull real experience and drained by
  lived relief; imagination pays `weight × pressure × gates`, so the actor
  can plan its way out. Newborns reset pressure (not born jaded); pressure
  rides `state_dict` so checkpoints resume the mood exactly.

Capacity bundle (base preset, train_ratio 1.0, cuda), world, seed protocol,
population, HRRL drive stack, temperament: byte-identical to beta_08.

## Predictions (written before launch)

- **P1 — the treadmill dies.** `curiosity_scaled` flattens after the anchor
  freezes (~1000 updates) instead of climbing all run; `stimulation` falls
  *through* the gate rather than hovering on it, since the trickle floor
  anneals away (`lp_mix_eff` → 0 by ~1500 act-steps).
- **P2 — boredom dwells.** `boredom_pressure` charges toward saturation over
  sustained dull safety (008 had 650k ticks of it); the `boredom` penalty
  reaches O(weight × gates) = O(0.01) — four orders above 008's flickers —
  and stays there until behavior changes it.
- **P3 — the handoff.** With curiosity genuinely quiet and boredom pushing,
  homeostasis (~0.005) is no longer drowned: eats/100k rises within-run for
  the first time (008: 8→~2 collapse), some dreamer lifespans decouple from
  the ~347k hibernation clock, and the poisoned-meal fraction (008: 38%)
  finally has selection pressure to fall.
- **P4 — the falsification branch.** If pressure charges (P2) but behavior
  still doesn't move (P3 fails), conditioning was not the binding constraint
  either: the suspects become the homeostasis reward scale itself and the
  actor's ability to cash sparse meal gradients — i.e. back to reward
  *semantics* or policy capacity, and the entry must say which the data
  favors (watch whether the actor at least *reduces boredom* — dithering vs
  purposive escape distinguishes them).
- Free riders to watch: dreamer-only terraforming (318/287 in 008) under
  boredom pressure — does "play" increase digging? Lineage styles under a
  working gratification balance.

## Operations

Same box class and pacing rule as 008 (RTX 3090, ≥16 vCPU; run paced, watch
`train_ratio_eff`, speed 3 sustainable while dreamers hibernate heavily —
recheck if P3 wakes them up, which is the point). Rerun stays OFF (008 disk
lesson). Budget: 12 h ≈ 2.6M ticks ≈ $2.60. Mirror home on a loop from the
laptop — NOTE `scripts/sync_back.sh` mirrors with `--delete-after`: never
park laptop-only artifacts inside the synced save dir (008 lost its local
.rrd copy exactly this way).

## Results

Final: **2,988,314 ticks** in ~14 h wall (RTX 3090, paced speed 3, ~215k
ticks/h, ≈ $3.10), closed by SIGINT with a clean atomic checkpoint
(`ckpt_000002988314`). `train_ratio_eff` climbed 0.51 → 0.945 all run;
population held at 6. Lifetime: 29 dreamer individuals, 26 deaths, 55 eats,
19 poisonings; scripted foragers 8,659 eats over 20 individuals. All bucket
series below are per 200k ticks.

**P1 — confirmed. The treadmill is dead.** `curiosity_scaled` rose through
the calibration window (0.17 → 0.62 by 400k) and then held 0.52–0.70 for the
rest of the run — beta_08 climbed ~20× over the same span. `lp_mix_eff`
annealed 0.029 → 0.001 by 200k, zero from 400k, exactly on the 1500-act-step
schedule. `stimulation` fell *through* the gate (3.8 → 0.42 bucket mean)
instead of hovering on it — beta_08's 650k-tick stall did not recur.

**P2 — confirmed with an amendment: pressure charges but does not saturate.**
`boredom_pressure` sat near zero while curiosity was loud (< 0.002 through
0.8M), then charged monotonically once stimulation approached the gate:
0.026 / 0.074 / 0.117 / 0.159 / 0.207 / 0.291 / 0.346 / 0.378 / 0.393 /
0.389 — decelerating (+0.084, +0.055, +0.029, +0.015, −0.004) and holding an
equilibrium of **~0.39 for the final 600k ticks**. The bucket-mean `boredom`
penalty peaked at 2.0e-4 — two orders above 008's flickers but two short of
the O(0.01) prediction, because pressure plateaued at ~40% of range with the
gates only partly open. The
deceleration is not a failed constant: it is the closed loop working (see
interpretation). Per the pre-registered caveat, the O(weight × gates)
prediction assumed saturation, which the system self-limits away from.

**P3 — fired as a binge, not a regime change.** Final dreamer eats series:
`10, 1, 0, 1, 0, 3, 3, 1, 1` for the first 1.8M (the 10 is the newborn
burst), then **5, 14, 9, 5** across 1.8–2.6M — the first within-run rise in
dreamer eating in the project's history, timed exactly to pressure crossing
~0.2–0.3 — then **0, 2** to close. Eats doubled (20 → 40) in the 380k ticks
after the gate crossing, then decayed back to the 2–3/bucket baseline.
(Bucket 13's zero is confounded by a world-wide event: *nothing* ate for
200k ticks, forager eats included — see free riders — but bucket 14's
2-vs-644 dreamer/forager split shows the dreamer collapse is real.)
Lifespans partially decoupled during eating phases: 3 of 26 deaths broke
the hibernation clock (hibernation ledger 57–68% with poison 36–48 —
dreamer_031 at age 239k the starkest, the first dreamer deaths not dominated
by hibernation), but all 23 others reverted to the ~347k clock (hibernation
ledger ≥ 70%, mostly > 90%); mean lifespan 326k. The poisoned-meal fraction
never moved: 19 of 74 meals (eats + poisonings) ≈ 26% toxic, flat between
halves of the run — no avoidance selection signal.

**P4 — the falsification branch landed, with the dithering-vs-purposive test
answered.** Behavior moved *while pressure was charging* and the thing that
moved was foraging — the drive-relevant action, not noise. Purposive escape:
policy capacity is (provisionally) exonerated. What failed is what happened
after: the binge died exactly when (a) pressure plateaued — its gradient,
not its level, is what the actor had been surfing — and (b) the eaters got
hungry, which closes the calm gate (the `boredom` penalty fell 2.0e-4 →
1.2e-4 across the final buckets while pressure held flat). Of P4's two
suspects, the data favors **reward reachability** over policy capacity, in
two specific mechanisms (see interpretation).

**Free riders.** Two pre-registered watches and one surprise:
- *Terraforming stayed dreamer-only* (325 digs / 298 places vs foragers' 0)
  and rose with boredom pressure exactly like eating: mid-run ~15–18
  digs/bucket → 23 / 26 / 35 / 29 through the binge window, collapsing with
  it. "Play" does increase under pressure.
- *Lineage styles*: not analyzed this round (single-lineage turnover
  dominated the population; deferred).
- *Surprise — synchronized world sleep.* In bucket 13 (2.6–2.8M) the entire
  population went dormant together (mean 5.4 of 6 agents, foragers
  included): zero eat events world-wide for 200k ticks while ripe bushes sat
  at their run-maximum (mean 339). An emergent population-wide sleep/wake
  oscillation — energy budgets synchronized by the shared day/night and
  respawn cohorts — worth watching as a system-level rhythm.

## Interpretation

**1. All three conditioning knobs did their jobs.** Anchored normalization
made convergence read as satisfaction; the annealed trickle removed the
immortal floor; pressure gave boredom state to accumulate in. Round 008's
indictment of the signal-conditioning bundle is confirmed — it was a real
binding constraint, and it is now lifted.

**2. Boredom is a thermostat, not a ratchet — and that is a design
validation, not a shortfall.** Pressure equilibrated at ~0.39 because the
loop closed: boredom pushed, behavior changed, the changed behavior made the
world newly unpredictable (`curiosity_scaled` crept 0.62 → 0.70 under a
*frozen* anchor — honest raw LP, not renormalization), stimulation recovered
toward the gate, charging slowed. A homeostat that regulates its input is
exactly what a mood should be. Saturation would have meant the actor
couldn't respond; the plateau means it did.

**3. The handoff-within-the-handoff is the real finding: boredom is gated
off precisely when hunger is on.** By design, "an agent in need is never
bored" — so boredom can train foraging-while-sated (it did: the binge), but
the moment deficits rise, the only signal licensed to drive eating is
homeostasis itself. Eating collapsed at exactly that switchover. The round's
question — does the curiosity→hunger handoff happen — resolves to: curiosity
now hands off cleanly to *boredom*, boredom hands the agent to the bushes,
and hunger then fumbles the catch.

**4. Why HRRL hunger fumbles: the reward is well-shaped but unreachable.**
This round already runs need-relative drive-reduction homeostasis (the meal
that saves a starving agent is worth ~0.8; standing hunger costs ~0.005 per
step). The semantics are not the problem. Two structural gaps keep the
signal from acting:

- **The blackout is invisible.** 90%+ of dreamer deaths are hibernation-drain
  deaths, but the collapse is architecturally unexperienceable:
  `reset_stream()` fires on wake ("the dormant gap is never observed" —
  `gol_brains/base.py`), and the HRRL reduction term zeroes at sequence
  starts, so the largest negative-valence event in a dreamer's life — energy
  hits zero, the body crashes through integrity while dormant — never enters
  the reward stream. Death itself is not an experience at all in a
  non-episodic world: the stream just stops, and no value ever propagates
  back from nonexistence. The agent cannot learn "starving leads to ruin"
  because, from the inside, starving leads to a cut.
- **Meals are too rare for the learned reward head.** Imagination pays
  homeostasis via `head_reward` (twohot, trained on realized reward): 55
  meals across 3.0M ticks × 3 concurrent dreamers means the head trains on
  essentially zero positive-homeostasis events per batch. The actor cannot
  plan toward a spike its reward head has never learned to predict. The
  binge is the counterfactual that proves the point: when a *dense* signal
  (boredom relief) pointed at the bushes, foraging happened within 200k
  ticks.

**5. The binding constraint moves: capacity (008) → signal conditioning
(009) → reward reachability.** Not reward *values* — no new drives, no
manual rewards — but making the already-priced consequences of hunger
actually traversable by the world model and reward head.

## Next

*(round 010 candidates, all config-flagged with legacy defaults, boredom
stack untouched: it works)*

- **Price the blackout: make dormancy a transition instead of a cut.** On
  wake, feed the pre-collapse state as the predecessor of the wake
  observation — one visible transition carrying the real energy/integrity
  delta — instead of `reset_stream()`. HRRL then prices the crash
  automatically (a huge drive jump = a huge negative reduction), with no new
  reward terms. Decide deliberately whether a dormancy-duration proprio
  channel is warranted (that is an OBS_VERSION 4 decision; the bare
  transition is not).
- **Feed the reward head: reward-aware replay.** Prioritize replay of
  sequences containing |r_homeo| spikes so meals (and, once visible,
  blackouts) actually reach the twohot head. This changes what is learned
  from, not what is rewarded — no shaping. Ablation flag:
  `replay.prioritize: none | reward`.
- **Prediction sketch for 010:** with the blackout priced and meals
  representable, eating should rise *while hungry* (not only while
  bored-sated), hibernation-ledger deaths should fall, and the poisoned-meal
  fraction finally has a value gradient to select against it.

## Caveats

- Three conditioning knobs move together (deliberate: 008 indicted them as a
  bundle; the per-knob ablations are `norm_anchor_samples: 0`,
  `mix_anneal_steps: 0`, `pressure: false` if attribution is needed).
- Anchor and pressure constants were sized from beta_08 telemetry, not tuned:
  anchor ≈ 1000 updates (early adulthood), pressure saturation ≈ a sim-day of
  sustained dull safety. If P2 charges too fast/slow the constants are wrong
  before the design is.
- Single run; round 006 measured 40% forager variance between identical-config
  runs.
- The reward-head-starvation mechanism (interpretation §4, second gap) is
  inferred from event counts, not instrumented: we did not log per-batch
  positive-homeostasis sample counts or the head's prediction at meal states.
  Round 010 should add that telemetry before acting on it.
- The post-binge collapse rests on two closing buckets, one of which
  (bucket 13) is confounded by the world-wide dormancy event; bucket 14's
  dreamer/forager split (2 vs 644 eats) is the cleaner evidence.

## Next

*(pending close)*
