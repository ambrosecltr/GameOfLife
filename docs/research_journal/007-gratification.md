---
round: 007
title: the gratification round — do agents develop interests?
date: 2026-07-06            # launched, sanity-passed, and closed at 3.44M the same day
status: complete
question: Does replacing surprise-curiosity with the gratification stack (LP curiosity, HRRL drive reduction, boredom, temperament) produce persistent motivation, bodily rhythm, and individual interests?
headline: Individuality arrived before survival competence — the three lineages developed distinct, persistent behavioral profiles (forager/social/mixed) and actor entropy fell for the first time in the beta series, but relative LP + an unconverged model held curiosity at ~3 forever, so homeostasis stayed ~500× quieter, boredom's gates never opened, and eating didn't improve; the balance the stack was designed around needs a converging world model, i.e. capacity.
runs:
  - save: saves/beta_07
    config: configs/run/local_social.yaml
    brain: configs/brain/dreamer.yaml
    commit: "cb8f632"       # the round-007 commit (gratification stack; entry finalized after)
    ticks: 3441440          # stopped cleanly (final checkpoint ckpt_000003441440)
    role: experiment
baselines: [005, 006]
tags: [motivation, interests, reward, temperament, capacity]
---

# 007 — the gratification round

## Why this round

Rounds 004–006 established that level-of-surprise curiosity is a bootstrapping drive, not
a lifelong one (motivation decays as the world becomes predictable), and that turning the
hunger volume up is a crutch that obs v3 erased entirely (006). This round attacks the
motivation half structurally: need-relative pleasure, learning-progress interest,
restlessness, and innate individual difference — nothing object-specific wired in. Full
design rationale, hypotheses H1–H4, and failure modes live in
[`docs/beta07-gratification-plan.md`](../beta07-gratification-plan.md); this entry is the
run record.

**beta_06 is the primary A/B** (same `local_social.yaml`, same obs v3 vision, legacy
reward) — every difference is attributable to the reward machinery. beta_05 is the
shape-comparison for "did the motivational decay stop" (vision confounded).

## What changed

All in one working tree, committed as this round's commit (world/brain code plus the new
`configs/brain/dreamer.yaml` defaults). Versus beta_06, the dreamer reward machinery only:

- `reward.homeostasis: drive` — HRRL drive reduction (eat-when-starving ≈ +0.6, snack at
  satiety ≈ 0; `level_penalty: 0.01` standing hunger gradient).
- `reward.curiosity: lp` — learning-progress curiosity over 32 online k-means latent
  regions, `mix_disagreement: 0.1` newborn trickle.
- `reward.boredom` — weight 0.02, gated on drives met (< 0.15) AND stimulation flat (< 0.5).
- `temperament` — heritable log-normal multipliers (σ 0.25) over seven abstract knobs.
- World side: circadian rest affordance (`rest_basal_mult`, `night_rest_bonus`) — the only
  world change foragers feel.
- Observability: per-robot `resting`/`near_robots`/`near_bushes`; `gol-stats --interests`.

`w_homeostasis` back at 1.0 (escalation to 2.0 held in reserve — the beta_05 lesson).
`inherit_weights: lineage`, population 3 dreamers + 3 foragers, nano, CPU, seed as config.

## Results

### Launch + sanity pass (2026-07-06, ~1.5M ticks)

Launched 10:34 local, headless on the M1, running ~1,200 ticks/s wall (the sim never
waits for the learner). Population pulse at 1.4M: 23 spawns, 17 deaths (11 dreamer),
forager eats 4,333, dreamer eats 81 (≈0.4/day/dreamer — in range of prior early eras),
poisonings 33, ripe stock 217 and cycling.

Per-dreamer aggregates over the trailing 200k ticks (three living: 020/021/022, plus
short-lived 015/017/023 in window):

| check | expected | observed | verdict |
|---|---|---|---|
| `lp_regions` | 32 quickly | 32 on every dreamer | ✓ |
| `boredom` | ≈ 0 for newborns | exactly 0 everywhere (both gates closed; drive_level 0.48–0.70 ≫ 0.15) | ✓ |
| temperament | 3 distinct draws, exact within lineage | 015/021, 017/022, 020/023 pair identically across respawns | ✓ |
| `resting`/`near_*` fields | present | present; `--interests` runs | ✓ |
| LP warm-up | `curiosity_scaled` off the 5.0 clamp after ~100k | 0.10–0.18 | ✓ |
| homeostasis vs stimulation | within an order of magnitude | **borderline — see below** | ⚠ watch |

Early hypothesis signals (all provisional at this age):

- **H1/H2:** actor entropy flat-to-falling (8.6→8.5, 8.1→7.9) — opposite of beta_05/06's
  rise. `drive_level` oscillates (per-agent spans ~0.32–0.88), not trending flat.
- **H3:** `--interests --window 100000` between-agent divergence grows monotonically
  0.098 → 0.19 across windows. Profiles at this age are dominated by dormancy
  (rest ≈ dormant ≈ 0.93), so this is mostly rhythm-difference, not yet "interests."
- Capacity wall, quantified live: ~400 updates against 21–28k buffered act-steps →
  effective train_ratio ≈ 0.016 vs the 0.25 target. The learner is ~15× backpressured on
  CPU. This is the number the cloud round (`beta_08_capacity.yaml`) exists to move.

### The magnitude watch item (top-priority check, beta_05 lesson)

`stimulation` (= `r_cur`, normalized LP clamped [0,5], the curiosity term actually in the
reward) runs sustained at 2.6–3.9. Logged `reward_homeostasis` is a batch mean, so meal
spikes average away: it reads −0.005 to −0.007 (the level-penalty drift) and no sampled
batch exceeded +0.003 in 44k samples. Real comparison: a ~0.6 meal spike against ~3.0
continuous curiosity — homeostasis is ~5× quieter at its loudest instant, ~500× on
average. Strictly fails "within an order of magnitude"; behaviorally not yet a problem
(entropy not rising, drives oscillating, eats normal). Per plan: let the run speak;
`w_homeostasis: 2.0` comes off the bench if eats/day sags by the 3M review.

Second watch item: **the boredom stim gate may be unreachable.** The gate needs
`r_cur < 0.5`, but the LP normalizer holds `r_cur` at O(3) and, being relative +
running-normalized, may never let it drop that low even when learning stalls. If
`boredom` is still exactly 0 at 3M+ while stimulation has sagged, `stim_threshold` is
mis-scaled against the normalized signal.

### Final results (3.44M ticks ≈ 143 sim-days; run stopped cleanly)

Totals: 53 spawns, 47 deaths; dreamer eats 146 (n=33, 30 deaths — 29 hibernation,
1 poison), forager eats 9,976 (n=20, 17 deaths). Forager anchor holds: 9.68 eats/10k
vs beta_06's 8.9, inside the 40% run-to-run variance band round 006 measured; forager
lifespan thirds 22.8→22.1→18.5 days ≈ beta_06's 23.6→19.6→17.4. The world didn't move;
whatever differs below is the reward stack.

Era table (500k-tick windows; per-dreamer eats/day; curiosity_scaled = legacy
*disagreement* normalized — NOT the reward term, see watch item 1):

| era | eats/day | awake | curiosity | homeo reward | entropy | value | model loss |
|---|---|---|---|---|---|---|---|
| 0 | 0.30 | .121 | .0254 | −.0049 | 8.49 | 4.1 | 163 |
| 1 | 0.82 | .168 | .0058 | −.0055 | 8.53 | 13.0 | 75 |
| 2 | 0.22 | .138 | .0042 | −.0058 | 8.34 | 25.6 | 49 |
| 3 | 0.42 | .165 | .0020 | −.0058 | 8.12 | 42.1 | 39 |
| 4 | 0.06 | .092 | .0015 | −.0060 | 8.11 | 59.7 | 34 |
| 5 | 0.32 | .146 | .0012 | −.0059 | 8.11 | 79.8 | 31 |
| 6 | 0.19 | .120 | .0011 | −.0057 | 8.14 | 94.2 | 29 |

Final-era cross-run comparison (3.0–3.44M):

| final era | beta_05 (hungry, v2) | beta_06 (baseline, v3) | beta_06h (hungry, v3) | **beta_07 (gratification, v3)** |
|---|---|---|---|---|
| dreamer eats/day | 0.29 | 0.48 | 0.16 | **0.19** |
| awake fraction | 12% | 16% | 8.7% | **12%** |
| actor entropy | 6.17 (rising) | 9.51 (rising) | 9.19 (rising) | **8.14 (fell 8.49→8.11, flat after)** |
| value | — | 2.9 (drifting up) | −0.47 (went negative) | **94 (still climbing)** |
| total dreamer eats | 196 | 158 | 157 | **146** |
| dreamer lifespan thirds (days) | 14.4→14.4→14.6 | 13.5→14.3→13.9 | 14.5→14.4→13.8 | **14.6→13.1→14.4** |

**Watch item 1 (homeostasis magnitude) — failed, structurally.** `stimulation`
(= r_cur, the LP term actually in the reward) still ran 2.7–3.9 at 3.4M — it never
decayed — while batch-mean `reward_homeostasis` sat at −0.005..−0.008. The mechanism:
`lp.relative: true` makes raw LP a *fraction* of each region's error, and the
capacity-starved model (loss 163→29, still falling; ~970 updates against ~42k
act-steps per life, effective train_ratio ≈ 0.023) keeps making proportional progress
everywhere — so scale-free LP never goes stale, and the std-only RunningMeanStd holds
the normalized signal at O(3) indefinitely. The critic priced it: value climbed
monotonically to era-mean 94 (live agents 47–129), i.e. the return is ~99% perpetual
curiosity. Against that, a +0.6 one-step meal spike is invisible — ~5× quieter at its
loudest instant, ~500× on average, exactly the failure the plan flagged, and one
`w_homeostasis: 2.0` cannot close. Note this is *not* beta_06's phantom-curiosity
(stale replay valuing a dead signal): the progress is real; it just never runs out on
CPU. Legacy disagreement (`curiosity_scaled` 1.20→0.06) decayed on schedule — the LP
formulation specifically is what can't sag while the model is far from converged.

**Watch item 2 (boredom gate) — confirmed unreachable; boredom was inert all run.**
`boredom` read exactly 0.0 on every dreamer at every sample. Both gates stayed shut:
the stim gate needs r_cur < 0.5 but the normalizer holds r_cur ≈ 3 (as predicted at
the sanity pass); the drive gate needs drive_level < 0.15 but observed drive_level
never left 0.32–0.88 (with setpoints 0.85/1.0/1.0 and convex pooling, a body that
eats ~0.2 meals/day is never "sated"). The boredom mechanism contributed nothing to
this round — neither harm nor play pressure.

**H1 (motivation persists) — partial.** The monotone motivational sag of
beta_05/beta_06h did not reproduce: eats/day oscillates without trend (era 4 dipped
to 0.06, era 5 recovered to 0.32) and awake fraction holds 0.09–0.17 with no slide
(beta_06h fell monotonically 0.143→0.087). But levels didn't improve — final era 0.19
vs beta_06's 0.48 is a wash inside the ±0.2 noise band, totals 146 vs 158 — and the
landscape stayed un-flat for the wrong reason (LP can't decay, per watch item 1),
so this is weak evidence for the design and strong evidence about capacity.

**H2 (bodily rhythm) — half a pass, with a surprise.** `drive_level` oscillates
per-agent (spans 0.32–0.88) rather than trending — need→act→sate exists. Actor
entropy *fell* 8.49→8.11 then held — the first beta run where it didn't rise
(beta_06: 9.22→9.51; beta_06h: 8.65→9.19; comparable action space) — the drive
gradients are being used, not diffused over. But the sleep-at-night prediction came
out *inverted*: pooled across dreamers past 1M ticks, awake-resting correlates
*positively* with light (+0.115; resting 6.2% of awake day-samples vs 0.4% at night)
while dormancy skews to night (corr −0.30; 95% dormant at night vs 76% by day).
Agents rest in daylight — consistent with `solar_trickle` making day-rest
net-cheaper than night-rest's 2× recovery bonus, though not proven — and the night
dormancy is largely mechanical (energy runs out; `regrow_daytime_only` means night
offers nothing). Circadian structure emerged; the phase is the opposite of the
affordance we advertised.

**H3 (interests, not noise) — qualified yes.** Between-agent divergence grew
0.098→0.19 over the first 1.5M then plateaued, fluctuating 0.14–0.24 through 3.4M
(no collapse — the replay-turnover fear didn't materialize). Within-agent stability
for agents with ≥4 windows sits 0.84–1.00 vs the newborn ~0.5 baseline. And the
profiles are genuinely differentiated, not rhythm-only: dedicated foragers
(dreamer_023 forage 0.84, _052 0.85, _041 0.79), social specialists (dreamer_006
near-robots 0.52, _007 0.25, _044/_045 0.23), and mixed dig/place dabblers
(dreamer_008, _013). Individuality is real at this capacity; where it comes from is
H4's surprise.

**H4 (temperament shows through) — the strongest result in the round, with a
confound.** Mapping all 33 dreamers to their three lineages by temperament
fingerprint: lineage B (temperament_w_curiosity 1.37) averaged **0.51 forage**
across its 11 members vs 0.30 / 0.27 for lineages A (w_cur 0.76) and C (w_cur 1.42)
— and produced *zero* social members, while A and C produced all seven
(near-robot fractions 0.15–0.52). These are coherent, lineage-consistent behavioral
signatures persisting across ~11 successive respawns each. The confound: lineage
carries both the temperament draw *and* the inherited weights, so this could be
temperament expression or cultural transmission through warm-starts — round-question
4 territory either way, and exactly what the `random_living` round can split.
n=3 lineages remains anecdote; the within-lineage spread is wide (lineage B spans
forage 0.004–0.85).

## Interpretation

The stack's parts each did their job: LP flows and is path-dependent, drives
oscillate, temperament is exact across checkpoints, the observability works. What
failed is the *balance between them*, and it failed for the round-006 reason, not a
new one. The design assumes interest goes stale as niches are mastered — that LP
sags, letting homeostasis and boredom take over the stage. But mastery requires
learning capacity, and at ~0.023 effective train_ratio the model never converges on
anything, so relative LP pays out ~3 per step forever, the critic inflates toward
value ≈ 100, and the hunger economy stays a rounding error. beta_07 is stuck in
permanent adolescence: everything is still learnable, so nothing else ever matters.

The right read is that beta_07 doesn't refute the gratification design — it shows the
design's preconditions aren't met on nano/CPU. The one lever that changes the
preconditions is capacity (beta_08). If, with a base preset and train_ratio 1.0, the
model actually converges regionally, LP should genuinely decay there, the stim gate
becomes reachable, and the drowned-homeostasis gap closes on its own; if instead LP
stays pinned at the clamp even while regions master, the LP normalization itself
(std-only, relative) is mis-designed and needs rework before any more reward tuning.
Meanwhile the round paid for itself twice: entropy falling for the first time in the
series says drive gradients are usable even at this capacity, and the
lineage-signature result says persistent behavioral individuality is already here —
arriving through inheritance before it arrived through interest.

## Caveats

- Logged `reward_homeostasis` is a training-batch mean — it cannot show meal spikes; the
  drift/spike decomposition above is inferred from the drive-reward design plus max/min
  scan. A per-step homeostatic-reward histogram would settle it properly.
- The era table's `curiosity_scaled` column is the *legacy disagreement* normalizer, not
  the LP reward term — its 1.20→0.06 decay is what the old signal would have done. The
  reward term is `stimulation`/`lp_reward` in the per-brain metrics (held 2.7–3.9).
- The H2 sleep test uses awake-resting (resting AND NOT dormant) vs `light`, pooled over
  ticks ≥ 1M; dormancy still contaminates the picture since hibernation is
  energy-forced, not chosen.
- The H4 lineage signatures confound temperament with inherited weights (both travel
  with lineage under `inherit_weights: lineage`); splitting them needs `random_living`.
- Single run; round 006 measured 40% forager variance between identical-config runs.
  Trust qualitative signals (value sign, entropy drift, within-run trends) over ±0.2
  eats/day deltas.
- Entropy levels are comparable to beta_06/beta_06h (same obs v3 action space) but not to
  beta_05.

## Next

- **Run the capacity round** — `configs/run/beta_08_capacity.yaml` +
  `configs/brain/beta_08_dreamer.yaml` (base preset, train_ratio 1.0, cuda; reward stack
  byte-identical to beta_07's). beta_07 is its local A/B; foragers anchor. It now
  doubles as the test of the LP-decay hypothesis: if LP sags where the model converges,
  the gratification balance engages as designed; if LP stays pinned, the normalization
  (std-only, relative) needs rework before further reward tuning.
- **Config debts for whichever round follows beta_08** (not worth a local rerun now):
  `boredom.stim_threshold` must be scaled against observed normalized r_cur (0.5 vs an
  O(3) signal is unreachable); `boredom.drive_threshold: 0.15` is likely too strict
  when observed drive_level bottoms at ~0.32 — consider gating on drive *slope* or a
  higher threshold; consider mean-relative (not std-only) LP normalization so "no
  progress" maps to ~0.
- Round-007 code is committed (cb8f632); commit this finalized entry + README index row.
- **Post-review discovery (beta_08 prep):** `training.train_ratio` was never enforced —
  no code read it. The learner thread round-robined one update per brain per
  ~1s-minimum round regardless of hardware, so beta_07's effective ratio ~0.023 was
  partly an artificial throttle, not pure compute; and on a GPU the throttle would have
  capped updates at ~1/s/brain, making the staged beta_08 a placebo. Fixed post-007:
  the learner now paces to train_ratio against lived act-steps (debt-capped —
  backpressure still skips, never stalls), and the round exposed a second artifact now
  measured: `act_latched_frac` ≈ 0.23 in an unpaced M1 smoke — roughly a quarter of a
  sprinting dreamer's act-steps repeat a latched command while its brain is mid-update.
  Treat beta_07 behavior data accordingly; paced runs shrink this to ~0. The learner
  also went one worker per brain (siblings learn concurrently; ~1.7× total throughput
  on shared M1 cores, ~3× expected on GPU) — thread-agnostic learn() math, so the
  07↔08 A/B is unaffected; torch.compile deferred since kernel fusion changes numerics.
  New instruments that ship with the fix: `train_ratio_eff`, `act_steps`, `learn_seconds`,
  per-region LP stats (`lp_p50/p90`, `lp_stale_frac`, `lp_occ_entropy`), homeostatic
  spike stats (`homeo_max`, `homeo_spike_frac`), boredom gate telemetry
  (`boredom_calm_gate`/`dull_gate`), and `gol-stats --circadian`. beta_08 must run
  *paced* — the ratio-vs-world-speed math is in `beta_08_capacity.yaml`'s header.
- Free findings to keep: the day-rest/night-dormancy inversion (solar trickle beats the
  night recovery bonus as an affordance — if "sleep at night" matters, the economy
  advertises the wrong phase); lineage-consistent behavioral signatures under lineage
  inheritance (cultural-transmission signal, research question 4). `--circadian` adds
  the control-arm contrast: scripted foragers awake-rest at *night* (corr −0.89 with
  light, night_frac 0.90 — the fixed policy idles when no food regrows), so the
  night-rest phase is available and mechanically sensible; the dreamers' day-rest is a
  choice, not a constraint.
- After close: prune beta_* save dirs per round 006's note (keep manifest +
  events/metrics or just this entry's numbers).
