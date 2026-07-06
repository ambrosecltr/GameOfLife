---
round: 007
title: the gratification round — do agents develop interests?
date: 2026-07-06            # launch + sanity pass; results pending (target ≥3M ticks)
status: running
question: Does replacing surprise-curiosity with the gratification stack (LP curiosity, HRRL drive reduction, boredom, temperament) produce persistent motivation, bodily rhythm, and individual interests?
headline: pending — sanity pass at 1.5M all green (entropy not rising, drive_level oscillating, interest divergence growing 0.10→0.19); open watch items are the homeostasis/stimulation magnitude gap and a possibly unreachable boredom stim gate
runs:
  - save: saves/beta_07
    config: configs/run/local_social.yaml
    brain: configs/brain/dreamer.yaml
    commit: TBD             # the round-007 commit — the commit that introduces this entry
    ticks: 1500000          # in progress at sanity pass; target ≥3M
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

### Final results

*(to be filled at ~3M: era tables via `tools/era_stats.py`, eats/day + awake-fraction
trajectories vs beta_06/beta_05, entropy trend, `--interests` stability + divergence,
temperament↔profile correlations, forager anchor check)*

## Interpretation

*(pending run completion)*

## Caveats

- Logged `reward_homeostasis` is a training-batch mean — it cannot show meal spikes; the
  drift/spike decomposition above is inferred from the drive-reward design plus max/min
  scan. A per-step homeostatic-reward histogram would settle it properly.
- `resting` is inferred from low motor activity and includes hibernation dormancy
  (scripted foragers read 0.72 "resting"). H2's sleep-at-night test needs awake-resting
  vs `light`, not the raw flag.
- Single run; round 006 measured 40% forager variance between identical-config runs.
  Trust qualitative signals (value sign, entropy drift, within-run trends) over ±0.2
  eats/day deltas.
- Entropy levels are comparable to beta_06/beta_06h (same obs v3 action space) but not to
  beta_05.

## Next

- 3M review: the two watch items above, then fill Results/Interpretation, set
  `status: complete`, write the headline, add the README index row.
- Cloud capacity round queued behind this one: `configs/run/beta_08_capacity.yaml` +
  `configs/brain/beta_08_dreamer.yaml` (base preset, train_ratio 1.0, cuda; reward stack
  byte-identical to beta_07's). beta_07 serves as its local A/B; foragers anchor.
- After close: prune beta_* save dirs per round 006's note.
