---
round: 006
title: obs v3 and the capacity wall
date: 2026-07-06
status: complete
question: Do richer senses (color vision, gaze control) change life under both reward regimes — and does the round-005 hunger effect carry over to obs v3?
headline: Richer senses tripled scripted foraging but sank the learners — the hunger effect of round 005 did not replicate under obs v3; the critic learned the future is hungry (value went negative) and the policy still couldn't act on it. Binding constraint confirmed as learning capacity, not reward design.
runs:
  - save: saves/beta_06
    config: configs/run/local_social.yaml
    brain: configs/brain/dreamer.yaml
    commit: "2202117"
    ticks: 3361800
    role: control (curiosity-only baseline; run by config mistake, kept as the paired arm)
  - save: saves/beta_06h
    config: configs/run/local_hunger.yaml
    brain: configs/brain/dreamer_hungry.yaml
    commit: "2202117"
    ticks: 3440100
    role: experiment (run from a clean worktree at 2202117 to avoid uncommitted round-007 world changes)
baselines: [004, 005]
tags: [vision, motivation, capacity, variance]
---

# 006 — obs v3 and the capacity wall

## Why this round

Obs v3 (2202117) is the biggest embodiment change since the beta series began: rays now
carry depth + shaded RGB + a hit-kind one-hot, gaze control adds 2 action dims and 2
proprio dims, and the world gained visual grain. OBS_VERSION 2 → 3 (older brains won't
load). The intended experiment was round 005's hunger config under the new senses;
beta_06 was accidentally launched with `local_social` instead, which turned out to be
useful — it makes a clean paired ablation at one commit: beta_06 (curiosity-only) vs
beta_06h (hungry), the same design as rounds 004 vs 005.

## What changed

- Obs v3 (commit 2202117): color vision (toxic vs ripe distinguishable only by color;
  other robots perceptually salient as "alive"), gaze control (±45° pitch, ±90° yaw
  head), world grain. World model gains RGB + kind prediction heads.
- beta_06h reward = round 005's `dreamer_hungry.yaml`, verified live in the data:
  homeostatic reward −0.072..−0.087 (matches beta_05's −0.07..−0.08; beta_06's baseline
  reward sat at −0.003..−0.004, the ~25× gap expected from the config).
- Everything else identical to rounds 004/005 (same world yaml, population 3+3, nano,
  CPU, train_ratio 0.25, seed 7).

## Results

### Foragers (control policy): obs v3 is a large body win, with worse selectivity

- Eat rate 2.54 → 8.89 eats/10k ticks vs beta_05 (2,640 → 8,948 total); first-cohort
  lifespans 16.3 → 23.6 days. Gaze/color made the fixed policy dramatically better at
  finding food.
- Per-eat poison rate doubled (0.30% → 0.60%; forager poisonings 8 → 54). Gaze finds
  toxic bushes just as well as ripe ones; color alone bought no selectivity.
- The ecology held under ~3.5× grazing pressure: no ratchet; ripe stock ran lower than
  beta_05 (era means 222–307 vs a flat ~326) with toxic share ~20% vs ~16%, but
  *recovered* over the run (222 → 293). Forager eats/day fell 39 → 11.5 across eras
  with cohort lifespans 23.6 → 19.6 → 17.4 — an early gorge on the initial standing
  crop settling toward a leaner equilibrium.

### Dreamers: the hunger effect did not replicate

Final-era (3.0–3.5M) comparison across the three runs:

| final era | beta_05 (hungry, v2) | beta_06 (baseline, v3) | beta_06h (hungry, v3) |
|---|---|---|---|
| dreamer eats/day | 0.29 | 0.48 | **0.16** |
| awake fraction | 12% | 16% | **8.7%** |
| lifespan trend (thirds, days) | 14.4→14.4→14.6 | 13.5→14.3→13.9 | 14.5→14.4→13.8 |
| actor entropy | 6.17 (from 6.04) | 9.51 (warmup 9.22) | 9.19 (from 8.65) |
| total dreamer eats | 196 | 158 | 157 |

beta_06h era table (500k-tick windows; per-dreamer eats/day, mean entropy/value):

| era | eats/day | awake | curiosity | homeo reward | entropy | value | model loss |
|---|---|---|---|---|---|---|---|
| 0 | 0.40 | .143 | .0244 | −.072 | 8.65 | 1.69 | 164 |
| 1 | 0.64 | .190 | .0056 | −.081 | 8.81 | 1.79 | 82 |
| 2 | 0.24 | .124 | .0047 | −.084 | 9.05 | 1.65 | 57 |
| 3 | 0.48 | .126 | .0030 | −.087 | 8.97 | 1.35 | 47 |
| 4 | 0.34 | .127 | .0021 | −.083 | 8.98 | 0.75 | 42 |
| 5 | 0.26 | .116 | .0016 | −.086 | 9.07 | 0.04 | 39 |
| 6 | 0.16 | .087 | .0014 | −.085 | 9.19 | −0.47 | 36 |

- Hungry vs baseline totals are identical (157 vs 158 eats), and the hungry arm ended
  the run eating less and sleeping more than any run to date. All 30 dreamer deaths in
  beta_06h were hibernation.
- **The critic went negative** — the only run where value crossed zero (1.69 → −0.47,
  monotone from era 1) while the hunger penalty held steady at −0.085. Contrast beta_06,
  where value drifted *up* to 2.9 on phantom curiosity (stale replay valuing a signal
  that no longer exists). The hungry critic correctly learned the future is net-negative:
  persistent hunger, no curiosity left, no meals coming. The gradient is not just present
  in the reward stream — it is *perceived* — and the policy still can't convert it.
- World model handled v3 fine: loss 164 → 36 (plateau ~2× beta_05's 19 — RGB + kind
  heads add residual burden), depth error → 0.032 vs beta_05's 0.027.
- Normalized curiosity ended lower than ever (beta_06h/beta_06 ~0.07 vs beta_05's 0.15).

## Interpretation

Obs v3 raised the capacity bar while capacity stayed fixed. The extra richness that
tripled the *scripted* foragers' intake is pure cost to an update-starved learner: more
to predict (double the residual model loss), two more continuous action dims to search,
same nano/CPU/train_ratio-0.25 budget of a few hundred updates per lifetime. Round 005's
partial win was eaten by the bigger problem. This *strengthens* the round-005 conclusion:
reward design is no longer the binding constraint — learning capacity and experience
density are. Making hunger louder cannot help a policy that gets too few updates to learn
what food is worth in a bigger observation space.

## Caveats

- **Run-to-run variance is large.** The scripted foragers — identical policy, commit,
  and world config in beta_06 vs beta_06h — differed by 40% (8.9 vs 5.3 eats/10k;
  8,948 vs 5,454 total). The shared world diverges chaotically once dreamer actions
  differ. Dreamer deltas of ±0.2 eats/day between single runs are noise; trust the
  qualitative signals (value sign, entropy drift, awake fraction, within-run trends).
- beta_06 vs beta_05 confounds obs version with reward config; the valid pairings are
  beta_06 vs beta_06h (reward, same commit) and beta_06/beta_06h vs beta_04/beta_05
  (obs v3, matched reward).
- Entropy scales aren't comparable across obs versions (gaze added 2 action dims);
  within-run trends are.
- beta_06h was run from a clean worktree at 2202117; the main tree already carried
  uncommitted round-007 changes (circadian rest is world-side and on by default — it
  would have silently broken the pairing).

## Next

- The capacity experiment: cloud round with small/base preset and higher train_ratio,
  same paired configs — unchanged from round 005, now better motivated.
- Round 007 (the gratification stack: LP curiosity, HRRL drive-reduction, boredom,
  temperament — see `docs/beta07-gratification-plan.md`) attacks the drive side
  structurally instead of turning the hunger volume further. Code complete and
  uncommitted at write-up.
- Free findings to keep: color alone buys no selectivity (poison rates rose for both
  brains); the ecology survives ~3.5× grazing pressure; the critic-vs-reality gap
  (beta_06's rising value on collapsed reward) is a stale-replay artifact worth watching.
- After round 007: prune beta_* save dirs (drop checkpoints/brains, keep manifest +
  events/metrics or just this entry's numbers).
