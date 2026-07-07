---
round: 011
title: the reachability round — price the blackout, feed the reward head
date: 2026-07-07            # pre-registered at build/launch
status: running
question: With the conditioning stack held fixed at beta_09 and only reward *reachability* changed — the dormancy blackout priced as one visible HRRL transition, and reward-aware replay feeding salient steps to the twohot head — does eating finally rise while hungry (not only while bored-sated), do lifespans leave the hibernation clock, and does poison avoidance get a value gradient?
headline: ""
runs:
  - save: saves/beta_10
    config: configs/run/beta_10_reachability.yaml
    brain: configs/brain/beta_10_dreamer.yaml
    commit: pending          # set at launch
    ticks: pending
    role: experiment
baselines: [009, 008]
tags: [reward, reachability, replay, dormancy]
---

# 011 — the reachability round

## Why this round

Round 009 ended with the handoff-within-the-handoff: curiosity hands off to
boredom, boredom drives a real foraging binge, and hunger fumbles the catch —
the binge dies exactly when the calm gate closes and homeostasis has to carry
the behavior alone. The verdict was that HRRL semantics are right but the
reward is structurally unreachable: (a) the dormancy blackout never enters
the lived stream, and (b) meals are too rare for the twohot reward head.

**A census of beta_09's only surviving brain blob (dreamer_043, 9,035-step
buffer) sharpened (a) into something worse than invisibility.** The lineage
replay ring is continuous across stream breaks and the HRRL reduction was
computed across the gaps, so replayed windows were *paying* the collapse:

- ~130 wake transitions (energy ~0 → 0.4) each read as **+1.2 reward** —
  and with wakes ~70 steps apart, nearly every 64-step window contains one;
- 8 death→rebirth stitches (dying body's last step followed by the
  newborn's full tank) each read as **+3.9** — in ~61% of batches;
- the two real meals in the buffer were worth +0.54 and +0.34.

The reward head's targets were dominated by hibernation-recovery jackpots.
"Starving leads to a cut" was wrong twice over: replay taught *collapse →
free energy, occasionally → reborn fully fed*. This is a candidate
mechanistic explanation for the stability of the hibernation attractor
across beta_07/08/09, and it reframes the round: not just making the crash
visible, but making its price *honest*.

## What changed vs beta_09 (the only knobs)

Config (both in `beta_10_dreamer.yaml`):

- `reward.blackout: priced` — on wake, the pre-collapse state is the
  predecessor of the wake observation: one visible transition carrying the
  gap's real energy/integrity delta, priced by HRRL with no new reward
  terms. Four deliberate sub-decisions (see brain.py): salience survives
  the wake (reward-aware replay can find every blackout); the buffer keeps
  the pair adjacent (both modes); the live recurrent state still resets
  (the mind was off); and blackout stops training as a termination
  (cont → 1 everywhere — imagination must be able to traverse the crash,
  or the priced reduction can never reach the actor).
- `replay.prioritize: reward` — 4 of 16 batch rows drawn from windows
  containing a salient step (|realized drive delta| > 0.1). Changes what is
  learned from, never what is rewarded.

Code (all modes, effectively a semantics bugfix, commit 4472684): stream
breaks are now marked in the buffer (`is_first`) and the HRRL reduction is
zeroed across them — respawn always, wake only under `cut`. So beta_10's
baseline is *cleaner* than beta_09's, and the census numbers above are the
measured size of the artifact this removes.

Deliberately untouched: the conditioning stack (it works), capacity bundle,
world, population, seed protocol. `spike_loss_weight` stays 0: in the gym
screen, prioritization alone moved the head's spike error and adding loss
weighting on top gave no additional improvement (see Screens).

## Screens (offline gym, dreamer_043's life, --fresh-model, nano, seed 0)

Four arms × 2000 updates, replay shaped to beta (16×64, no burn-in/recent),
the only surviving beta_09 blob — n=1 life, wake-spike-dominated buffer, so
these earn the knobs, they don't license behavioral claims:

- A `prioritize=none`  vs  B `prioritize=reward`: RESULTS_PLACEHOLDER_AB
- C = B + `spike_loss_weight=4`: RESULTS_PLACEHOLDER_C
- D = B + `blackout=priced` (machinery smoke: cont→1 path runs): finite
  losses, no divergence.

Screen caveats: the blob predates salience and break markers, so backfill
fakes 8 rebirth spikes among the ~271 salient steps (documented, kept — the
screen question is whether the head can fit spikes when oversampled, not
their provenance), and uniform coverage here is ~90% of windows (wake-dense
life) where a foraging life would be ~1% — prioritization's marginal value
on meals is understated by this data.

## Predictions (written before launch)

- **P1 — the head learns the loud moments.** `reward_head_spike_err` falls
  within-run and stays below beta_09-era levels once spikes flow
  (`spike_row_frac` ≥ 0.25 by construction); `loss_reward` does not
  regress. If spike error stays flat with spikes in every batch at base
  scale over a long life, the pre-registered follow-up is
  `spike_loss_weight` (screened: no harm, no gym-visible gain).
- **P2 — collapse stops paying, dormancy restructures.** With the wake
  jackpot replaced by the honest net delta (energy recovery MINUS the
  integrity crash) and rebirth stitches zeroed, the collapse-wake-collapse
  ratchet loses its subsidy. Expect: mean dormant fraction falls or dormant
  spells consolidate (fewer, longer, deliberate — e.g. night-aligned)
  instead of the chronic ~70-step crash cycle dreamer_043 lived; the
  hibernation-ledger share of deaths falls below beta_09's 23/26.
- **P3 — eating while hungry.** The win condition, inherited from 009: an
  eats-per-200k rise that *survives the calm gate closing* (009's binge
  died there), i.e. sustained eating during high-drive stretches, not only
  the bored-sated binge. Poisoned-meal fraction finally moves off ~26%
  (it now has a value gradient against it once meals are representable).
- **P4 — the conditioning stack replicates.** Boredom pressure equilibrates
  near ~0.39 (thermostat); if it saturates instead, something regressed.
  `curiosity_scaled` stays flat after the anchor freezes; `lp_mix_eff` → 0
  by ~1500 act-steps for founders.
- **P5 — the falsification branch.** If the head fits spikes (P1) and
  collapse stops paying (P2) but eating-while-hungry still fails (P3), then
  reachability was not the binding constraint either; the remaining
  suspects are the actor's ability to cash a *represented* sparse value
  (policy/credit assignment) vs the world's affordances (meal density/
  geometry). The dreamer/forager eat gap and whether imagination value at
  low-energy states turns negative (it should, once the crash is priced)
  say which.
- Free riders to watch: terraforming under pressure (rose with the binge in
  009); the synchronized population sleep (bucket-13 phenomenon) — a priced
  blackout should perturb it if it was hibernation-economics-driven;
  whether `cont → 1` changes imagination value scale visibly at the anchor.

## Operations

RTX 3090 Ti pod (pwt3qk9qkdm3cn, $0.27/h, direct TCP SSH so plain
rsync/sync_back work). Benchmarked on-box at the round config (2026-07-07):
solo 175 ms/update; **3-worker contention 0.487 s/update/brain** (2.05
upd/s/brain, ~6.1 aggregate) — the honest pacing number. At ratio 1.0 an
awake brain needs tick_rate/5 upd/s: all-3-awake continuously sustains only
~10 t/s, beta_09's ~90% dormancy sustains 60+. Launch at speed 2 (40 t/s)
and adjust from live `train_ratio_eff` — P2 predicts dormancy falls, which
*cuts* sleep-learning headroom; the pacing budget is itself part of the
experiment's telemetry this round. Verify `updates == act_steps − warmup`
at first checkpoint. Rerun OFF. Paced, never headless. Sync-back loop from
the laptop (checkpoint mirror excludes brains; pull ≥1 dreamer blob before
the pod dies — the beta_09 lesson: only dreamer_043 survived, and the
death of 045/046 cost this round two screening lives).

## Results

*(pending)*

## Interpretation

*(pending)*

## Next

*(pending close)*
