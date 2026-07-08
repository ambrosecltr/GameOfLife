---
round: 011
title: the reachability round — price the blackout, feed the reward head
date: 2026-07-07            # pre-registered at build/launch; closed 2026-07-08
status: closed
question: With the conditioning stack held fixed at beta_09 and only reward *reachability* changed — the dormancy blackout priced as one visible HRRL transition, and reward-aware replay feeding salient steps to the twohot head — does eating finally rise while hungry (not only while bored-sated), do lifespans leave the hibernation clock, and does poison avoidance get a value gradient?
headline: "Reachability exonerated: the head learned the loud moments (spike err 0.25→0.06), the crash was priced honestly — and of 105 dreamer meals exactly 1 was eaten hungry. Boredom feeds; hunger still can't. Binding constraint moves to the actor/affordances fork."
runs:
  - save: saves/beta_10
    config: configs/run/beta_10_reachability.yaml
    brain: configs/brain/beta_10_dreamer.yaml
    commit: 5dae6f2
    ticks: 2676729
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

Config (all in `beta_10_dreamer.yaml`):

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

Plus one screened-in knob, same constraint: `reward.spike_loss_weight: 4`
(spikes count 5× in the twohot reward loss) — journal 010's pre-registered
follow-up, taken because the gym showed the head fits spikes better under
weighting *even when oversampled* (see Screens). Three knobs, one indicted
constraint — the round-009 precedent; per-knob ablation flags remain.

Deliberately untouched: the conditioning stack (it works), capacity bundle,
world, population, seed protocol.

## Screens (offline gym, dreamer_043's life, --fresh-model, nano, seed 0)

Four arms × 2000 updates, replay shaped to beta (16×64, no burn-in/recent),
the only surviving beta_09 blob — n=1 life, wake-spike-dominated buffer, so
these earn the knobs, they don't license behavioral claims:

Windowed mean `reward_head_spike_err` (0–500 / 500–1000 / 1000–1500 /
1500–2000 updates):

- A `prioritize=none`: 0.81 / 0.78 / 0.76 / **0.68**
- B `prioritize=reward`: 0.68 / 0.65 / 0.61 / **0.57** (−16% vs A, every window)
- C = B + `spike_loss_weight=4`: 0.68 / 0.63 / 0.56 / **0.50** (−12% vs B,
  best in every window; its raw loss_reward is 2× B's by construction —
  the weighted metric isn't comparable across arms)
- D = B + `blackout=priced`: ≡ B to the 4th decimal on reward metrics —
  the cont→1 change is orthogonal to the reward head, machinery clean.

Screen caveats: the blob predates salience and break markers, so backfill
fakes 8 rebirth spikes among the ~271 salient steps (documented, kept — the
screen question is whether the head can fit spikes when oversampled, not
their provenance), and uniform coverage here is ~90% of windows (wake-dense
life) where a foraging life would be ~1% — prioritization's marginal value
on meals is understated by this data.

## Predictions (written before launch)

- **P1 — the head learns the loud moments.** `reward_head_spike_err` falls
  within-run (`spike_row_frac` ≥ 0.25 by construction). If it stays flat
  with spikes flowing and weighted, reachability's head-side mechanism is
  falsified at base scale and P5 branches on the actor instead.
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

Ran 2,676,729 ticks paced at speed 2 on the 3090 Ti pod (~$8, ~30h wall).
Clean close: on-demand checkpoint, single SIGINT, final atomic checkpoint at
2,676,729. Pacing identity held the entire run (updates == act_steps − 500
per brain at every checkpoint; `train_ratio_eff` 0.97–0.98 at close; speed 2
never had to drop — dormancy stayed high enough to bank sleep-learning).
27 dreamer lives, 24 dreamer deaths, 44 spawns total.

- **P1 CONFIRMED — the head learns the loud moments.** Mean
  `reward_head_spike_err` fell 0.25 → 0.06 by 800k and held there across
  every subsequent generation (0.03–0.08 per brain at close, ~10× better
  than founders; beta_09 had no such convergence). `spike_row_frac` ran
  0.31–0.69 all run. Both replay knobs did their mechanical job.
- **P2 FALSIFIED — collapse stopped paying and dormancy did not
  restructure.** 23/24 dreamer deaths hibernation-dominated (beta_09:
  23/26); median death age 342.6k with 13/24 within 3% of the ~347k
  hibernation clock; aggregate dormant fraction 0.84–0.91 with a late-run
  *rise* (0.91 from 1.6M on). Removing the wake subsidy (+1.2) and rebirth
  jackpot (+3.9) and pricing the crash honestly did not weaken the
  attractor. A transient generational signal (first children barely
  hibernated in their first ~100k) regressed as they aged. The one
  exception: dreamer_007 died of *fall damage* at age 97k — the first
  behaviorally-caused dreamer death in project history.
- **P3 FALSIFIED — the sharpest result of the round.** Joining every
  dreamer eat with the eater's energy at that moment: **105 meals, 104
  eaten sated, 1 eaten hungry** (energy < 40). A binge did fire — earlier,
  stronger, and longer than beta_09's (800k–1.5M, peaking 14–18 eats/200k,
  dominated by two brains: dreamer_017 21 eats, dreamer_018 24) — and
  collapsed at 1.6M exactly like beta_09's, back to 1–7/200k. Poisoned-meal
  fraction 24/105 = 23%, statistically unmoved from ~26%: no avoidance
  gradient emerged even with meals representable.
- **P4 AMENDED — the thermostat replicated at a higher setpoint.** Pressure
  charged sooner than beta_09 (0.2–0.3 by 500–800k vs 1.8M+) but did not
  equilibrate at ~0.39: it climbed to 0.50–0.57 (max 0.572) with
  stimulation pinned at 0.02–0.08 late-run. Not saturation — a higher
  equilibrium: the discharge loop (boredom → action → new experience)
  closes less effectively in this world-state. Anchored normalizers held
  (`curiosity_scaled` 0.87–1.16 at close); no treadmill regression.
- Free riders: terraforming remains dreamer-only and grew (447 digs / 426
  places vs beta_09's 325/298), again rising with the binge. The critic
  values of high-pressure/low-stim brains collapsed (30–77 vs 330–358 for
  stimulated siblings) — a "nothing is worth doing" value signature worth
  instrumenting properly. Forager ledger produced two project firsts:
  forager_010 died of *poison* at age 836,944 (2.4× the hibernation clock,
  longest life recorded) and forager_027 of *wear* at 767,713 — scripted
  behavior→survival coupling now spans three death modes.
- **Anchor caveat that is itself a finding:** scripted-forager intake
  oscillated 4–8× between generations (1264 → 145 → 198 → 964 per 200k...)
  with 300+ ripe bushes standing; entire forager generations starved into
  hibernation deaths (12/14 forager deaths) depending on spawn geography.
  A *perfect* policy's caloric intake is spawn-luck-dominated in this
  world. Meal affordances are patchy at the lifetime scale.

Blobs archived before pod teardown: mid-run trio at 390k
(`saves/archive/beta_10_blobs_390k`) and the final trio + foragers from
ckpt_2676729 (`saves/archive/beta_10_blobs_final`; dreamer_042 is the
designated screen blob — the healthy/stimulated one, value 358 at close).
Full save (metrics/events/manifest, brains excluded) mirrored to
`saves/beta_10`.

## Interpretation

The round did exactly what a falsification round should: **both reachability
mechanisms were installed, verified working, and behavior did not move.**
The reward head demonstrably represents meals and crashes now (P1); the
hibernation ratchet lost its replay subsidy and the crash carries an honest
HRRL price into training — and dreamers still eat only when bored-sated,
still die on the hibernation clock, and still eat 1-in-4 poison. Reward
reachability joins capacity (008) and signal conditioning (009) as
necessary-but-not-sufficient. Three rounds, three clean exonerations.

Per pre-registered P5, the binding constraint moves to the fork: **actor
credit-assignment** (the policy cannot cash a represented sparse value into
a multi-step foraging behavior when drive is high) vs **world affordances**
(meals too rare/patchy for *any* policy to couple eating to survival).
Both suspects gained evidence this round: the forager-generation starvation
oscillation says affordances are spawn-luck-dominated even for a perfect
policy; the collapsed critic values at high pressure say the actor's
imagination may genuinely price "nothing worth doing" (the pre-registered
value-at-low-energy introspection remains the discriminator to instrument).

The deeper structural reading (user's framing at close, now the round-012
hypothesis): **boredom and hunger are over-coupled as drives.** Boredom is
doing all the behavioral work — it trains foraging-while-sated, then gates
off exactly when hunger rises ("an agent in need is never bored," round
009), so eating never binds to survival. What's being learned is
"eating is entertainment," not "eating is how I keep existing." No
mortality-shaped credit ever forms, because the only drive that fires near
death (hunger) has never successfully driven the behavior that averts it.
The 94:1 sated:hungry ratio is that hypothesis stated as a measurement.

## Next

Round 012 direction (user, at close):

1. **Affordances arm: raise food production** (candidate: ~2× bush
   density/regrowth). If eating-while-hungry appears when meals are dense,
   the constraint was affordances and can then be titrated back down;
   if 94:1 persists in abundance, the actor/drive-coupling explanation
   stands alone. Cheap, decisive, and it directly tests the
   forager-starvation-oscillation finding.
2. **Decision-forensics arm: instrument how the actor decides.** Before
   more reward surgery, *look* at the decision: imagination value at
   low-energy states (does it go negative now that the crash is priced?),
   per-drive reward decomposition along real trajectories into and out of
   hunger, actor logits at hungry-near-food moments. The
   boredom/hunger-coupling hypothesis needs its own census-grade
   measurement, not another knob.
3. **Scale question: consider running 012 on nano brains.** Rationale
   (user): eat-to-survive is a basic mental function that shouldn't require
   large reasoning capacity; if mortality-shaped behavior can emerge on
   nano, base-sized brains are then free to host the complex emergence
   (communication, social) on top of it. Tension to resolve honestly at
   staging: round 008 showed nano never converges the obs-v3 world model
   (relative-LP never decays, conditioning preconditions unmet) — a nano
   round must either accept a converged-enough criterion short of base, or
   pair with the food/affordance change so the survival loop is learnable
   even under a rough model. Nano also runs local (M1), which changes the
   ops calculus entirely.

Out of scope carried forward: dormancy-duration proprio (OBS_VERSION 4),
death remains unexperienced, Stage 2 camera vision still pending.
