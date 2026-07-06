---
round: 008
title: the capacity round — does a mind that can converge develop taste?
date: 2026-07-06            # launched 2026-07-06, closed 2026-07-07 (host reboot at 2.005M of 3M target)
status: complete
question: With ~40× the update density (train_ratio 1.0, actually enforced) and a 7× bigger brain (base preset) on GPU, does learning progress finally decay where the model converges — letting the gratification balance (boredom gate, hunger airtime) engage as designed?
headline: Capacity was necessary but not sufficient — the model converged (loss 29→4.2), curiosity finally decayed (stimulation 3.7→~0.5, value −50% from peak), and boredom fired for the first time in the series, but it only ever flickered (≤1.6e-3, never accumulated), the running-std normalizer re-inflated curiosity 20× against the decay, and dreamer eating collapsed instead of rising (92 meals in 2M ticks, 38% of them toxic, vs foragers' 5026 at 0.6%); the curiosity→hunger handoff now needs normalization + boredom-accumulator redesign, not more capacity. Bonus: dreamers are the world's only terraformers (318 digs/287 places vs foragers' 0).
runs:
  - save: saves/beta_08
    config: configs/run/beta_08_capacity.yaml
    brain: configs/brain/beta_08_dreamer.yaml
    commit: "322f60b"       # pacing enforcement (55dd661) + sleep-learning debt cap (322f60b)
    ticks: 2005400          # target was ≥3M; host reboot ended it — trends unambiguous by 1.5M
    role: experiment
baselines: [007]
tags: [capacity, motivation, reward, infrastructure]
---

# 008 — the capacity round

## Why this round

Rounds 005–007 converged on the same verdict from three directions: learning capacity,
not reward design, is the binding constraint. Round 007's specific mechanism: relative
LP over a model that never converges (nano/CPU, achieved train_ratio ~0.023) pays out
curiosity forever, so homeostasis stayed ~500× quieter, boredom's gates never opened,
and the gratification balance never engaged. The design's precondition — that interest
can go *stale* — requires a mind fast enough to actually master regions.

While staging this round we found `training.train_ratio` had never been enforced (config
fiction — no code read it; the learner thread was hard-throttled to ~1 update/s/brain by
a `min_round_seconds` floor, so a GPU run under the old scheduler would have been a
placebo). Fixed post-007: the learner paces per-brain update debt to train_ratio
(skip-don't-bank, the sim never waits) with one worker thread per brain, and the round
ships with new instruments — `train_ratio_eff`, `learn_seconds`, `act_steps`, per-region
LP stats (`lp_p50/p90`, `lp_stale_frac`, `lp_occ_entropy`), homeostatic spike stats
(`homeo_max`, `homeo_spike_frac`), boredom gate telemetry (`boredom_calm_gate`/`dull_gate`),
`act_latched_frac`, and `gol-stats --circadian`.

## What changed vs beta_07 (the only knobs)

- `preset: nano → base` (~4M → ~30M params) and `devices: cpu → cuda`.
- `training.train_ratio: 0.25 → 1.0` — and it is now *real*: one update per lived
  act-step, enforced by the pacing learner.
- Scheduler infrastructure (pacing enforcement + per-brain workers) — thread-agnostic
  `learn()` math, so this changes update *density*, which is the variable under test.
- Reward stack, temperament, world config, population, seed: byte-identical to beta_07
  (`beta_08_dreamer.yaml` mirrors `dreamer.yaml`'s reward block exactly). Foragers are
  the cross-run anchor.

## Predictions (written before launch)

- **P1 — LP decays where the model converges (the design-vindication branch).**
  `lp_stale_frac` rises with age, `stimulation` sags from its O(3) plateau, `value`
  stops inflating (beta_07: monotone climb to ~94 on perpetual curiosity). If
  stimulation sags below ~0.5, the boredom stim gate becomes reachable for the first
  time — watch for the first nonzero `boredom` in project history.
- **P2 — hunger gets airtime and behavior moves.** With curiosity decaying, the drive
  reward's meal spikes (visible now via `homeo_max`/`homeo_spike_frac`) stop being
  drowned: eats/day should finally *rise* within-run (beta_07 oscillated 0.06–0.82 with
  no trend; beta_05/06h decayed), lifespans should break the ~14-day hibernation
  treadmill, entropy should keep falling (beta_07: 8.49→8.11, first-ever fall).
- **P3 — the falsification branch.** If the model visibly converges (loss plateaus,
  `lp_stale_frac` high) while `stimulation` stays pinned at the clamp, the LP
  normalization itself (std-only, relative) is mis-designed — rework it before any
  further reward tuning. Either branch is decisive; that's the point of the round.
- Interests (H3/H4 machinery from 007) come along for the ride: does divergence break
  past beta_07's ~0.15–0.19 plateau when agents can actually master niches?

## Operations (the pacing rule)

Achieved ratio ≈ U ÷ (t/s ÷ 5 × 3), U ≈ 3 ÷ `learn_seconds` with per-brain workers.
Run **paced** — an unpaced sprint collapses the ratio back to beta_07 starvation and
inflates the action-latch artifact. Launch (no `--headless`; `--rrd` records instead of
spawning a viewer):

    tmux new -s world
    PYTHONUNBUFFERED=1 uv run gol-run saves/beta_08 --new \
        --config configs/run/beta_08_capacity.yaml --rrd saves/beta_08/rec.rrd

Then read `learn_seconds` off the first metrics and set the world speed so
`train_ratio_eff` holds ≈ 1.0 (e.g. U = 37/s → ~62 t/s → `gol-ctl speed 3`; ~13 h to
3M ticks). Box: RTX 3090 class, ≥16 vCPU, ≥64 GB RAM (decided in 007 review: the RSSM
step is launch-latency-bound, so a bigger card buys nothing; vCPUs and RAM are what the
world/replay actually use). From the laptop: `scripts/sync_back.sh <host> -p <port>
saves/beta_08` on a loop; spot-kill costs at most one sync interval.

First-checkpoint sanity (~30 min): `train_ratio_eff` ≈ 1.0 and holding,
`act_latched_frac` ≈ 0 (paced), `lp_regions` 32, three temperament draws, forager eat
rate within variance of beta_07's 9.7/10k, `homeo_max` spiking ~0.6 on meals.

Launch-day operational finding (first attempt, tick 0–13k, discarded): newborn
dreamers hibernate early and often, and with the debt cap at 32 the learner froze
during every dormant spell — `updates/s` hit 0 while the world ran on. Fixed before
the real launch (322f60b): the cap now holds a full awake burst (1024), so the
learner works through banked experience during hibernation — sleep pays the day's
debt, updates stay strictly ratio × experience lived. Measured on the pod: base
preset `learn_seconds` ≈ 0.25–0.74 s/update with 3 concurrent workers (~4–6
updates/s total), so ratio 1.0 is sustainable against the ~12 awake-act-steps/s of
speed 1.0 only via the dormancy catch-up; if awake fraction runs high, achieved
ratio will sag below 1.0 and that is the honest number to report.

## Monitoring the live run (for whoever checks progress next)

The run lives on RunPod pod `pjilmbiyse472t` (community 3090, $0.22/hr, 28 vCPU/62GB),
launched 2026-07-06 in tmux session `world`, repo at `~/GameOfLife`, log at
`~/GameOfLife/beta_08.log`. **No auto-terminate is set** — terminate the pod in the
console (or `runpodctl pod delete pjilmbiyse472t`) when the round closes.

SSH (direct TCP; the `ssh.runpod.io` proxy is PTY-only — no command exec, no rsync):

    ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 15023 root@64.119.209.250

- **Watch stdout live:** add `-t ... 'tmux attach -t world'` (detach: `Ctrl-b d`).
- **Population/learning pulse, on the pod:** `cd ~/GameOfLife && uv run gol-stats
  saves/beta_08 [--compare|--interests|--circadian]`.
- **Capacity gauges** (the numbers this round is about): grep the tail of
  `saves/beta_08/metrics.ndjson` for `train_ratio_eff` (climbs toward ≈1.0 over life),
  `learn_seconds` (~0.25–0.65 s/update, 3 workers), `updates` vs `act_steps`.
  **How to read them:** `updates = act_steps − 500` exactly (warmup acts carry no
  debt), so ratio_eff asymptotes to 1.0 rather than sitting there from birth. GPU at
  0%/P8 with all dreamers dormant means debt fully paid — correct, not stalled;
  verify with the updates↔acts identity. `act_latched_frac` runs ~0.4, NOT ≈0: at
  ratio 1.0 the learner is always mid-update during awake bursts and act() latches —
  the known artifact is the price of full density (a lock-free weight snapshot for
  act() is the future fix if it matters).
- **Speed policy (35-min sanity, tick 20k):** enforcement verified exact (updates =
  acts − 500 on all three dreamers). Because dreamers are dormant most of the time,
  the GPU keeps up far below capacity — raised to `gol-ctl speed 3` (~60 t/s, 3M in
  ~14 h ≈ $3 instead of 42 h ≈ $9). Debt banks during awake bursts (cap 1024) and
  drains during sleep, so time-averaged ratio holds ≈1.0 while awake fraction stays
  low. **If dreamers get much more active later (the P2 hope), ratio_eff will sag —
  check it, and `gol-ctl speed 1` to restore full density at the cost of wall-clock.**
- **Control:** `uv run gol-ctl pause|resume|speed <x>|checkpoint` on the pod
  (port 7301). A clean stop is `gol-ctl checkpoint`, then ONE Ctrl-C in tmux.
- **Mirror home** (run on the laptop; spot-kill insurance + local analysis):
  `RSYNC_RSH="ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 15023"
  scripts/sync_back.sh root@64.119.209.250 -p 15023 saves/beta_08`
- **Rerun: OFF since tick 255,966.** The `.rrd` recording grew ~80 MB/min at
  `rerun_fps: 10` (6.9 GB by tick 250k — on track to fill the 40 GB disk near
  ~1.2M and crash the run) and made the mirror re-pull one giant file every cycle.
  Clean stop → deleted the pod's rrd → resumed with `--set
  observability.rerun=false` (paced, no logger). A 5.2 GB local copy of the first
  ~85 min survives on the laptop (`uv run rerun saves/beta_08/rec_000000000000.rrd`)
  for eye candy. Future cloud rounds: `rerun_fps: 1` or rerun off; never record
  full-fps rrd on a multi-hour run.
- Resume bookkeeping: learner workers take no retroactive debt on first sight, so
  whatever updates were *owed* at the restart (banked debt from the last awake
  burst) were forgiven — `updates` will run slightly below `acts − 500` from here;
  the shortfall is bounded by one debt cap (≤1024 per brain).
- Budget: at `speed 3` (~60 t/s) → 3M ticks ≈ 14 h ≈ $3.10 total.

## Mid-run review (~1.06M ticks, 2026-07-06)

Run healthy: tick 1,059,700 of 3M, population 6 (3 dreamers / 3 foragers), GPU 27%,
disk 26%, `train_ratio_eff` 0.92 and still climbing toward 1.0, `learn_seconds` 0.25.
Holding `speed 3` — ratio is not sagging, so no reason to slow down (~9 h to 3M).

- **P1 is landing.** The model converges for the first time (`loss_model` 29 → 4.3),
  `lp_stale_frac` climbs 0 → ~0.25–0.31, `stimulation` sags from its 3.4–3.7 plateau
  to ~0.7–1.1 (dipping toward the 0.5 gate), and **the first nonzero `boredom` in
  project history** appeared at ~tick 338k — every dreamer born since shows it (max
  1.45e-3, dreamer_017). Tiny, but the gate is provably reachable now. `value` peaked
  ~678 at ~580k and has *declined* to ~541 — the perpetual-curiosity annuity of
  beta_07 (monotone climb) is over.
- **P3 watch, not triggered but real:** raw stimulation falls while `curiosity_scaled`
  *rises* (0.09 → ~1.38) — the std-only relative normalization re-inflates curiosity's
  scale as LP shrinks. Stimulation is sagging anyway, so the falsification branch
  isn't hit, but the normalization is fighting the decay; keep this for the close.
- **P2 not moving yet.** Dreamer eats per 100k ticks: 8, 8, 12, 10, 4, 1, 9, 3, 1, 5 —
  flat-to-down, no hunger airtime in behavior; `homeo_spike_frac` ~0.007. Dreamer
  lifespans 172k–378k ticks with no trend. Policy entropy keeps falling (7.5 → ~3.3),
  continuing beta_07's first-ever decline.
- Interest divergence still plateaued at ~0.10–0.18 — not breaking past beta_07.
- **Bug found (once, benign):** thread `learner-dreamer_007` died with
  `KeyError: 'dreamer_007'` at `scheduler.py:341` — a race where `_supervise` prunes
  `_owed[rid]` while that brain's last `learn()` is in flight; the decrement after
  learn() hits the missing key. No effect (agent was dead; supervisor reaps the
  thread) but fix before the next round.
- `act_latched_frac` runs ~0.78 steady-state, not the ~0.4 recorded at the 35-min
  sanity check — the latch artifact at ratio 1.0 is bigger than the runbook says.
  Behavior comparisons vs beta_07 must weigh this.
- Mirror sync had not been running on the laptop; started (15-min loop). Resume
  forgiveness confirmed harmless: updates ≈ acts − 500 within one debt cap.

### Second checkpoint (~1.49M, halfway)

Run healthy (ratio_eff 0.94, disk 26%, no new learner crashes; GPU idles at 0% during
dormant spells as designed). All 1.06M trends continue, three sharpenings:

- **Stimulation now sits ON the boredom gate** (0.51–0.66 vs the ~0.5 threshold);
  `lp_stale_frac` reached 0.39–0.41; `value` down 40% from its 678 peak to ~413. The
  decay branch of P1 is fully real. **But boredom does not *accumulate*** — it
  flickers at 1e-4–1e-3 on every dreamer generation (16 dreamers now) and never
  builds. Gate reachable ≠ gate dwelled-in.
- **P2 is inverting, not just flat.** Dreamer eats per 100k ticks collapsed:
  8, 8, 12, 10, 4, 1, 9, 3, 1, 5, 0, 1, 1, 4, 3 — the 1.0–1.1M window had *zero*
  dreamer eats. And dreamer lifespans are eerily clock-like: the last six dreamer
  deaths span 340,855–352,413 ticks (±2%), while foragers range 403k–556k. Lifespan
  looks wear/hibernation-determined with behavior contributing nothing — the
  hibernation treadmill unbroken, just with a longer period than beta_07's.
- **The P3 suspect keeps strengthening:** `curiosity_scaled` climbed further
  (1.36 → 1.70) as raw stimulation fell — the running-std normalizer is actively
  re-inflating curiosity's scale, ~20% more since 1.06M. Policy entropy still
  falling (now ~2.3–3.2).

Watch for the close: if 1.5M more ticks of on-the-gate stimulation never converts to
dwelled-in boredom or a single upturn in eating, the round's answer is "capacity was
necessary but not sufficient — the std-only relative normalization (and possibly the
boredom accumulator's time constants) is the binding design flaw," which is P3 in
slow motion rather than the pinned-at-clamp version pre-registered.

## Results

Run ended at **tick 2,005,400** (of the 3M target) by a RunPod host reboot — not a clean
stop; the last atomic checkpoint is the resume point. ~9.5 h of paced wall-clock, ~$2.60.
36 robots lived (22 dreamers across 3 lineages, 14 foragers), 30 deaths, population 6
throughout. Pacing enforcement held for the whole run: final `train_ratio_eff` 0.956,
`learn_seconds` 0.25–0.74, `act_latched_frac` ~0.78 steady-state.

**P1 — confirmed. Curiosity decays where the model converges.**

| gauge | birth→early | end of run |
|---|---|---|
| `loss_model` | 29 | ~4.2 |
| `stimulation` | 3.4–3.7 plateau | 0.5–1.0 band (on the 0.5 gate from ~1.35M) |
| `lp_stale_frac` | 0.0 | 0.3–0.5 band |
| `value` | climb to 683 peak @ ~640k | 343 (−50% from peak) |

First nonzero `boredom` in project history at ~tick 338k. Every dreamer generation
since flickers (1e-5 to 1.6e-3, max dreamer_020) — **but it never accumulates**, even
with stimulation sitting on the gate for the last 650k ticks. Gate reachable ≠ gate
dwelled-in.

**P2 — falsified. Hunger never got airtime; eating collapsed instead of rising.**
Dreamer eats per 100k ticks: 8, 8, 12, 10, 4, 1, 9, 3, 1, 5, 0, 1, 1, 4, 3, … — total
92 dreamer meals vs 5,026 forager meals. Death-ledger decomposition shows exactly two
dreamer death modes: (a) the **hibernation clock** — 9 of 19 dreamer deaths within ±2%
of ~347k ticks, ledgers reading hibernation 91–96, wear 5–10, poison 0; (b)
**poison-accelerated** — lifespans 109k–263k with poison ledger 36–84. Behavior
contributes nothing to dreamer survival. Foragers show the coupling dreamers lack:
eats 90–589 funds repair 10–89, lifespans 403k–556k, wear-dominated. The treadmill is
unbroken, just longer-period than beta_07's.

**P3 — the slow-motion branch, and the round's real answer.** `curiosity_scaled` rose
monotonically 0.09 → 1.86 (~20×) while raw LP decayed — the lifetime running-std
normalizer measurably re-inflates curiosity's scale as the signal shrinks: the
built-in hedonic treadmill, caught on instruments. Stimulation sagged anyway (the
clamp and region structure still let absolute decay through), so the pre-registered
"pinned at clamp" version didn't trigger — but the two mechanisms that *did* bind are
concrete: (1) std-only relative normalization fights its own decay; (2) the boredom
accumulator's time constants don't integrate gate-touching into pressure.

**Outside the pre-registered lanes:**

- **Dreamers are the world's only terraformers**: 318 digs + 287 places, foragers 0.
  Intrinsic motivation produces all world-modification in the system.
- **Dreamers never learned bush discrimination**: 38% of dreamer meals poisoned
  (35/92) vs 0.6% for foragers (31/5026) — despite obs v3 color vision existing for
  exactly this. Plausibly also a within-run eating deterrent (poison ledger damage).
- **Lineage individuality replicated at capacity** (beta_07's H4): three temperament
  draws persisted through 22 dreamers; lineage C (w_cur 0.76, w_homeo 1.11) ran
  systematically lower `value` (257–378 vs 300–906 for A/B) and supplied most of the
  forage-flavored profiles; A/B skewed pure-rest. Interest divergence stayed
  plateaued ~0.15 (dips to 0.03 during synchronized hibernation).
- **Circadian structure exists but is weak in dreamers**: dormancy 0.77 day / 0.95
  night (corr −0.28) vs foragers' hard nocturnal economy (eat-at-night corr −0.87).
- Policy entropy fell 7.5 → ~2–3 by mid-run (continuing beta_07's first-ever fall);
  late aggregate wobbles up as young births dominate the mean.

## Interpretation

Rounds 004–007 said "capacity is the binding constraint." This round bought the
capacity and got the receipt: the model converges, interest goes stale, boredom
exists. Everything 007 said was impossible now happens — **and the cascade still
stalls at step one**. The design assumed decaying curiosity would hand the floor to
hunger; instead the floor stayed empty: homeostasis is still O(0.005) against a
curiosity signal whose *normalizer* keeps re-amplifying it (0.09→1.86), and the
boredom accumulator discards what the gate admits. The binding constraint has moved,
for the first time, from capacity to **reward-stack design** — specifically signal
conditioning, not drive semantics. That is progress: 007 couldn't distinguish "the
stack is wrong" from "the mind is too small to engage it." 008 can: the mind engaged
it, and two specific components failed in specific, instrumented ways.

The terraforming and poisoning findings sharpen the same point from outside: the
dreamers are behaviorally *alive* (they alone modify the world; lineages have
persistent styles) but survival-*blind* (eating collapsed, no bush discrimination,
death by hibernation clock). Curiosity built explorers; nothing yet builds survivors.

## Caveats

## Caveats

- **Two knobs moved together** (model size and update density) — deliberate, since the
  round tests "capacity" as a bundle; if the result needs attribution between them, the
  follow-up ablation is base@0.023 or nano@1.0.
- beta_07 ran under the throttled scheduler; that *is* the baseline as-lived, but its
  behavior data carries the ~23%+ action-latch artifact of unpaced running. Compare
  trajectory shapes and within-run trends, not fine levels.
- Single run; round 006 measured 40% forager variance between identical-config runs.
- Ended at 2.005M of the 3M target by host reboot (unclean stop; last atomic
  checkpoint coherent). Trends were unambiguous and stable from ~1.5M, so the round
  closes on 2M.
- `act_latched_frac` ~0.78 all run (not the ~0.4 from the 35-min sanity) — behavior
  levels carry a large latch artifact; within-run *trends* are the comparable thing.
- The learner-worker KeyError race killed one worker thread once (benign — the agent
  was already dead). Fixed post-launch in 398573a; the pod ran pre-fix code all round.

## Next

- **Round 009 (the conditioning round, ~12 h GPU):** keep capacity fixed at base@1.0,
  change only signal conditioning, behind config flags per convention —
  1. **Normalizer rework** (primary): replace lifetime running-std LP normalization
     with an absolute or early-life-anchored scale so global convergence reads as
     global satisfaction. beta_08's curve: `curiosity_scaled` 0.09→1.86 over one run
     is the number to kill.
  2. **Boredom accumulator time constants** (secondary): stimulation sat on the gate
     for 650k ticks and produced ≤1.6e-3 boredom — integration is broken, tune the
     accumulator so sustained gate-touching becomes pressure.
  3. Watch the free riders: does eating recover once curiosity can actually yield?
     Does bush discrimination (38% poisoned meals) emerge when homeostasis gets
     airtime? Do the dreamer-only terraforming and lineage styles survive?
- Prune beta_* save dirs per round 006's note once the beta_08 mirror is verified.
- Terminate pod pjilmbiyse472t (runbook: no auto-terminate is set).
