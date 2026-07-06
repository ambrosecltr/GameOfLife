---
round: 008
title: the capacity round — does a mind that can converge develop taste?
date: 2026-07-06            # pre-registered at launch prep; results pending (target ≥3M ticks)
status: planned
question: With ~40× the update density (train_ratio 1.0, actually enforced) and a 7× bigger brain (base preset) on GPU, does learning progress finally decay where the model converges — letting the gratification balance (boredom gate, hunger airtime) engage as designed?
headline: pending
runs:
  - save: saves/beta_08
    config: configs/run/beta_08_capacity.yaml
    brain: configs/brain/beta_08_dreamer.yaml
    commit: "322f60b"       # pacing enforcement (55dd661) + sleep-learning debt cap (322f60b)
    ticks: 0                # target ≥3M
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

## Results

*(pending)*

## Interpretation

*(pending)*

## Caveats

- **Two knobs moved together** (model size and update density) — deliberate, since the
  round tests "capacity" as a bundle; if the result needs attribution between them, the
  follow-up ablation is base@0.023 or nano@1.0.
- beta_07 ran under the throttled scheduler; that *is* the baseline as-lived, but its
  behavior data carries the ~23%+ action-latch artifact of unpaced running. Compare
  trajectory shapes and within-run trends, not fine levels.
- Single run; round 006 measured 40% forager variance between identical-config runs.

## Next

*(pending close: fill results, headline, README index row; prune beta_* save dirs per
round 006's note)*
