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
  `saves/beta_08/metrics.ndjson` for `train_ratio_eff` (should climb to ≈1.0 via
  dormancy catch-up), `learn_seconds` (~0.25–0.65 s/update, 3 workers),
  `act_latched_frac` (≈0 when paced), `updates` vs `act_steps`.
- **Control:** `uv run gol-ctl pause|resume|speed <x>|checkpoint` on the pod
  (port 7301). A clean stop is `gol-ctl checkpoint`, then ONE Ctrl-C in tmux.
- **Mirror home** (run on the laptop; spot-kill insurance + local analysis):
  `RSYNC_RSH="ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 15023"
  scripts/sync_back.sh root@64.119.209.250 -p 15023 saves/beta_08`
- **Rerun viewing:** the run records rotating `.rrd` files into the save dir
  (`rec_*.rrd`, new file every 6 sim-hours ≈ 432k ticks). They sync home with the
  mirror; open locally with `uv run rerun saves/beta_08/rec_000000000000.rrd`. The
  in-progress file usually opens fine (you see everything logged so far); completed
  rotations are the safe bet. Live streaming wasn't wired for this run — recordings
  only.
- Budget: ~20 t/s → 3M ticks ≈ 42 h ≈ $9.20 total; user balance was $10 at launch.

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
