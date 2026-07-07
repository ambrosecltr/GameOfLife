---
round: 010
title: swift nano — real update density on a laptop
date: 2026-07-07
status: planned
question: With the swift speed core making train_ratio 1.0 real locally, does a nano dreamer reproduce beta_09's conditioning behaviors — or does conditioning secretly depend on base-preset capacity?
headline: (write at close)
runs:
  - save: saves/swift_01
    config: configs/run/swift_01_local.yaml
    brain: configs/brain/swift_01_dreamer.yaml
    commit: (pin at launch)
    ticks: 0                # target ≥1M for the anchor/boredom dynamics to unfold
    role: experiment
baselines: [008, 009]
tags: [capacity, motivation, conditioning, performance, track:swift]
---

# 010 — swift nano: real update density on a laptop

## Why this round

Two pressures met:

1. **Local tests were weak evidence.** Round 007 ran nano at achieved ratio ~0.023;
   round 008 proved update density is the binding variable. Every local nano run since
   obs v3 has therefore tested starvation, not the brain. Cloud rounds are the only
   honest instrument, and they cost 12+ hours per answer.
2. **The nano update was dispatch-bound, not compute-bound.** Profiling showed a ~4M-param
   learn() spending ~40% of its 440ms in a single slow `Tensor.var(dim=0)` call
   (the Plan2Explore disagreement), ~25% in Python/dispatch overhead around tiny
   sequential ops, and only ~35% in real matmuls + backward.

The swift track exists to make the local tier a legitimate testbed: same world-model
family as beta, the bet is *efficiency* — learn faster with no quality loss, so more
rounds per week and (if nano hosts the conditioning behaviors) screening on the laptop
before spending cloud hours. New prefix per configs/README.md: swift shares beta's
architecture family but not its capacity bundle, so the counters must not collide.

## What changed

The speed core (commit on branch `swift/nano-optimization`), all parity-pinned by tests:

- **Manual ensemble variance** — `disagreement()` computes E[x²]−E[x]² instead of
  `Tensor.var(dim=0)` (the 59ms/call CPU slow path). Biggest single win: 440→242ms.
- **Batched ensemble (`EnsembleMLP`)** — the 8 Plan2Explore members as stacked einsums
  instead of 8 module calls × 3 sites/update. Pre-swift ModuleList checkpoints migrate
  on load (beta_08/09 blobs stay analyzable; their Adam moments reset on migration).
- **Closed-form distributions** — `TanhNormal`, `DiscreteDist`, inverse-CDF categorical
  replace `torch.distributions` objects on every RSSM step, imagination step, and act().
- **Adult imagination skip** — once the LP trickle anneals to 0, the imagination-path
  ensemble pass is a multiply-by-zero and is skipped.
- **`replay.burn_in`** — gradient-free prefix warms (h, z) so `seq_len` halves (32+8
  vs 64) at equal samples/update (batch 32); backward was ~40% of the post-var-fix
  update. Kills the zero-init-state lie at the same time.
- **`replay.recent`** — one batch row pinned to the newest experience window
  (DreamerV3 online-queue mixing), ring-seam correct.
- **`foreach` Adam**, and flags staged for later ablation: `training.compile`
  (measured neutral once var died), `training.optimizer: muon`, `training.l2_init`.

Measured on M1 Pro (scripts/bench_learn.py, 20 timed updates, 1024 samples/update
throughout):

| config | learn p50 (solo) |
|---|---|
| pre-swift nano, cpu, 16×64 (beta_07-era shape) | 474ms (historical) / 441ms (re-measured) |
| swift core only, cpu, 16×64 | 242ms |
| swift core + 32×32+burn8, cpu | 209ms |
| swift core + 32×32+burn8, mps | 188ms newborn / **174ms adult** |

Contention (3 learner workers, the real population): **cpu 7.6 updates/s
aggregate vs mps 4.9** — one GPU queue serializes siblings, eight cores share —
so the run config uses cpu despite mps winning solo. ≈ **2.7× per update**
solo and ~3× aggregate against the pre-swift 3-worker CPU baseline, so at
act_every 5 / tick_rate 20, three nano dreamers hold ratio 1.0 locally at
awake fraction ≤ ~0.6, with dormancy + the debt cap as slack. The "20M ticks
where 5M fit" ambition cashes out as: update speed bounds world speed at fixed
ratio, so ~3× per update ≈ ~3× ticks per wall-hour at ratio 1.0, before the
adult skip and any future parallel-unroll architecture.

Conditioning knobs are beta_09's verbatim (anchored normalizers at 1e6 samples,
trickle anneal 1500 act-steps, boredom pressure 0.002/0.0002, HRRL drive,
temperament). Deviation: replay capacity 300k vs beta's 500k (local RAM/disk;
checkpoints serialize the buffer).

## Results

(at close; canaries below)

- `train_ratio_eff` ≥ 0.9 by first checkpoint — else the round tests starvation again
  and the world must slow down. **Kill if it can't hold 0.8 at any speed.**
- `curiosity_scaled` must not re-inflate after the anchor freezes (~1000 updates).
  If it re-inflates at nano too → conditioning bug, not capacity; if it holds at
  base (beta_09) but not nano → capacity×conditioning interaction, round 011 material.
- `lp_mix_eff` → 0 by ~1500 act-steps/dreamer; adult `learn_seconds` should drop
  visibly when it does (the imagination skip is observable in the metrics).
- `boredom_pressure` charges under sustained calm+dull; **kill by ~600k ticks** if
  all three conditioning canaries are dead flat while the model converges —
  that's beta_08's signature and means nano can't host the dynamics.
- Behavioral gauges vs beta_09: eats/100k, death-ledger mix, falling entropy,
  lineage-consistent individuality (007's finding).

## Interpretation

(at close)

## Caveats

- Not a strict A/B against beta_09: the swift replay recipe (burn-in, recent slot,
  32×32 shape) rides along with the capacity change. If nano reproduces beta_09's
  conditioning, that validates the *bundle*; attributing any difference between
  swift_01 and beta_09 to capacity alone requires a beta-side rerun on the swift
  core (cheap: the core is math-equivalent, flags off).
- MPS numerics differ from CUDA/CPU (fp32 accumulation order); determinism holds
  per-device only.
- The recent-slot changes replay statistics; LP's "retention counts" semantics now
  see one guaranteed-fresh window per batch. Watched via lp_occ_entropy.

## Next

- If nano hosts the conditioning behaviors: swift becomes the screening tier
  (hypotheses run local-first), and the beta bundle inherits the speed core —
  base-preset updates get the same var fix / batched ensemble / burn-in, which
  buys either 2-3× world speed on the same pod or the same round at ~1/3 cost.
- swift_02 ablation candidates, one bundle at a time: `optimizer: muon`,
  `l2_init` (plasticity maintenance — 008's "capacity necessary but not
  sufficient" may partly be trainability loss), `compile` on mps.
- Parallel-sequence world model (transformer/SSM over the unroll) as its own
  track prefix — the GRU's sequential unroll is the remaining structural ceiling
  once dispatch overhead is gone.
- Offline conditioning gym (scripts/conditioning_gym.py): replay a recorded life's
  buffer through candidate conditioning stacks without running the world — the
  009-style knob rounds should be screened there in minutes first.
