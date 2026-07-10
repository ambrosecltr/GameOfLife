---
round: aion-001
title: continuous-time S5 foundation
date: 2026-07-10
status: implementation-complete, two-gpu-arm-staged
question: can a resettable S5 world model consolidate 1,024-step experience efficiently, retain multi-timescale organism context, and preserve slow identity through measured blackouts without decoupling the mind's lived time from the world?
headline: "STAGED, NOT RUN: the selected two-lineage arm assigns one full-budget Aion to each RTX 3090 and clears 35.68-38.25 safe ticks/s with zero action deadline misses; no behavioral claim exists before the persistent world begins."
runs:
  - save: saves/aion_01
    config: configs/run/aion_01.yaml
    brain: configs/brain/aion_01_s5.yaml
    commit: not launched
    ticks: 0
    role: experiment (base/CUDA developmental arm)
  - save: saves/aion_01_2gpu
    config: configs/run/aion_01_2gpu.yaml
    brain: configs/brain/aion_01_s5.yaml
    commit: not launched
    ticks: 0
    role: experiment (selected two-lineage, one-learner-per-GPU arm)
baselines: [beta-013]
tags: [aion, s5, state-space-model, long-context, dormancy, lifelong-learning]
---

# Aion 001 — continuous-time S5 foundation

## Why this round

Beta 013 is the strongest surviving artificial-organism stack, but its GRU
transition trains sequentially over 64-step replay windows. That is a poor fit
for both the available GPU and the research target: organisms whose relevant
causes may span blackouts, seasons, relationships, and substantial fractions of
a life. Proposal 005 establishes Aion as the S5 hedge rather than rewriting beta
before evidence exists.

## What changed

- New registered `aion` checkpoint family and `AionBrain`; beta remains
  checkpoint-compatible with older Dreamer saves.
- Four S5 blocks replace the deterministic GRU transition. The stochastic
  latent, predictive heads, endogenous affects, vector critic, and temporal
  skills are retained.
- Replay context increases from beta 013's 64 steps to 1,024 graded steps plus
  256 burn-in steps. Batch 8 and train ratio 0.25 yield 2,048 graded replay
  timepoints per lived act.
- Parallel replay uses a resettable affine scan; live action and imagination
  remain recurrent.
- Replay now distinguishes full stream starts from wakes and stores elapsed
  blackout duration.
- The runtime counts missed perception cycles through dormancy and checkpoints
  that timing.
- Wake resets observation state, action, active skill, and fast S5 modes while
  preserving and time-decaying the slowest half of the modes. New-body reset
  still clears all live state.
- Benchmark tools now instantiate the registered brain kind rather than
  silently forcing Dreamer, auto-fill enough replay for long windows, and
  report replay timepoints per second.
- Native complex S5 storage was replaced by FP32 paired-real parameters,
  transition/state, and scan. The large protected real projection GEMMs remain
  TF32-eligible; BF16 projection is rejected pending parity evidence.
- Precision is now an explicit `ieee_fp32 | tf32 | amp_bf16` policy with CUDA
  capability checks, process-wide TF32 ownership, FP32 probabilities/losses,
  exact async-snapshot checkpointing, and checkpoint metadata.
- Learning credit is derived from checkpointed lived acts and completed updates.
  A measured governor applies bounded causal backpressure and logs safe/actual
  tick rate, capacity, debt, deadlines, and limiter. Dropping is explicit only.
- Universal dormancy first pays whole update debt, then removes wall pacing.
  Settled event-free intervals bulk-integrate to the tick before the next real
  boundary while preserving RNG, virtual timestamps, missed opportunities, and
  scalar wake/death/event ordering.
- Multi-GPU device lists assign independent brains round-robin with no all-reduce.
  Cross-brain batching, streams, pinned transfer, and scan compilation remain
  profiler-gated because local hardware cannot justify them.

## Mechanical evidence before launch

- Parallel and recurrent paired-real S5 states match across reset/wake patterns;
  paired-real outputs, states, and parameter gradients match a complex64
  reference within FP32 operation-order tolerance.
- Automated tests backpropagate finite gradients through 256 burn-in plus 1,024
  graded steps, compare scans across multiple sequence lengths and reset/wake
  patterns, probe retention at 100/250/500/1,024/100,000 steps, and preserve a
  delayed cue through 1,024 repeated recurrent steps.
- A dtype regression proves the protected recurrence remains FP32 inside BF16
  autocast and catches the `0.9999500012 -> 1.0` BF16 decay failure.
- A tiny Aion completed `act()`, `learn()`, and checkpoint roundtrip with finite
  loss.
- Legacy native-complex checkpoints migrate with model/update parity and no
  ambiguous paired-real cross-load; exact published inference snapshots roundtrip.
- Seeded dormant fast-forward differentials match ordinary stepping for bodies,
  ecology heaps, RNG, events, recharge, integrity, fatigue, held age, and world time,
  while stopping scalar-exactly before wake, death, spoilage, metrics, lifecycle,
  and checkpoint boundaries.
- Deaths are reconciled on every world tick; dormant last observations and deferred
  terminal records survive checkpoint/resume instead of leaving an orphaned lineage.
- Aion 01 holds the same 2,048-feature latent width as beta 013. The world model
  contains approximately 41.1 million real scalar parameters versus beta's
  43.5 million, so the new branch is not winning by being larger.
- Single-GPU 5090 and H100 measurements established that three siblings
  contending on one device could not sustain 20 ticks/s with headroom. The
  selected pod instead isolates two full-budget Aions on two RTX 3090s.

## Results

The long-lived ecological round has not started. No long-memory, survival, or
emergence claim is made.

The selected Community Cloud pod has two RTX 3090 24 GB cards and costs
$0.44/hour in total. With Torch 2.12.1+cu130, CUDA 13.0, and protected-FP32 S5
inside `amp_bf16`, two synchronized 20-update two-card gates measured:

- 4.197-4.500 aggregate updates/s across two independent learners;
- 35.68-38.25 sustainable ticks/s after the configured 0.85 headroom;
- 0 action deadline misses, 111.3-115.9 ms action p95, and 128.3 ms worst maximum;
- at most 6,336 MiB peak reserved VRAM on each 24,124 MiB device;
- 34,385-36,861 graded replay timepoints/s and $0.00332-$0.00355 per million
  timepoints.

A fresh 200-tick run and 20-tick resume using the exact round config completed
with `aion_000` and `aion_001` checkpointed beside four scripted foragers. The
manifest retained `learning: [cuda:0, cuda:1]`, and resume advanced coherently
from tick 200 to tick 220.

## Caveats

- The observation-only posterior enables parallel consolidation but may lose
  history-conditioned inference; the recurrent S5 state remains available to
  the prior and all downstream decisions.
- The portable PyTorch scan performs `O(L log L)` work to obtain logarithmic
  sequential depth. Target-GPU measurement decides whether this engineering
  trade is worthwhile.
- Aion 01 requires CUDA BF16. Unsupported precision/device requests fail; Apple
  MPS is not an Aion 01 target and does not silently fall back to FP32.
- Long predictive state is not exact episodic memory.

## Next

1. Commit the staged implementation and preserve the two-card benchmark artifacts.
2. Launch `saves/aion_01_2gpu` through `scripts/start_aion_01_2gpu.sh`.
3. Confirm the first learned updates keep safe capacity above 25 ticks/s, debt
   bounded, dropped credit at zero, and action deadline misses at zero.
4. Record the launch commit and first stable checkpoint in this journal.
