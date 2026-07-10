---
round: aion-001
title: continuous-time S5 foundation
date: 2026-07-10
status: staged
question: can a resettable S5 world model consolidate 1,024-step experience efficiently, retain multi-timescale organism context, and preserve slow identity through measured blackouts without decoupling the mind's lived time from the world?
headline: "STAGED, NOT RUN: Aion is now a separate checkpointed brain lineage with stable four-block S5 dynamics, recurrent live thought, parallel 1,024-step replay consolidation, and a lifecycle that resets fast sensation while slow context decays across measured dormancy. Mechanical and target-3090 performance gates precede any memory or behavior claim."
runs:
  - save: saves/aion_01
    config: configs/run/aion_01.yaml
    brain: configs/brain/aion_01_s5.yaml
    commit: not launched
    ticks: 0
    role: experiment (base/CUDA developmental arm)
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

## Mechanical evidence before launch

- Parallel and recurrent S5 states matched to approximately `7e-7` in the
  direct smoke probe.
- Automated tests backpropagate finite gradients through 1,024 steps.
- A tiny Aion completed `act()`, `learn()`, and checkpoint roundtrip with finite
  loss.
- Aion 01 holds the same 2,048-feature latent width as beta 013. The world model
  contains approximately 41.1 million real scalar parameters versus beta's
  43.5 million, so the new branch is not winning by being larger.
- Full target-CUDA and three-sibling contention measurements remain pending.

## Results

Not run. No long-memory, learning-speed, survival, or emergence claim is made.

## Caveats

- The observation-only posterior enables parallel consolidation but may lose
  history-conditioned inference; the recurrent S5 state remains available to
  the prior and all downstream decisions.
- The portable PyTorch scan performs `O(L log L)` work to obtain logarithmic
  sequential depth. Target-GPU measurement decides whether this engineering
  trade is worthwhile.
- Complex S5 kernels target CUDA and CPU. Apple MPS support is not an Aion 01
  requirement.
- Long predictive state is not exact episodic memory.

## Next

1. Run the single-brain and three-worker RTX 3090/3090 Ti benchmarks.
2. Record peak VRAM, update latency, recurrent act latency, and sustainable
   world tick rate.
3. Run mechanical delayed-cue and long-rollout probes before the ecological
   round.
4. Launch `saves/aion_01` only if the gates in proposal 005 pass.
