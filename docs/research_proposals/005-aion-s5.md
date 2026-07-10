---
proposal: 005
title: Aion — continuous-time S5 world models for lifelong digital organisms
date: 2026-07-10
track: aion
status: implemented, staged
targets_round: aion_01
question: Can a resettable S5 world model give a continuously living digital organism materially longer predictive memory and faster replay consolidation than beta's GRU RSSM, while keeping one recurrent thought-step aligned with every lived world-step and preserving identity through sensory blackouts without pretending the blackout was observed?
depends_on: [beta-013, anima-007]
tags: [s5, state-space-model, lifelong-learning, memory, world-model, dormancy, gpu, artificial-organism]
---

# 005 — Aion

## The founding bet

Beta and anima answer different questions. Beta has learned expectation,
imagination, and a critic but compresses its temporal context through a GRU.
Anima removes the world model and critic entirely; seven rounds showed that an
unlearned valence baseline cannot supply a reliable teaching signal over a
declining mortal life. Aion keeps the learned organism machinery that survived
those tests and changes the temporal substrate.

The bet is not “S5 solves intelligence.” It is narrower and falsifiable:

1. long replay sequences should train with substantially more GPU parallelism
   than a Python-level GRU unroll;
2. a structured continuous-time state should retain predictive information at
   multiple learned timescales;
3. the same dynamics must still run recurrently one perception at a time, so
   faster consolidation never advances subjective time ahead of the world;
4. dormancy should interrupt sensation, not erase identity or freeze internal
   time while reality advances.

The architecture follows [S5](https://openreview.net/forum?id=Ai8Hw3AXqks),
with the world-model factorization motivated by
[S4WM](https://papers.nips.cc/paper_files/paper/2023/hash/e6c65eb9b56719c1aa45ff73874de317-Abstract-Conference.html)
and [Hieros](https://arxiv.org/abs/2310.05167). Those results justify a serious
branch, not an assumption of superiority.

## Architecture boundary

Aion is registered as `kind: aion` and owns its checkpoint family. It is not a
mode flag inside `kind: dreamer`, and beta checkpoints cannot be loaded into it.
The two lineages share only mechanisms that mean the same thing:

- observation encoder and predictive heads;
- categorical stochastic state and balanced KL objective;
- endogenous comfort, viability, curiosity, boredom, mortality, and
  self-generated skill controllability;
- imagination-trained actor and vector critic;
- replay, temperament, inheritance, and unnamed temporal skills.

The deterministic transition is four nonlinear residual S5 blocks. Each block
contains 128 diagonal modes represented as FP32 real/imaginary pairs, a 768-wide MIMO input/output map, stable
left-half-plane eigenvalues, HiPPO-LegS frequency initialization, and learned
continuous-time step sizes. Real and imaginary state pairs across four blocks
produce 1,024 deterministic features, so Aion 01 retains beta 013's total
2,048-feature latent capacity after adding the 1,024 categorical stochastic
features.

During live action, each block performs one recurrent transition. During replay,
the diagonal affine recurrence is composed with an associative scan. A
1,024-step window therefore has ten sequential tensor stages per block rather
than 1,024 Python-level recurrent calls. The scan uses only public PyTorch
operations; no JAX runtime or third-party S5 package enters the deployed stack.

### Numerical precision boundary

Aion 01 uses `training.precision: amp_bf16`, but that does not mean every tensor
is BF16. Eligible encoder, prediction-head, normalization/residual/gate,
ensemble, actor, and critic forward computation is autocast. Parameters,
optimizer state, losses/probability normalization, and reductions stay FP32.

The long-timescale recurrence is always FP32-equivalent: raw decay, frequency,
log step, continuous eigenvalues, `exp` and elapsed-time powers, persistent
state, associative composition/accumulation, and B/C projections. The configured
slow edge makes the reason concrete:

    A = exp(-0.5 × 0.0001) ≈ 0.9999500012
    A^1024 ≈ 0.95008863

BF16 and FP16 both round the one-step value to 1.0, destroying precisely the
slow modes this lineage tests. Paired-real storage preserves the complex
diagonal recurrence exactly while making its large real projection GEMMs TF32
eligible. BF16 projections are deliberately rejected until physical-GPU
long-retention and learning-parity evidence exists.

The original native-complex checkpoint format migrates one way into
`paired_real_v1`. Model optimizer moments restart because their parameter
mapping is ambiguous; the lineage, replay, live paired state, actor/critic
optimizers, and wake semantics remain intact.

The posterior categorical state is inferred from the current observation
embedding in parallel. The S5 predictive prior must use the recurrent lifetime
state to predict that posterior. This is the key speed/representation trade-off:
the posterior itself has no historical context, while every downstream feature
contains the long-lived deterministic S5 state. If that factorization loses
important ambiguity resolution, the experiment must show it rather than hiding
it behind the architecture claim.

## Time and lifecycle semantics

One S5 step corresponds to one perception/action opportunity. Parallel replay
does not create extra lived time; it consolidates already-lived sequences.

The runtime now counts missed perception cycles during dormancy and checkpoints
that count with the population. Replay stores two distinct boundaries:

- `first`: a new life/body stream; all recurrent state resets;
- `wake`: the same organism after a sensory blackout; fast modes reset, slow
  modes persist.

At wake, the longest-timescale half of every S5 block survives. Its transition
is raised to the measured blackout duration, so those modes decay and rotate as
simulated time passes. The observation-grounded stochastic state, previous
action, active temporal skill, and fast sensorimotor modes reset. The wake
observation then rebuilds current sensory state alongside persistent context.

A dormant death follows the same temporal rule before the terminal sample is
recorded. A new body is different: all live recurrent state resets. Learned
weights and replay remain the consolidated/heritable substrate under lineage
inheritance, but a newborn does not inherit the donor's immediate sensorimotor
thought.

This is still not unlimited autobiographical memory. S5 state is a learned,
finite predictive summary. Exact episodic recall or culture-scale external
memory would require a later retrieval mechanism and must not be claimed from
long state-space timescales alone.

## Causal wall time and dormant time

Learning credit is a function of lived eligible acts and the configured train
ratio. A measured controller chooses wall-clock execution speed from aggregate
learner throughput, action latency, world cost, debt, headroom, and hysteresis.
It never changes replay, batch, sequence, update publication, or action cadence.
At bounded lag it pauses virtual advancement rather than shedding credit.

If every embodied organism is dormant, whole owed updates are paid before the
quiet world accelerates. Exact unpaced scalar stepping handles falling bodies.
Only settled, interaction-free intervals jump, and only to immediately before
the earliest wake/death, spoilage, ecology, transient, lifecycle, metrics,
checkpoint, or end boundary. The scalar boundary preserves RNG and event order;
bulk accounting preserves virtual solar/integrity/fatigue/age and Aion blackout
duration. An awake resting body or an awake scripted body disables the jump.

## Aion 01 operating point

`configs/brain/aion_01_s5.yaml` uses:

- 1,024 gradient-carrying replay transitions;
- 256 additional burn-in transitions;
- batch size 8;
- two recent replay slots plus reward-salient sampling;
- learned S5 steps initialized from 0.0001 to 0.1, spanning immediate dynamics
  into many-thousand-step modes;
- train ratio 0.25.

The consolidation budget is intentionally substantial:

    8 batch × 1,024 graded steps × 0.25 updates/lived act
    = 2,048 replayed graded transitions per lived act

Beta 013 nominally uses `16 × 64 × 1.0 = 1,024`. Aion therefore receives twice
the replayed evidence per lived act and sixteen times the temporal reach, but
does not demand sixteen times as many optimizer updates merely because each
update became longer.

## Pre-launch gates

No long-run interpretation until:

- the parallel scan matches recurrent evaluation numerically;
- gradients remain finite through a 1,024-step context;
- wake resets only fast modes and replay reproduces the same transition as
  live recurrence;
- checkpoint identity prevents beta/Aion cross-loading;
- old replay checkpoints default safely to no wake and unit elapsed time;
- one full Aion update and three-worker contention are benchmarked in FP32,
  TF32, and hybrid BF16 on RTX 4090, RTX 5090, and H100 SXM;
- the selected batch/context fits with stable memory headroom;
- the governor holds the configured credit budget with zero dropped credit and
  bounded inference publication lag;
- concurrent action p95 meets its deadline and reserved VRAM retains at least
  15% headroom.

Target commands:

```bash
scripts/bench_aion_preflight.sh /tmp/aion-preflight <gpu-hourly-price>
```

## Falsification

Aion does not replace beta merely by running. The S5 bet is weakened or rejected
if matched evidence shows any of the following:

- full-update throughput does not improve once longer context is accounted for;
- recurrent imagination becomes slow enough to erase consolidation gains;
- one-step or rare terminal prediction degrades materially;
- 100/250/500/1,024-step predictive probes show no retained-context advantage;
- delayed-cue, blackout, seasonal, social-identity, or delayed-consequence
  probes do not improve;
- online gradients or eigenmodes become unstable over a long life;
- behavior and development are worse at matched world, population, and replay
  evidence.

If S5 wins prediction and throughput but not behavior, the temporal substrate is
exonerated and the next constraint lies elsewhere in the organism. If it loses
the mechanical gates, beta remains the world-model lineage and Aion becomes an
informative architectural null rather than a forced migration.
