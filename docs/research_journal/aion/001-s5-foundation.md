---
round: aion-001
title: continuous-time S5 foundation
date: 2026-07-10/11
status: complete
question: can a resettable S5 world model consolidate 1,024-step experience efficiently, retain multi-timescale organism context, and preserve slow identity through measured blackouts without decoupling the mind's lived time from the world?
headline: "PARTIAL: S5 sustained lifelong learning and produced repeated late foraging competence (713 safe meals; peak 67 in one body), but survival did not emerge by 7.01M ticks — Aions stayed 78.5% dormant, ingested 21.1% toxic food, lost lifespan in the final cohort, and collapsed onto temporal skill 7."
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
    commit: e90d792
    ticks: 7012204
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

The selected arm ran to a clean checkpoint and shutdown at tick **7,012,204**
(292.2 world-days). The cold archive is
`saves/archive/aion_01_2gpu`: compressed complete logs, the manifest, the final
world/scheduler metadata, and the selected `aion_001` lineage brain as
`best_brain_aion_114.pt.zst`. The other two complete cloud checkpoints were
deliberately not duplicated; the selected blob is the durable learnable artifact.

### Mechanical and learning result

The two-GPU runtime remained coherent for the whole round. At close it carried
zero dropped learning credit, 1.75 updates of maximum debt, no action latching,
and 68,689 cumulative inference-budget misses handled by adaptive pacing. The
two lineages closed at 67,218 / 81,457 completed updates and 272,997 / 329,952
lived replay samples. Final learner time was ~292 ms/update per active brain;
safe/actual throughput was 53.17/53.40 virtual ticks/s.

The world model learned the easy categorical structure quickly and then
plateaued rather than diverging. In 500k-tick eras, model loss fell from 13.97
to the 5-6 range, kind error from 0.0367 to ~0.002, and depth error from 0.0042
to ~0.003. Curiosity rose from 0.0052 to ~0.010 as lived distribution kept
changing. S5 therefore passed the long-running numerical/throughput gate; this
round does not support blaming recurrent capacity or training instability for
the behavioral null.

### Behavior and survival

Across both lineages Aion recorded **713 safe meals, 191 toxic meals, 6,559
digs, and 6,334 placements**. Its embodied safe-eat rate closed at 0.511 per
10k ticks versus the scripted foragers' 12.234 (~24x lower). Toxic food was
21.1% of Aion ingestions despite a 15% world stock and visible toxic identity;
the scripted controls ingested only 0.53% toxic food. Better food acquisition
did not become discrimination.

Foraging competence was nevertheless real and appeared repeatedly rather than
as one founder accident. The strongest body, `aion_086`, ate 67 safe and 9
toxic bushes, repaired 67.5 integrity, and died of poison at age 99,846.
`aion_113` later ate 35 safe / 5 toxic and also died of poison. The recurring
phenotype was therefore *find and manipulate food patches*, not *eat safely to
preserve the body*. Near-exact dig/place pairing is consistent with a learned
pickup/replacement loop; the data do not justify calling it construction.

There were 54 Aion deaths: 36 hibernation-dominant, 17 poison-dominant, and one
exhaustion-dominant. Mean/median/max completed lifespan was
253,255 / 276,991 / 389,416 ticks. The first ten deaths averaged 277,159
(median 292,017); the final ten fell to 233,224 (median 241,246). One 389k
record and substantial repair in food-rich lives did not produce a population
lifespan trend.

Aions were dormant for **78.5%** of recorded body samples. Awake resting was
0.25% in daylight and 0.15% at night, versus scripted foragers' 4.3% / 88.1%:
there is no evidence of deliberate circadian rest or intentional solar naps.
Observed solar recharge followed forced energy-zero hibernation.

The eight-way temporal manager collapsed early. In the final metric quartile,
skill-use entropy was 0.0136 and skill 7 was active in 99.60% of samples. More
ticks did not restore diversity, making the manager/policy interface a concrete
next bottleneck rather than an unspecified request for more training.

### Lineage selection

The persistent founder-001 mind, embodied as `aion_114` at close, is the kept
brain. Across its 29 bodies it achieved 425 safe / 107 toxic meals, 557.0
integrity repaired, the 389,416-tick lifespan record, 81,457 updates, and a
329,952-sample buffer. Founder-000 (`aion_110`) achieved 288 / 84 meals, 415.5
repair, a 365,802 maximum lifespan, 67,218 updates, and 272,997 samples.
Founder-001 was the stronger behavioral and learned-data artifact despite its
active lives often ending earlier from poison.

### Pre-launch capacity evidence

The selected Community Cloud pod has two RTX 3090 24 GB cards and costs
$0.44/hour in total. With Torch 2.12.1+cu130, CUDA 13.0, and protected-FP32 S5
inside `amp_bf16`, two synchronized 20-update two-card gates measured:

- 4.197-4.500 aggregate updates/s across two independent learners;
- 35.68-38.25 sustainable ticks/s after the configured 0.85 headroom;
- 0 action deadline misses, 111.3-115.9 ms action p95, and 128.3 ms worst maximum;
- at most 6,336 MiB peak reserved VRAM on each 24,124 MiB device;
- 34,385-36,861 graded replay timepoints/s and $0.00332-$0.00355 per million
  timepoints.

A fresh 200-tick run and 20-tick resume before launch using the exact round config completed
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
- Only two persistent lineages ran in one world. Spawn location and local bush
  patches confound body-level meal counts; no matched frozen-checkpoint
  evaluation was run across standardized spawn layouts.
- The replay buffers had not reached their 500k capacity at close. This limits
  any claim that Aion had exhausted all possible development, but the stable
  poison/rest/skill-collapse nulls make time alone an inadequate explanation.
- Aion and scripted-forager rates are an affordance sanity comparison, not a
  matched architecture benchmark. The foragers have a hand-written policy.

## Next

1. Use the archived founder-001 brain for frozen, matched-layout probes: safe
   versus toxic choice, food search from standardized spawns, low-energy
   daylight rest, and delayed consequence retention.
2. Diagnose the temporal manager's near-total skill-7 collapse before another
   long ecological run; establish whether the hierarchy is broken, redundant,
   or suppressing actor diversity.
3. Treat S5 as the retained temporal substrate unless the delayed-context
   probes fail. The next live round should target actor/discrimination/skill
   conversion, not replace the world model merely because survival remained null.
4. Preserve the charter boundary: no designer survival reward or scripted food
   target. Any next mechanism must improve learning from endogenous consequence.
