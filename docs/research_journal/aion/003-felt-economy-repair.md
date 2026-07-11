---
round: aion-003
title: felt economy with a valid actor contract
date: 2026-07-11
status: running
question: with the intended felt economy actually present in imagination and a valid bounded continuous policy update, does Aion convert learned world structure into toxin discrimination, deliberate rest, and bodily preservation without naming a target behavior?
headline: "RUNNING: the actor-contract repair passed a 145.3k-tick closed-loop gate (positive imagined viability, bounded variance, low saturation, and separated damage/continuation learning), and a fresh 7.01M-tick world launched from commit dda13ca; there is no behavioral result yet."
runs:
  - save: saves/aion_03_economy_soak
    config: configs/run/aion_03_economy.yaml
    brain: configs/brain/aion_03_economy.yaml
    commit: dda13caefe18c1403400d57228308e3b952c25b2
    ticks: 145300
    role: disposable actor-contract soak (final retained world/scheduler checkpoint 126694; learned brains discarded)
  - save: saves/aion_03_economy
    config: configs/run/aion_03_economy.yaml
    brain: configs/brain/aion_03_economy.yaml
    commit: dda13caefe18c1403400d57228308e3b952c25b2
    ticks: running, finite ceiling 7012204
    role: experiment (two fresh Aions, distinct descendants, four scripted anchors)
baselines: [aion-001, aion-002-invalid]
tags: [aion, wellbeing, pain, actor-contract, dormancy, mortality, descendants, felt-economy]
---

# Aion 003 — felt economy with a valid actor contract

## Interpretation boundary

Aion 03 is not a new reward hypothesis. Its brain YAML is machine-checked equal
to Aion 02's intended scientific configuration. It is the first valid execution
of proposal 006 after Aion 02 was invalidated by implementation defects.

The replacement changes only the actor contract:

1. regulated wellbeing enters the same imagined viability channel optimized by
   the actor, rather than existing only in replay and telemetry;
2. continuous policy standard deviation is smoothly bounded to `[0.1, 1.0]`;
3. continuous score-function samples are detached before `log_prob()`;
4. predicted coma/death body boundaries constrain the continuation used by fear
   and imagined return discounting.

The world, population, action cadence, S5/replay shape, curiosity, bodily affect
coefficients, temperament, and per-lived-act learning budget are unchanged.
Aion 02 learned weights were discarded and do not enter this round.

## Mechanical and target-hardware gate

Local validation passed Ruff, strict mypy across 73 source files, shell/config
checks, and all 305 tests. The suite was split by package after the Mac killed a
single monolithic process under accumulated Torch memory pressure; the isolated
large smoke test and every package partition passed.

The selected Vast.ai instance is two RTX 4090s, 128 GB RAM, 50 GB disk, Torch
2.12.0+cu130, and `amp_bf16`, at $0.7871/hour. Its workspace is not a persistent
volume. The exact two-brain contention gate measured:

| measure | result |
|---|---:|
| safe ticks/s after 0.85 headroom | 47.20 |
| slowest-brain safe ticks/s | 47.21 |
| aggregate learner updates/s | 5.55 |
| action p50 / p95 / max | 12.8 / 28.1 / 122.8 ms |
| action deadline misses | 0 |
| peak reserved VRAM | 6,466 / 6,464 MiB |
| policy std max | 0.737 / 0.721 |
| imagined-action saturation | 0.86% / 0.39% |
| sampled rest | 1.90% / 2.08% |
| imagined viability | +0.135 / +0.180 |

## Closed-loop repair soak

The pre-registered 100k ticks were not enough for a symmetric read: one fresh
organism had seven updates and the other had not crossed the 4,096-sample
warmup. The same atomic world was therefore resumed rather than weakening the
gate. Metrics continued through tick 145,300; the last deliberately retained
world/scheduler checkpoint is tick 126,694.

At the final read, the two brains had 277 / 346 updates and 5,206 / 5,480 lived
samples. Model loss fell to 26.73 / 14.16, depth error to 0.0197 / 0.0044, and
kind error to the final-window mean 0.0519. The actor contract remained intact:

| measure | Aion 000 | Aion 001 |
|---|---:|---:|
| imagined viability | +0.1247 | +0.1557 |
| policy std mean / max | 0.554 / 0.820 | 0.608 / 0.790 |
| imagined-action saturation | 1.3% | 2.5% |
| sampled rest | 0.8% | 1.4% |
| ordinary continuation | 1.000 | 1.000 |
| elapsed-wake continuation | 0.002 | 0.002 |

The damage head separated its sparse classes instead of drowning or predicting
damage everywhere: positive-example probability reached 0.99909 while
negative-example probability fell to 0.00010. Elapsed-wake continuation moved
toward the physical `gamma ** elapsed` target. Values stayed finite and no
learning credit was dropped.

This soak does **not** establish improved behavior. Both Aions were fresh, spent
88.4% / 89.0% of the final 30k-tick window dormant, and ate no food while only a
few hundred updates old. The gate establishes that the configured organism is
now the organism the actor optimizes; the long ecological run decides what it
learns to do.

The evidence-only archive is `saves/archive/aion_03_preflight`. It contains the
contention output, manifest, metrics/events, logs, final check-in, and retained
world/entity/scheduler state. Disposable soak brains were not kept.

## Official launch

The official save `saves/aion_03_economy` launched fresh from clean commit
`dda13caefe18c1403400d57228308e3b952c25b2` with a finite ceiling of 7,012,204
ticks. Supervisor owns the process group and restarts unexpected exits from the
latest atomic checkpoint for only the remaining ticks. A clean finite exit does
not restart.

Vast reports that this instance has no persistent workspace volume. No automatic
off-box mirror is configured for the running world.

## Pre-registered reads

Compared with Aion 01 at the same world and learning budget:

1. toxic ingestion falls below the contemporaneous toxic share of available
   bushes and declines with experience;
2. acute poison transitions remain negative in realized and imagined affect,
   and the damage head stays separated on positive versus negative examples;
3. energy-collapse hibernation and hibernation-dominant death fall while awake
   rest rises beyond Aion 01's 0.15–0.25% band;
4. controlled critic probes preserve `healthy > worn > dying > dead`;
5. descendants inherit learned substrate without carrying the parent's live
   recurrent state or being counted as the same organism;
6. if these mechanisms remain healthy but behavior does not improve, the next
   bottleneck is policy conversion/consequence retrieval—not another bodily
   reward coefficient.

Immediate stop conditions are non-finite losses, zero imagined viability while
wellbeing is positive, policy std outside `[0.1, 1.0]`, sustained action
saturation above 50%, dropped learning credit, incoherent checkpoints, or a
process/checkpoint failure.
