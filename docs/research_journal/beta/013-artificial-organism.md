---
round: 013
title: the artificial-organism round — world model, endogenous affect, and subconscious skills
date: 2026-07-10
status: staged
question: if the beta mind is treated as an artificial organism rather than a flat reward-maximizing policy — direct imagined interoception, separate affect values, learned temporal skills, learnable mortality, chronological moods, and a measured livable body — can its existing desire to survive finally become durable multi-step competence without tasks, demonstrations, pretrained behaviours, or a designer fitness function?
headline: "STAGED, NOT RUN: the first organism architecture combines the critic anima proved constitutive with direct imagined bodily feeling, six non-collapsing affect-value channels, eight unnamed learned temporal skills, weighted mortality learning, chronological boredom, non-blocking inference, and anima's measured world corrections. Mechanical canaries precede any emergence claim; Phase A tests development with lineage continuity, Phase B restores earned reproduction only after an individual organism can learn durable control."
runs:
  - save: saves/beta_12
    config: configs/run/beta_12_organism.yaml
    brain: configs/brain/beta_12_organism.yaml
    commit: not launched
    ticks: 0
    role: experiment (base/cloud developmental arm)
baselines: [012, anima-007]
tags: [artificial-organism, world-model, temporal-skills, affect, mortality, open-endedness]
---

# 013 — the artificial-organism round (initial entry)

## The project statement

This round makes the original goal explicit:

> Provide a persistent simulated reality with physical bodies, scarcity,
> mortality, other minds, communication, and reproduction; provide minds with
> the general mechanisms needed to learn and evolve; then observe whether
> survival, sociality, communication, mating, culture, and more complex traits
> arise as consequences of living there.

The target is **open-ended evolvability without a fixed objective**. “No reward”
means no externally assigned task reward, goal label, behavioral script,
demonstration, pretrained skill, or designer fitness score. It does not mean a
body must be unable to feel its own condition. Energy, integrity, fatigue,
pain, curiosity, boredom, predicted mortality, and self-generated
controllability are endogenous consequences available to an organism, just as
light and water are.

## What the two tracks taught us

The result is no longer “try another reward coefficient.” The combined tracks
now constrain the architecture.

### Anima: feeling without learned expectation is insufficient

Seven rounds tested a world-model-free, backprop-free plastic brain. Frozen
brains tied or beat plastic brains; reduction, level, and level-minus-EMA
valence each failed for the same reason. Over a mortal, declining life, an
unlearned causal baseline inherits the decline and becomes biased. A useful
teaching signal needs a learned expectation — a critic. That closes the anima
family as chartered, while validating bodily feeling as the content the critic
should learn about.

Anima also left world-side measurements beta must inherit:

- signal cost 0.01 silently consumed 36.6% of plastic awake spend; 0.001 makes
  communication physically cheap enough to acquire meaning;
- water speed 0.5 already doubles crossing time, so a 3x drain double-counted
  the hazard; 1.75 plus the OBS-v5 in-water sense is a coherent thick medium;
- wake 38, brownout 25, and repair threshold 60 make one light cycle restore a
  body to functional but not self-repairing condition;
- the action ledger exposed a code defect: configured turn cost and water speed
  were not actually used. This round wires and tests both.

### Beta: caring exists, conversion does not

Round 012 installed the project's first live mortality gradient: healthy bodies
out-valued near-death bodies by +67 where beta 010 had been flat. Behavior did
not convert: zero meals below energy 25, 11/14 deaths on the hibernation clock,
and boredom saturated. The continuation model still predicted 1.0 at death
because 1–15 terminal rows were lost among thousands of unweighted living
rows. The flat action policy also had to rediscover every motor detail at every
step, while long learner updates latched its last action in the real body.

The surviving hypothesis is therefore: **the organism has motivation, but its
brain lacks the representational and temporal machinery that turns it into
stable embodied competence.**

## What changes in this round

This is an architecture round. Its parts are tested together because they form
one causal path from imagined consequence to sustained action.

### 1. Direct imagined interoception

The world model predicts the next proprioceptive state. During imagination the
brain now applies the known bodily equations directly:

    comfort = 3 · (drive_now − drive_next) − 0.01 · drive_next
    viability = −1 · barrier_next
    fear = 0.1 · log P(continue_next)

The learned twohot reward head remains trained and observable, but is a
diagnostic, not a required second rediscovery of energy, integrity, fatigue, and
mortality. Nothing names an action: any imagined behavior that improves the
predicted body receives the same consequence.

### 2. A vector affect critic

One scalar critic can let common curiosity updates numerically erase rare
mortality evidence. The new critic keeps six distributional values until the
actor's final decision:

1. comfort,
2. viability,
3. curiosity/learning progress,
4. boredom,
5. predicted mortality risk,
6. learned-skill controllability.

The actor still chooses one action from their sum. This is not six objectives
given by a designer; it is six internal currencies kept legible long enough to
learn their different time scales. Each has its own value/return/affect metrics.

### 3. Learned temporal skills — the subconscious layer

Every five actions, a manager selects one of eight unnamed latent intentions.
A worker conditions on that intent to control drive, gaze, signaling, and the
gripper. A discriminator observes the change in latent world state over the
skill interval and rewards intentions whose consequences it can identify:

    r_skill = log q(skill | state_after − state_before) + log(number_of_skills)

This supplies reusable “how” without specifying “what.” No latent is assigned
to walking, eating, approaching, avoiding, signaling, or social behavior. If
such primitives become useful, the organism must discover and reuse them. The
manager can then reason at a slower temporal scale while the worker handles
subconscious control.

### 4. Mortality and mood become learnable

- terminal continuation loss is weighted 32x; metrics separately report the
  terminal fraction and predicted continuation on terminal vs living rows;
- boredom pressure advances in `act()` from chronological lived safety and the
  latest learned stimulation estimate. Replaying old experience cannot change
  the order of the organism's mood;
- the all-dull boredom equilibrium is reduced to 0.5, avoiding round 012's
  permanent saturated pressure of 1.0;
- the viability barrier has an explicit total cap of 4.0. The old cap was per
  component, so energy plus integrity could silently reach 8.0.

### 5. Learning no longer freezes action

The learner publishes an immutable encoder/dynamics/controller snapshot every
16 updates. The simulator uses the last complete snapshot while training mutates
its private weights. The old mode remains for existing configs, but this round
should have `act_latched_frac = 0` even during long base-model CUDA updates.

### 6. A measured physical substrate

The round uses 2x food, near-free signal, water drain 1.75, water speed 0.5,
wake 38, and beta's 150k senescence half-life. The scripted forager remains the
reality calibration: if it cannot remain active and repair itself, the world is
still confounding the mind.

Phase A retains each learned mind across replacement bodies (`lineage`). This is
not the final evolutionary experiment: with only three expensive base minds,
earned budding would create weak selection while repeatedly deleting unfinished
development. Phase B will restore budding, mutated inheritance, and differential
reproduction after the individual-organism canaries pass.

## Pre-launch mechanical canaries

No pod launch and no emergence interpretation until all pass:

- full tests, ruff, and strict mypy green;
- a direct-imagination unit probe values predicted hunger reduction positively
  without consulting the reward head;
- vector critic emits finite values/returns for all six channels;
- skill discriminator trains, manager entropy is finite, replay preserves skill
  identity, and skills remain fixed for their configured duration;
- terminal weighting is numerically exact and live metrics expose
  `cont_terminal`, `cont_alive`, and `terminal_frac`;
- a concurrency test proves a snapshot brain enters learning while its action
  lock is held;
- the configured turn cost and water speed multiplier change the measured
  physical result;
- on the target CUDA box, benchmark three learners and set world speed so the
  train-ratio identity remains within the 1024-update debt cap.

## Pre-registered live questions

### P1 — does the hierarchy stay alive?

After warmup, `skill_usage_entropy` should stay above 0.5 and discriminator
accuracy should exceed chance (12.5%) without converging to a single skill.
The important evidence is not the index: it is temporally persistent,
repeatable action/state-change profiles per index. Failure means fix the skill
mechanics before interpreting survival.

### P2 — can imagination represent death?

`cont_terminal` must fall materially below `cont_alive`; it must not remain
near 1.0 at the lethal floor. The fear affect/value should be near zero in safe
states and negative near recorded terminal states. Failure isolates terminal
conditioning, not behavior.

### P3 — does motivation convert into survival competence?

The primary behavioral read remains meals while genuinely hungry: nonzero and
rising meals below energy 25, not a larger sated binge. Supporting reads:
dormant fraction falls from round 012's 0.84–0.91, death ages leave the ~347k
hibernation clock, awake self-repair persists across lineages, and poison
avoidance improves from consequence.

### P4 — do temporal skills carry multi-step behavior?

Around food encounters, ask whether stable skills compose into approach →
gaze/align → eat, and whether the same skill has a consistent consequence in
other places. Around water, poison, bodies, and sound, look for reusable
avoidance, investigation, signaling, or feeding sequences. Name them only
after measurement; never train the names into them.

### P5 — are curiosity and moods still informative?

Boredom pressure must remain dynamic rather than saturating; learning-progress
curiosity should still go stale in mastered regions; the six critic channels
should show trade-offs rather than identical traces. Communication remains
nearly free, but structured signaling is an observation, not a round gate.

## Decision branches

- **A mechanical canary fails:** stop and repair that mechanism. No behavioral
  verdict.
- **Mechanics pass, skills diversify, survival remains flat:** the next suspects
  are planning horizon, model accuracy at rare affordances, and the skill
  manager's credit assignment — not another hand-authored reward.
- **Hungry eating and durable active life emerge:** replicate, then begin Phase B
  with earned budding and mutated inheritance. The question becomes whether
  natural selection stabilizes and elaborates the learned phenotype without a
  fitness score.
- **Complex social or communicative behavior appears first:** record it without
  retrofitting a success criterion. Open-ended results are allowed to answer a
  different question than survival competence.

## Operations

Staged, not launched. Use `configs/run/beta_12_organism.yaml`, run paced, and
benchmark on the exact CUDA host first. Pull at least one complete brain blob
early. At each checkpoint verify train ratio, action-latch fraction, terminal
separation, skill health, affect-channel scales, hungry meals, energy ledger,
and the forager anchor. The initial architecture should be treated as a
research instrument until those measurements say it is healthy.
