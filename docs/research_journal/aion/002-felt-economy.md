---
round: aion-002
title: coherent felt economy
date: 2026-07-11
status: invalidated, actor-contract-defects
question: does a coherent body-valued economy convert Aion 01's learned food-manipulation competence into toxin discrimination, deliberate rest, and bodily preservation without naming a target behavior?
headline: "INVALIDATED AT 3,778,200 TICKS: wellbeing was logged but omitted from imagined actor affect, continuous policy variance was unbounded, and continuous REINFORCE actions were not detached. The run does not test the felt-economy hypothesis."
runs:
  - save: saves/aion_02_economy
    config: configs/run/aion_02_economy.yaml
    brain: configs/brain/aion_02_economy.yaml
    commit: 8251f53398478b4ca4a819ffee086ebc80b5624c
    ticks: 3778200
    role: invalidated implementation run (two fresh Aions, distinct descendants, four scripted anchors)
baselines: [aion-001, beta-013]
tags: [aion, wellbeing, pain, dormancy, mortality, descendants, felt-economy]
---

# Aion 002 — coherent felt economy

## Invalidation

The paid run was stopped and the pod deleted after three actor-contract defects
were confirmed against deployed source and live metrics:

1. `_viability_reward()` computed wellbeing for replay and telemetry, but the
   direct-proprio `_imagination_affect()` path omitted it. Aion 02 configured
   viability reduction and tax to zero, so live `affect_viability` was exactly
   zero while `wellbeing` reported approximately +0.13. The actor never received
   the round's defining positive-conscious-life signal.
2. Continuous policy standard deviation used unbounded `softplus(raw_std) + 0.1`
   while the actor maximized pre-squash Gaussian entropy. Mean policy entropy
   rose from -1.20 in the first 500k ticks to 20.67 after 3.5M; current bodies
   reached 11.7 and 29.3. Tanh-squashed commands saturated near -1/+1.
3. The continuous action sample remained attached when passed back through
   `log_prob()`. The update labelled REINFORCE was therefore not a score-function
   gradient. Discrete actions did not share this defect.

The observed 80.3% Aion dormancy was the exact physical consequence: aggregate
awake drain was 0.01468 energy/tick, predicting 2,589 ticks from wake energy 38;
the observed median was 2,623 awake ticks followed by 19,883 dormant ticks.
Nineteen of 28 Aion deaths were hibernation-dominant and nine poison-dominant.

No behavioral result from this run may confirm or falsify proposal 006. Learned
brain artifacts were deliberately discarded; metrics, events, manifest, log,
final report, and deployed source were retained for the postmortem.

Required repair gates before any replacement run:

- controlled imagined states prove wellbeing enters the viability affect channel;
- predicted energy/integrity boundaries constrain imagined continuation;
- continuous score-function samples are detached from their generating action;
- policy standard deviation remains in [0.1, 1.0], with saturation and sampled
  rest reported by every learning update;
- a short closed-loop soak shows finite losses, nonzero imagined viability,
  bounded policy saturation, and no upward entropy/variance runaway.

## Why this round

Aion 01 retained a stable S5 world model and repeatedly learned to find and
manipulate food, yet spent 78.5% of sampled body time dormant, ingested toxic
food at 21.1%, deliberately rested for at most 0.25% of awake samples, and
collapsed its temporal manager onto one skill. Proposal 006 treats those nulls
as an organism-level alignment problem rather than another isolated coefficient.

This is intentionally a foundation rebaseline. The individual contracts are
mechanically gated; the combined live result will not be attributed to one knob.

## What changes

- A regulated conscious body receives bounded positive wellbeing from viability
  and comfort state. The Aion 01 standing danger tax is removed.
- Acute poison/fall/exhaustion integrity loss produces linear pain, supported by
  a new damage-event world-model head. Chronic wear and unconscious decay do not.
- Emergency hibernation suspends affect, preserves and advances slow S5 context,
  and discounts the eventual wake by measured elapsed time.
- Bodily death terminates one organism. `inherit_weights: descendant` creates a
  distinct newborn carrying learned substrate instead of reincarnating the same
  brain object.
- Collapsed temporal skills and uncalibrated boredom are disabled. Fear remains
  at the Aion 01 value until continuation learning is verified.
- Signal use, meal vitals, falls, wellbeing, pain, elapsed continuation, and
  death continuation gain standing observability.

The world config, population count, action cadence, S5/replay shape, curiosity,
temperament, and per-lived-act update budget remain matched to Aion 01.

## Mechanical calibration

The selected Aion 01 founder-001 replay contains 329,952 chronological samples,
29 body streams, 311 wakes, and 108 acute-damage observations. Screening it with
the Aion 02 config produced:

| body band | samples | mean body affect |
|---|---:|---:|
| healthy | 32,774 | +0.230580 |
| worn | 74,154 | +0.148416 |
| dying | 94,618 | +0.029661 |
| dead | — | 0 |

All five counterfactual ripe meals remain positive; all five toxic meals are
negative (`-0.091` to `-0.485`). Archived wakes had median elapsed scale 3,977
acts and median future discount approximately `0.000006`.

The reproducible command and full table live in proposal 006.

## Pre-registered reads

1. Toxic ingestion falls below contemporaneous toxic availability and declines
   with experience.
2. Damage prediction separates acute injury from ordinary wear; realized and
   imagined poison affect remain negative.
3. Hibernation and hibernation-dominant death fall while awake rest rises beyond
   the Aion 01 band.
4. Controlled critic reads maintain `healthy > worn > dying > dead`.
5. Descendants inherit consolidated competence but not the parent's live state
   or per-life affect.
6. If these gates pass and safe behavior still fails, the next bottleneck is
   policy conversion/consequence retrieval, not bodily reward calibration.

## Original launch gates

- local **PASSED**: ruff, strict mypy (73 files), 299 tests, archive compatibility,
  calibration, config and shell validation;
- pod: exact two-GPU Aion 02 contention/VRAM/action gate;
- short soak: finite losses, learned damage positives, separated continuation
  targets, correct descendant checkpointing, bounded reward/value scales;
- these gates were insufficient: they checked finite execution and logged
  wellbeing, not whether the actor actually received wellbeing or had a valid
  bounded action-gradient path.
