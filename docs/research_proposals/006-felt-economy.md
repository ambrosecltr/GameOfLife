---
proposal: 006
title: the felt economy — a coherent bodily basis for Aion
date: 2026-07-11
status: invalidated, actor-contract-repair-required
targets_round: aion_02
question: Aion 01 learned a usable food-manipulation phenotype but remained mostly unconscious, ate toxic food indiscriminately, never learned deliberate rest, and collapsed its temporal manager. Can a coherent bodily economy — positive regulated wellbeing, acute injury pain, honest unconscious-time discounting, represented bodily death, and distinct descendant inheritance — turn learned world structure into a body-preserving policy without naming or rewarding a target behavior?
depends_on: [003, 005, aion-001]
baselines: [aion-001, beta-013]
tags: [aion, reward-geometry, wellbeing, pain, interoception, dormancy, mortality, inheritance, calibration]
---

# 006 — the felt economy

## Status and purpose

The first implementation run was invalidated at tick 3,778,200. Wellbeing was
present in replay reward and telemetry but absent from direct imagined affect;
continuous policy variance was unbounded; and continuous actions were not
detached for the stated REINFORCE update. Consequently, that run did not test
this proposal. The bodily contract below remains the hypothesis, while any
replacement run must first pass the actor-level repair gates recorded in
`docs/research_journal/aion/002-felt-economy.md`.

This is a **foundation-alignment round**, not a one-knob causal ablation. Earlier
rounds repeatedly repaired one mechanism while leaving the rest of the
organism's value system internally inconsistent. Aion 02 deliberately aligns
the bodily contracts together, validates each one mechanically, and then asks
what behavior emerges from the composed organism.

That makes the interpretation boundary explicit:

- a positive result establishes that the aligned organism is a viable research
  baseline; it does not identify one change as the cause;
- a negative result routes through the separately logged mechanical gates
  before motivating another reward coefficient;
- later research reintroduces parked mechanisms one at a time.

## What Aion 01 actually established

Aion 01 closed cleanly at tick **7,012,204**. The authoritative result is
`docs/research_journal/aion/001-s5-foundation.md`; the selected founder-001
brain and its chronological 329,952-sample replay are archived at
`saves/archive/aion_01_2gpu`.

The durable findings are:

- the S5 substrate remained numerically stable for 81,457 updates in the
  selected lineage and learned easy categorical world structure;
- Aion repeatedly acquired a real food-search/manipulation phenotype: 713 safe
  and 191 toxic meals across both lineages, with 67 safe meals in the strongest
  body;
- that competence did not become bodily preservation: toxic food was 21.1% of
  Aion ingestion versus a 15% world stock and 0.53% for scripted controls;
- 36/54 Aion deaths were hibernation-dominant and 17/54 poison-dominant;
- Aions were dormant for 78.5% of body samples and deliberately rested for only
  0.15-0.25% of awake samples;
- temporal skill 7 occupied 99.60% of final-quartile samples.

The correct architectural conclusion is limited: S5 passed the long-running
numerical and throughput gate, so Aion 02 retains it. Live matched-layout
retention and delayed-consequence probes are still required before claiming a
memory advantage over Beta.

### Corrections to the first forensic draft

The first draft usefully identified the felt-economy hypothesis but overstated
several measurements:

- `value` is a mean over the current imagination batch, not a controlled
  value-of-death probe;
- `terminal_frac` is the fraction of real terminal targets sampled from replay,
  not an imagined-suicide rate;
- `affect_*` channels are imagined actor-training components, not realized
  per-lived-step return;
- the first poison arithmetic row incorrectly included viability reduction even
  though Aion 01 configured `viability.scale: 0`;
- the early skill occupancy and entropy figures were mutually inconsistent;
  the final closeout's 99.60% occupancy and 0.0136 normalized entropy are the
  coherent pair.

These corrections weaken the claim that Aion 01 was actively seeking death.
They do not change the observed hibernation, poison, rest, or skill-collapse
nulls, or the need to rebuild the bodily economy from explicit contracts.

## Design decisions

The project owner accepted these decisions on 2026-07-11:

1. **One body is one organism.** Bodily death terminates that organism. Learned
   state may seed a distinct descendant, but the parent is not reincarnated.
2. **Regulated conscious life feels positive.** Wellbeing is a pure function of
   bodily state, not an action or named behavioral target.
3. **Hibernation is emergency coma.** It is involuntary and affect is suspended.
   Stillness while awake is the existing voluntary sleep/rest affordance.
4. **Acute injury causes pain.** Poison, falls, and exhaustion are acute;
   ordinary wear and unconscious hibernation decay lower wellbeing without
   creating constant conscious pain.
5. **Skills and boredom are parked.** Fear remains unchanged until continuation
   learning is directly verified.
6. **Aion 02 starts fresh.** No Aion 01 weights enter the live round; the same
   world, population size, action cadence, replay budget, and two-GPU shape are
   retained for comparison.

## The bodily value contract

The round must satisfy this ordering on both analytic states and recorded Aion
01 bodily trajectories:

```text
healthy > worn > dying > dead = 0
```

A safe, fed, intact, rested organism receives bounded positive wellbeing. Hunger,
fatigue, injury, and proximity to a lethal boundary reduce that stream. Acute
damage is felt immediately. Death ends the stream. Coma earns no felt reward and
discounts the value of the eventual wake by the physical time that passed.

This is not a constant `+1 alive` bonus. It depends only on regulated bodily
state and approaches zero at the capped lethal boundary. No food identity,
action, survival label, lifespan score, or externally assigned task enters the
learner.

## Implementation contract

### 1. Regulated wellbeing

The standing viability tax is disabled for Aion 02. Its replacement is:

```text
safe       = clamp(1 - viability / barrier_cap, 0, 1)
regulated  = exp(-comfort_decay * comfort_drive)
wellbeing  = wellbeing_weight * safe * regulated
```

With `wellbeing_weight: 0.25` and `comfort_decay: 1.0`, the maximum discounted
stream at `gamma: 0.997` is approximately 83.3. Merely staying above the lethal
barrier is insufficient: hunger, injury, or fatigue lowers the comfort factor.

The existing HRRL drive-reduction term remains. Its small level penalty remains
as the immediate felt direction of a deficit; wellbeing supplies the positive
standing level that the penalty-only economy lacked.

### 2. Acute pain

Pain is the normalized integrity lost on a transition explicitly marked by the
world's acute-damage event:

```text
pain = -pain_weight * max(previous_integrity - integrity, 0)
```

`pain_weight: 5.0` makes a 12-point toxic injury worth `-0.6`, an 8-point fall
worth `-0.4`, and chronic micro-wear worth no acute pain. The world model gains
a damage-event head so imagination gates predicted integrity loss on predicted
acute damage rather than treating all senescence as pain. Positive damage
examples receive 32x model-loss weight and reward-salient replay.

### 3. Unconscious time

`blackout: suspended` replaces Aion 01's `priced` mode.

- slow S5 context survives and advances by the measured missed-act count;
- fast sensorimotor and stochastic state reset at wake;
- comfort, viability, and pain deltas are severed across the unconscious gap;
- no wellbeing or pain accumulates while unconscious;
- the wake transition's continuation target is multiplied by
  `gamma ** (step_scale - 1)`, so the full transition discount is
  `gamma ** step_scale` rather than one ordinary step;
- death remains an exact zero continuation target.

This is semi-Markov discounting of an unexperienced interval, not a duration-
multiplied reward. In particular, accumulated integrity loss is not charged as
pain once per missed cycle.

### 4. Death and descendants

`death_terminal: true`, terminal example weighting, and `fear_weight: 0.1`
remain. The continuation instrumentation now separates actual death targets
from elapsed-time discounts so a long coma cannot be mislabeled as a death in
the report.

The runtime adds `inherit_weights: descendant`. A dead brain finishes its final
terminal record; a newly constructed brain then inherits its learned weights,
replay, optimizer state, and mutated temperament while resetting recurrent
state, accumulated mood, and per-life returns. The child is a distinct object
and a distinct organism. Legacy `lineage` reincarnation remains unchanged for
historical runs.

### 5. Parked mechanisms

- `temporal_skills.enabled: false`: removes the collapsed manager, worker skill
  input, discriminator, and unearned skill affect from this round.
- `boredom.weight: 0`: removes an uncalibrated pressure while the new bodily
  distribution is unknown.
- fear, curiosity, temperament, S5 shape, replay shape, train ratio, world
  physics, and population composition otherwise remain unchanged.

### 6. Observability

Aion 02 records:

- wellbeing, pain, damage-head loss and positive/negative probabilities;
- actual terminal fraction separately from elapsed-discount fraction;
- continuation predictions on ordinary life, elapsed wake transitions, and
  death transitions;
- exact per-life comfort, viability/wellbeing, and pain returns;
- signal vector and magnitude per robot, with recent channel entropy in
  `aion_stats.py`;
- energy and integrity before/after eating and poisoning, plus integrity around
  falls;
- measured blackout duration and effective continuation discount.
- explicit parent/child inheritance events for descendant births.

## Offline calibration result

Command:

```bash
uv run python scripts/aion_economy_screen.py \
  saves/archive/aion_01_2gpu/best_brain_aion_114.pt.zst \
  --brain configs/brain/aion_02_economy.yaml
```

The screen uses the selected Aion 01 lineage's chronological replay with body
starts and wakes treated as affect discontinuities. It is a mechanical reward
screen, not a counterfactual behavioral claim.

Results with the staged defaults:

| recorded band | samples | wellbeing/step | body affect/step |
|---|---:|---:|---:|
| healthy | 32,774 | +0.231960 | **+0.230580** |
| worn | 74,154 | +0.154801 | **+0.148416** |
| dying | 94,618 | +0.040782 | **+0.029661** |
| dead | — | 0 | **0** |

The required ordering passes. Across all non-birth replay samples, mean bodily
affect changes from Aion 01's large negative viability-dominated geometry to
`+0.108426`; acute pain remains sparse and sharp (108 recorded damage events,
minimum transition `-0.600586`).

Counterfactual meal gate:

| energy | integrity | ripe | toxic |
|---:|---:|---:|---:|
| 0.15 | 1.00 | +1.474511 | **-0.090739** |
| 0.30 | 1.00 | +1.284698 | **-0.109195** |
| 0.50 | 0.80 | +0.636052 | **-0.384855** |
| 0.30 | 0.60 | +0.851472 | **-0.456147** |
| 0.85 | 1.00 | +0.250000 | **-0.485303** |

All ripe meals remain positive and every toxic meal is negative. The archived
wakes carried a median 3,977 missed acts; the median effective future discount
under the suspended contract is approximately `0.000006`.

## Original gates and why they failed

Local gates passed on 2026-07-11:

- `uv run ruff check .`;
- `uv run mypy packages` (73 source files);
- `uv run pytest -q` (299 passed; 14 existing TorchScript deprecation warnings);
- exact Aion 02 config/YAML and shell validation;
- fresh tiny Aion 02 act, learn, checkpoint, suspended-wake, and descendant tests;
- the archived Aion 01 81,457-update checkpoint loaded with its legacy six-channel
  critic and no damage head.

Remaining external gates:

1. Two-GPU contention gate passes the exact Aion 02 config with zero dropped
   credit, zero benchmark deadline misses, at least 15% VRAM headroom, and at
   least 25 safe ticks/s.
2. A short paced soak verifies:
   - damage-head positive examples are learned rather than drowned;
   - healthy/worn/dying/dead value ordering holds on controlled states;
   - elapsed wake continuation approaches the time-discount target;
   - wellbeing is positive while conscious and absent during coma;
   - no new action-latching, NaNs, or reward-scale domination appears.

These gates were necessary but not sufficient. They established finite
mechanics, replay calibration, and throughput, but never asserted parity between
configured bodily affect and the actor's imagined affect, never bounded policy
variance, and never checked the score-function gradient boundary. No replacement
ecological run launches until those actor-level gates and a short closed-loop
soak pass.

Target commands:

```bash
scripts/bench_aion_2gpu.sh /tmp/aion-02-preflight <pod-hourly-cost> \
  configs/brain/aion_02_economy.yaml
scripts/start_aion_02_2gpu.sh saves/aion_02_economy_soak 100000
```

## Pre-registered live predictions

Compared with Aion 01 at matched world and learning budget:

1. Toxic ingestion falls below the contemporaneous toxic share of available
   bushes and declines with experience.
2. Acute poison transitions remain negative in realized and imagined affect;
   the damage head separates damage from ordinary wear.
3. Energy-collapse hibernations and hibernation-dominant deaths fall; deliberate
   awake rest rises above Aion 01's 0.15-0.25% band.
4. Controlled critic probes preserve `healthy > worn > dying > dead`.
5. Descendants inherit learned competence without carrying the parent's live
   recurrent state or being counted as the same organism.
6. If food manipulation remains but safe eating does not improve despite gates
   1-4 passing mechanically, the remaining bottleneck is policy conversion or
   consequence retrieval, not another bodily reward coefficient.

## Round files

- `configs/brain/aion_02_economy.yaml`
- `configs/run/aion_02_economy.yaml`
- `scripts/aion_economy_screen.py`
- `scripts/bench_aion_2gpu.sh` with the Aion 02 brain config argument
- `scripts/start_aion_02_2gpu.sh`

The Aion 01 archive is calibration evidence only. Aion 02 begins with fresh
brains and does not load the selected Aion 01 artifact into the live population.
