---
proposal: 002
title: plastic-valence — a backprop-free, world-model-free brain that learns by feeling
date: 2026-07-07
track: anima                 # provisional name for the plastic-neuroevolution track (vs beta = world-model)
status: proposed
targets_round: anima_01      # first round of a new track; forks beta_09's base, runs local on M1
question: Can a brain with no gradient descent and no world model — a recurrent net whose fast weights adapt online via neuromodulated Hebbian plasticity, gated by an evolved homeostatic valence signal — keep itself alive in beta's world, and does the evolved valence map diverge across lineages into persistent individuality?
depends_on: [009]            # forks beta_09's world + drive-reduction block; does NOT require budding (001) to be built
independent_of: [001]        # uses the existing respawn+inherit+mutate loop; budding is an optional later integration
arc: plastic-valence brain (A, this doc) → self-model / introspection (B) → dreaming / offline consolidation (C) → evolved aging / plastic senescence (D)
sibling_track: active-inference (a parallel comparison track, not a sequel in this arc)
tags: [architecture, neuroevolution, plasticity, neuromodulation, affect, homeostasis, individuality, continual-learning]
---

# 002 — plastic-valence

This is the first round of a **new track**, not the next beta round. Every round from 004
to 009 asked the same question with the same architecture — can *this one Dreamer* learn
fast enough — and the answer kept hinging on one brain's gradient-descent learning speed
(Q5). This proposal changes the architecture, not the world. It keeps beta's world, beta's
body, beta's obs contract, and even beta's definition of *feeling*, and it removes
everything that sits between feeling and learning: the world model, the imagined rollouts,
the actor-critic, and gradient descent itself.

The charter now says the brain family is a research variable and `beta` is only the
world-model track. This is the second track. Its provisional name is **anima**; the code
brain family is `kind: plastic`. (Track = experiment lineage / save prefix; kind =
architecture family in code — the two are deliberately distinct, per the naming note in
`CLAUDE.md`. The track name is changeable up to launch, since round configs freeze then.)

## The one new idea: decouple feeling from maximization

beta already has feeling. Its homeostasis is HRRL drive-reduction (Keramati & Gutkin):
`reward = movement of internal state toward setpoints`, so the meal that saves a starving
agent outweighs a snack at satiety. That valence signal is real and it is good. What beta
does *with* it is the thing worth challenging: valence becomes a **reward an actor-critic
maximizes inside an imagined world-model rollout**, via backprop. Feeling → reward →
planner → gradient.

This track keeps the *identical* feeling signal and strips out everything downstream. Let
`M` = beta's drive-reduction signal. In beta, `M` is a reward to be maximized. Here, `M`
is a **neuromodulator that gates plasticity**: it decides which of the synapses that just
fired get consolidated, and nothing performs `argmax` over expected `M`. There is no
planner reaching for pleasure; pleasure just decides what sticks.

That is the whole bet, and the two tracks answer it head-to-head in the same world, same
obs contract, same drive definitions: **does feeling have to be planned-toward to shape a
life, or is it enough for feeling to decide what is learned?** beta gives one answer; a
nervous system gives the other.

## Why this, why now

- **It forks beta_09, which cleared the signal-conditioning failures.** beta_09 fixed the
  three round-008 pathologies (running-std re-inflation, cold-start trickle floor,
  boredom-with-no-accumulator). The drive-reduction block that survived that conditioning
  is exactly what this track reuses as `M`, so we inherit a *calibrated* feeling signal
  instead of re-deriving one.
- **It is the cheapest brain the project can run.** No imagination horizon, no ensemble of
  world-model heads, no replay-buffer gradient steps — just forward passes plus a local
  outer-product update. That is what lets us run a **large founder population on the M1**
  (see calibration), which is the point: more bodies means more unpredictable interactions
  and more chances for emergent events, the thing beta's 3-dreamer runs could never have.
- **It is the most literal reading of the project premise.** "Provide the capacity, do
  zero pretraining, let evolution do the work." Here there is no optimizer to pretrain:
  within a life the brain adapts only through local plasticity; across lives the genome
  adapts through inheritance + mutation + differential survival. Learning and evolution are
  the *only* two adaptive processes, and neither is gradient descent.
- **It makes "feeling / identity" structural rather than observed.** beta_07 *found*
  persistent personalities; here identity is the slow genome plus the parts of the net that
  never get consolidated — the thing that persists while the fast weights churn.

## The architecture: an evolved neuromodulated plastic network

No gradient descent anywhere. A small recurrent core (start with a GRU; the continuous-time
liquid-net substrate, CfC/LTC, is deferred to a sequel) with two kinds of weights:

- **`W_slow`** — set once at birth from the genome, fixed for the whole life. The innate
  wiring.
- **`W_fast`** — plastic, starts at zero (or small), and is the only thing that changes
  while the agent is alive.

`W_fast` follows a **three-factor / neuromodulated Hebbian rule** (the "eligibility trace ×
global modulator" rule from computational neuroscience; Miconi's differentiable-plasticity
and backpropamine are the reference for making the plasticity coefficients themselves
adaptive — except we *evolve* them across lineages instead of backpropping them):

```
trace  ← (1 - 1/τ) · trace + (1/τ) · (pre ⊗ post)     # per-synapse eligibility
ΔW_fast = M · α · trace − decay · W_fast              # M gates consolidation
```

- **`M`** is the evolved homeostatic valence — beta's drive-reduction signal, computed from
  the same interoceptive state (energy/integrity/rest deficits and their deltas). Positive
  `M` (ate while hungry, repaired) consolidates the behavior that just fired; negative `M`
  (took damage, starving) is anti-Hebbian and suppresses it. Pleasure and pain are the
  learning gate, literally.
- The interoception→`M` mapping is **genome-encoded and evolved**, not designed. We give
  the setpoints (as beta does) and let lineages evolve how sharply their bodies feel each
  deviation. We never write "food is good" — an agent discovers that eating restores energy
  and its evolved valence marks that as worth keeping.

**Where exploration comes from.** In a no-reward architecture there is no policy being
optimized toward anything, so movement can't come from maximizing a curiosity reward (as it
does in beta). It comes from intrinsic motor activity — a heritable **restlessness** in the
motor output (action-space noise whose scale is a gene) plus whatever the recurrent
dynamics produce. Evolution shapes restlessness against consolidation: too little and the
agent never stumbles into food; too much and it never settles on what worked. This replaces
beta's curiosity drive with something cheaper and more primitive, and whether it is *enough*
is P1 below.

## The genome, and two ways to inherit

The simplified digital genome (reuses proposal 001's genome-primary decision and the
existing `Brain.inherit` path, `base.py:55`):

| gene group | what it sets |
|---|---|
| valence map | interoception → `M` weights (which deficits matter, how sharply) |
| plasticity | per-layer `α`, eligibility `τ`, `decay` |
| modulator | gain / sign / baseline of `M` |
| restlessness | motor-noise scale (the exploration drive) |
| innate wiring | `W_slow` init seed / scale |
| temperament | the existing heritable multipliers, carried over |

Inheritance has a clean, cheap ablation that maps onto the *existing* `inherit_weights`
flag and directly probes research-question 4 (cultural transmission):

- **Darwinian (`genome`)** — child inherits the mutated genome only; `W_fast` reinitialised.
  Pure genetic evolution: no learned experience crosses the generation gap. Nothing but the
  recipe is heritable.
- **Lamarckian (`lineage` / `random_living`)** — child also inherits the parent's *learned*
  `W_fast`. Experience passes on, as beta's weight-inheritance does today.

Running both and comparing is how we separate "the lineage evolved a better innate brain"
from "the lineage carried forward a well-trained one" — the same confound proposal 001 flags
(P3), but here it is a first-class arm because the genome and the learned weights are
physically separate tensors.

## Modelling the fundamentals of life (mapped to the real contract)

Everything routes through the existing `Observation`/`Action` contract — **no OBS_VERSION
bump**, invariant 6 untouched:

- **See / hear** — `rays` (color vision + gaze), `sound`, as-is.
- **Feeling (pleasure / pain)** — `M`, the evolved valence over interoceptive `proprio` +
  `events` (ate, took_damage). Native, and the *engine* of this round, not a bolt-on.
- **Energy / body / homeostasis** — the drive setpoints inherited from the beta_09 block.
- **Identity** — the slow genome + `W_slow` + the un-consolidated net: the part that
  persists when the fast weights churn.
- **Introspection / self-model (round B)** — a small head that predicts the agent's own next
  `proprio`/`M`; its error is a felt "surprise about myself" and the natural second
  modulator (a novelty/curiosity signal this architecture can actually compute without a
  world model).
- **Dreaming (round C)** — during the night dormancy the world already has, replay the
  eligibility traces through the plastic rule: offline consolidation, nearly free here.
- **Aging (round D)** — let `α` decay with `age_ticks` (a biological critical period); ties
  straight into the senescence arc.

## Config surface (illustrative — sized in the calibration pass, not final)

New brain family, forked from beta_09's drive block (the valence source) with the
world-model / actor-critic / replay / training / curiosity blocks removed:

```yaml
# configs/brain/anima_01_plastic.yaml
kind: plastic
core:
  hidden: 256                  # small GRU; no preset — this net has no world model to size
plasticity:
  alpha: 0.1                   # base plasticity coefficient (per-layer alpha is a gene; this is the founder mean)
  tau: 20.0                    # eligibility-trace time constant (act-steps)
  decay: 1.0e-3                # fast-weight decay toward zero
restlessness: 0.2              # founder-mean motor-noise scale (a gene)
# --- reused verbatim from beta_09_dreamer.yaml: the valence signal M ---
valence:                       # = beta's `reward.drive` block, now a neuromodulator not a reward
  scale: 3.0
  level_penalty: 0.01
  pow_m: 3.0
  pow_n: 2.0
  energy_setpoint: 0.85
  energy_weight: 1.0
  integrity_setpoint: 1.0
  integrity_weight: 1.0
  rested_setpoint: 1.0
  rest_weight: 0.5
genome:
  enabled: true
  sigma: 0.25                  # founder diversity (log-stddev over gene multipliers)
  mutation_sigma: 0.1          # drift applied on inherit
```

Run config, forked from `beta_09_conditioning.yaml` but retargeted to the M1 (cpu) and to a
**large cheap founder population** — no cuda, no pacing math (no learner-thread gradient
step to pace against; plasticity is a per-act-step local update):

```yaml
# configs/run/anima_01.yaml
world_config: configs/world/default.yaml       # same world as beta — comparability
tick_rate: 20
act_every: 5
devices: { inference: cpu, learning: cpu }
population:
  target: 24                   # illustrative; sized in calibration to hold a usable tick rate
  respawn_delay_ticks: 1200
  inherit_weights: genome      # Darwinian arm (anima_01a); lineage = Lamarckian arm (anima_01b)
  mix:
    - brain: configs/brain/anima_01_plastic.yaml
      count: 18
    - brain: { kind: scripted_forager }
      count: 6                  # the cross-round forage anchor, unchanged
```

## Integration points (where the code changes)

- **`gol_brains/plastic/`** — new family: a GRU core, the three-factor plastic linear
  layer, the genome dataclass, and the `M` computation (lift beta's drive-reduction math out
  of the dreamer reward module so both tracks share one definition of feeling).
- **`registry.py`** — add `kind: plastic` to `build_brain`'s dispatch.
- **`base.py` methods** — `act` runs the forward pass + applies the plastic update inline
  (learning is *in the act step*, so `learn()` stays a no-op and `target_train_ratio()`
  returns 0 — the learner thread never schedules this brain, and invariant 5 is satisfied
  trivially). `introspect()` surfaces `M`, mean `|W_fast|`, and trace magnitude.
  `state_dict()` serialises genome + `W_slow` + `W_fast`. `inherit()` mutates the genome and,
  depending on the flag, reinitialises or copies `W_fast`.
- **No world.py / scheduler.py changes** — this round uses the existing respawn+inherit
  loop; only a new `inherit_weights: genome` mode (reinit fast weights) is added. Endogenous
  budding (001) is an *optional later* integration, not a dependency.
- **`gol-stats` / metrics** — valence-map census across lineages (does interoception→`M`
  diverge?), per-agent `M` and plasticity traces, foraging rate vs the forager anchor and vs
  beta, per-tick wall cost (the "runs quick / fits on M1" claim, measured).

## Calibration first (the M2 / M1-fit pattern)

Before the real run, size the population to the hardware, exactly as the M2 economy was
calibrated on scripted foragers first:

1. **Measure per-tick cost** of one `plastic` brain on the M1 (cpu). It should be far below
   a dreamer's ~500 ms/update — forward pass + one outer product.
2. **Scale `target`** up until the world holds a usable tick rate (aim to keep speed-1
   real-time-ish, ~20 t/s; more agents is better for emergence, so push it as far as the
   tick rate tolerates). Record the number; that is the round's founder population.
3. **Lock it, then run.** Separates "the population is too big for the M1" from "plastic
   brains can't forage."

## Predictions (pre-registered — including the failure branches)

- **P1 — does it forage at all? (the chicken-and-egg, first thing to check.)** A brain with
  only restlessness for exploration and only Hebbian consolidation for learning may never
  stumble into enough meals to consolidate foraging before dying. If so, the population runs
  entirely on the respawn floor and forage rate sits at noise — a clean, informative null
  that says *this world needs planned-toward reward, not just felt reward*. Mitigations that
  don't cheat: the large founder population (some survive on luck and seed selection) and the
  frozen-net control below.
- **P2 — the evolved valence map diverges into individuality.** Across generations, the
  census of interoception→`M` genes should spread and drift *directionally* (toward whatever
  keeps agents fed/whole), not stay at founder diversity. This is beta_07's individuality
  finding, but now the thing carrying the identity is an explicit, inspectable gene vector.
- **P3 — plasticity earns its keep (the frozen-net control).** Run an arm with plasticity
  off (`alpha: 0`) — a pure evolved reflex agent, `W_fast` frozen at zero. If the plastic
  arm doesn't out-forage the frozen arm, within-life learning is contributing nothing and
  all adaptation is genetic — itself a strong, publishable result about this world.
- **P4 — Darwinian vs Lamarckian.** The `genome` (reinit fast weights) and `lineage` (carry
  fast weights) arms should diverge: Lamarckian should ratchet competence faster early
  (inherited training) but may homogenise; Darwinian isolates *evolved innate* competence.
  The gap between them is the cultural-transmission signal (Q4), cleanly separated because
  genome and learned weights are separate tensors.
- **vs beta (the headline comparison).** Does a backprop-free, world-model-free brain reach
  a foraging competence in the same ballpark as beta's Dreamer, at a fraction of the compute
  and with a much larger population? Either answer is a finding: parity would be a strong
  claim about how little machinery embodied survival needs; a large gap quantifies what the
  world model buys.
- **Population dynamics as a result in themselves.** With 24 cheap agents in one world,
  crowding, competition over bushes, and mutual perceptual salience ("alive" ray-kind) may
  produce dynamics the 3-dreamer runs never could. Any shape — stable, oscillating,
  clustering — is a finding.

## What this round can and cannot claim

- **Can:** whether felt-but-not-planned-toward valence can sustain a life (P1); whether an
  evolved valence map produces inspectable individuality (P2); whether plasticity beats a
  frozen reflex (P3); the Darwinian/Lamarckian gap (P4); a compute-matched competence
  comparison against beta.
- **Cannot (needs sequels):** anything about self-modelling or introspection (round B, where
  the self-prediction head and its novelty modulator arrive); dreaming / consolidation
  (round C); aging as an evolved trait (round D); and the interiority-vs-evolution question
  that the parallel **active-inference** track is meant to answer.
- **Inherited caveats:** single run per arm (006 measured ~40% forager variance between
  identical-config runs — trust trajectory shapes and directional trends, not fine levels);
  the temperament↔weights confound is the same one 001/007 carry, here made addressable by
  the genome/lineage split.

## The sequel arc (named, not built here)

- **B — self-model / introspection.** A head predicting the agent's own next state; its
  error is a felt surprise about oneself and the second modulator — the curiosity this
  architecture can compute without a world model.
- **C — dreaming / offline consolidation.** Replay eligibility traces through the plastic
  rule during night dormancy. Cheap here; a genuine memory-consolidation phase to instrument.
- **D — evolved aging / plastic senescence.** `α` decays with age; the young learn, the old
  exploit, and knowledge must transmit (B/C) or die with the body — the engine behind rearing
  and culture, and the tie-in to proposal 001's senescence sequel.
- **Parallel: the active-inference track** (its own prefix, `kind: active_inference`). Richer
  for pure interiority — free-energy, preference, self-model as native math — run *alongside*
  anima, not after it, so the two architectures can be compared on the same world.

## Invariant check

One persistent world, no episodes/reset (✓). No designer task or fitness function — `M` is
homeostatic valence (an allowed intrinsic drive) that *gates plasticity*; nothing performs
`argmax` over expected valence, and the valence map is evolved, not authored (✓). World +
brains still checkpoint together — genome + `W_slow` + `W_fast` serialise with the brain,
world state as today (✓). Sim never waits on learning — plasticity is an in-`act` local
update, the learner thread is never scheduled for this family (✓). Obs/action contract
unchanged, OBS_VERSION stays 3, so anima↔beta comparisons are valid (✓).
