---
proposal: 001
title: budding — endogenous reproduction as conservation, with a maturation ramp
date: 2026-07-06
status: proposed
targets_round: 009           # the first round that would run this; beta_08 must close first
question: If well-fed dreamers bud offspring — funded by conserved energy surplus, inheriting a mutated parent — does a population under no fitness function develop selection on temperament and a cross-generational ratchet in foraging competence?
depends_on: [008]            # needs beta_08's converging-model result; reuses its base/cuda/paced setup
arc: budding (A, this doc) → evolvable lifespan + senescence (B) → provisioning / nurture (C)
tags: [evolution, reproduction, lifecycle, temperament, cultural-transmission, ecology]
---

# 001 — budding

The step between what we have and full sexual reproduction. Today's respawn is
*reincarnation*: a fixed budget tops the population back up to `target`, and (under
`inherit_weights: lineage`) a dead mind waits in a stash for the next free body. Birth is
experimenter machinery — a queue, a spawn pad, a delay. This proposal makes birth a
**bodily process**: a dreamer that carries an energy surplus long enough *buds* a child
beside it, paying for the child out of that surplus, passing on its weights and a mutated
temperament. No reproduce reward, no reproduce drive, no obs change. Selection acts only
through who manages to stay in surplus — which is who forages and survives well — so it is
the opposite of a fitness function.

This is "proposal 001" in the evolutionary-lifecycle arc. It deliberately does the least
that still produces real selection, and names its two sequels (B, C) rather than bundling
them.

## Why this, why now

- **It converts beta_07's strongest finding from artifact to biology.** The one clearly
  life-like result to date was heredity: three lineages held distinct, persistent
  behavioral personalities across ~11 respawns each (007, H4). But that was reincarnation —
  the same minds cycling. Budding makes lineages actually *branch and compete*.
- **It needs no new competence from the agents.** Communication needs something worth
  saying; agriculture needs delayed-payoff credit assignment; both are beyond a dreamer
  eating ~0.2–0.5 meals/day. Evolution needs only differential survival, which the world
  already produces. It is the cheapest true life-like property available.
- **It is the only lever that can break the five-round competence wall.** Rounds 004–008
  each hinged on *one* brain's learning speed. Selection over an inheriting population lets
  foraging competence *ratchet across generations* instead of resetting the question every
  round.
- **It sequences cleanly after beta_08.** beta_08 is establishing that the model can
  converge (P1 landing: first nonzero boredom, curiosity going stale). Budding reuses that
  round's base/cuda/paced setup wholesale and moves exactly one concept — how bodies are
  born.

## The mechanism

### Reproduction = conservation (no threshold, no timer)

Repair already spends surplus energy: energy above `economy.repair_threshold` (60) funds
integrity repair at `repair_energy_per_point`, and *never drains the body below that line*.
Budding is the same idea pointed at a new body instead of patching the old one:

1. **Repair has first claim.** Surplus above `repair_threshold` funds repair first (as
   today). "In good health" is therefore not a new knob — a damaged body pays down its own
   integrity and buds nothing until it is whole again.
2. **Remaining surplus fills a gestation reserve.** Whatever surplus is left after repair
   trickles into a per-robot `gestation` reserve, skimming the parent's energy back down
   *toward* `repair_threshold` but never below it. Gestating makes you run leaner on
   buffer, never starves you outright — the selective cost lives in lost buffer and in the
   child competing for your bushes, not in engineered parental death spirals.
3. **Reserve full → birth.** When the reserve reaches the child's build cost
   (`economy.energy_max`, 100 — a newborn starts full, as now), a child is instantiated
   *next to the parent* with `energy = reserve`, and the reserve zeroes. Conservation is
   exact: ~100 units of surplus energy left the parent gradually over its life and became
   the child's starting body.

Crucially, **reward is never touched.** No term rewards reproduction, and `gestation` is
*not* added to proprio — OBS_VERSION stays 3, so beta_07/08 comparisons survive. A dreamer
never decides to reproduce and is never paid for it. Birth is simply what a well-fed body
does, felt only through consequences: your surplus quietly went somewhere, and a robot
appeared beside you.

### Honest accounting of the knobs

The *who-gets-to-reproduce* boundary introduces **no new number** — it is conservation
against two lines that already exist (`repair_threshold`, `energy_max`) with repair given
priority. But three genuinely new numbers appear, and pretending otherwise would be the
kind of config fiction round 008 caught:

| new knob | what it sets | analog | fixed in round A? |
|---|---|---|---|
| `gestation_efficiency` | reserve gained per unit surplus routed → the *timescale* of reproduction (effective parental cost = `energy_max / efficiency`) | `repair_energy_per_point` | constant |
| `cap` | max simultaneous dreamers; births suppress above it | — (hardware carrying capacity) | constant, turnable as GPUs grow |
| maturation ramp: `birth_actuation_floor` + `maturation_ticks` | how weak/slow a newborn is and how long until prime | `brownout_floor` / `brownout_threshold` | constant (becomes a *gene* in round B) |

`gestation_efficiency` is a **rate**, not a decision boundary — it tunes how many meals'
worth of surplus a child costs, not who is allowed one. The cap is explicitly the
hardware's carrying capacity, not an ecological claim.

### Maturation ramp (bundled, but separable)

A budded child spawns physically **weak**: its `BodySpec.max_speed`/`max_turn` start at
`birth_actuation_floor` (≈0.4) of adult values and ramp linearly to 1.0 over
`maturation_ticks` (~1 sim-day). Pure physiology — a multiplier on actuation exactly like
brownout, no reward or obs involvement.

It is bundled into round A because it is what makes a bud a *vulnerable* bud: it gives
birth a real cost, creates a survival filter on the young (so budding can't trivially peg
the cap), and lays the helplessness gradient that provisioning (round C) would later act
on. **But it is the one piece that can destabilize the round** (see P4). If the scripted
calibration pass shows it is fiddly, budding runs flat-bodied first and the ramp becomes
round A.2. Elder decline / senescence is explicitly *not* here — it needs an evolved
longevity tradeoff to be interesting (round B).

Foragers do **not** get the ramp and do **not** bud. They keep the fixed respawn budget and
byte-identical bodies, so `forager eats/10k` stays the cross-round anchor it has been in
every journal entry.

### The extinction floor (death stays meaningful)

- Dreamers above the floor die for real — no stash reincarnation. The population is
  genuinely lost members plus budded newcomers.
- When living dreamers drop **below the starting count (3)**, the floor engages: the
  delayed respawn spawns a **mutated copy of a random *living* dreamer** (survivor budding —
  `random_living` semantics + temperament mutation via `Brain.inherit`). A lineage that
  actually died out is *not* resurrected from a stash — that would quietly undo the
  selection we are trying to observe.
- Only at **total dreamer extinction** does a fresh brain enter, so the world can't die.

Both birth and floor reuse the existing `Brain.inherit` path (`base.py:55`), which already
copies weights and mutates temperament. Every new dreamer — budded or floor-spawned — is a
mutated copy of a living one; the population is always evolving, never reincarnating.

## Config surface (illustrative — calibrated in the scripted pass, not final)

New world-economy block (birth is world physiology, like repair):

```yaml
# configs/world/default.yaml  (economy)
reproduction:
  enabled: true
  learners_only: true          # dreamers bud; scripted bots keep the respawn budget
  # surplus line = economy.repair_threshold (60); child cost = economy.energy_max (100) — reused
  gestation_efficiency: 0.5    # reserve per unit surplus routed (after repair's claim)
  cap: 8                       # max simultaneous dreamers; births suppress above (hardware knob)
  maturation_ticks: 24000      # birth → full actuation (~1 sim-day)
  birth_actuation_floor: 0.4   # max_speed/max_turn fraction at birth, ramps to 1.0
```

Population config (floor policy for learners):

```yaml
# configs/run/beta_09_*.yaml  (population)
population:
  target: 6                    # floor only: 3 dreamers + 3 foragers, as beta_08
  inherit_weights: random_living   # floor spawns copy a *living* dreamer; fresh only at extinction
```

## Integration points (where the code changes)

- **`entities.py`** — add `gestation: float = 0.0` to `Robot` (checkpointed via
  `to_dict`/`from_dict`). Add an age→actuation multiplier read from `age_ticks` +
  `maturation_ticks`, applied where `max_speed`/`max_turn` are consumed by physics.
- **`world.py`** — in the per-tick economy (beside repair), route post-repair surplus into
  `gestation`; when it hits `energy_max`, emit a **`birth` intent** (parent id) and zero the
  reserve. Add `spawn_near(parent_pos)` (reuse `find_spawn`'s validity checks around the
  parent; fall back to `find_spawn`). Emit a `birth` event (parent, child, tick, pos).
- **`scheduler.py`** — consume `birth` intents in/near `_process_lifecycle`
  (`scheduler.py:125`): construct the child brain via `Brain.inherit(parent.state_dict())`,
  place it with `spawn_near`, respect `cap`. Make the floor path prefer a *living* dreamer
  donor (already the `random_living` branch, `scheduler.py:164`) and drop lineage-stash
  reincarnation for learners in this round.
- **`config.py`** — `ReproductionConfig` dataclass; wire into the economy config.
- **`gol-stats` / metrics** — per-generation temperament census, births/deaths ledger,
  budded-child survival-to-prime fraction, population time-series, generation depth.

## Calibration first (the M2 pattern)

The M2 economy was calibrated on scripted foragers before learners ran against it; do the
same here. Foragers forage reliably, so temporarily letting *them* bud is the clean probe
for the world dynamics with the brain factored out. Run scripted-forager budding to fix
`gestation_efficiency`, `cap`, and the maturation ramp so that (a) births actually occur,
(b) the population is stable rather than crashing or pegging the cap, and (c) budded young
survive to prime at a nonzero rate. **Lock those numbers, then run the real dreamers-only
config.** This separates "the reproduction economy is miscalibrated" from "dreamers aren't
competent enough to reproduce" — a distinction rounds 002/006 show is easy to lose.

## Predictions (pre-registered — including the failure branches)

- **P1 — does the mechanism fire at all?** *First thing to check.* At beta_08 competence
  (~0.2–0.5 eats/day, ~12% awake), a dreamer may **never** hold enough post-repair surplus
  to fill a reserve — births ≈ 0, the population runs entirely on the floor, and we've
  reinvented reincarnation. This ties reproduction directly to the competence beta_08 is
  establishing; if beta_08's P2 (hunger airtime, better eating) doesn't land, budding may be
  inert and that is a clean, informative null.
- **P2 — selection on temperament.** If births happen, the census-weighted temperament
  distribution (`w_curiosity`, `w_homeostasis`, per-drive weights) should shift
  *directionally* over generations toward whatever forages/survives, beyond the neutral
  drift of floor-only spawning. Null: drift only = selection too weak at this
  population/birth rate.
- **P3 — the competence ratchet (the prize, and the confound).** Forage competence should
  *rise across generations* — the wall five rounds couldn't clear. **Confound, stated up
  front:** under weight inheritance a rise can't be cleanly attributed to selection vs.
  simply more effective training carried forward through inherited weights — the same
  confound as 007's H4. Splitting it needs the `random_living` (temperament-only reset of
  weights) vs. `lineage` contrast; named as the round A.2 ablation, not claimed here.
- **P4 — the maturation ramp doesn't self-defeat.** Weak young must survive to prime at a
  nonzero rate; if they die on contact, the floor fires constantly and we're back to
  reincarnation. Calibrated on scripted foragers first. Also verify the ramp leaves the
  forager anchor untouched (it should — foragers don't get it).
- **Population dynamics as a result in themselves.** Stable, oscillating (predator-prey-ish
  against the bush economy), crash, or cap-pegged — any outcome is a finding. Malthusian /
  Lotka–Volterra-shaped dynamics *with no fitness function anywhere* would be the
  headline-worthy version.

## What this round can and cannot claim

- **Can:** whether endogenous reproduction is even reachable at current competence (P1);
  the shape of an unmanaged population's dynamics; whether temperament selection is
  detectable.
- **Cannot (needs sequels):** clean attribution of any competence ratchet to selection vs.
  inherited-weight training (needs the `random_living`/`lineage` split); anything about
  longevity as an evolved trait (round B); anything about care/provisioning (round C).
- **Inherited caveats:** single run (006 measured 40% forager variance between
  identical-config runs — trust trajectory shapes and directional trends, not fine levels);
  the temperament↔weights confound travels with lineage inheritance exactly as in 007.

## The sequel arc (named, not built here)

- **B — evolvable lifespan + senescence.** Make `maturation_ticks` and a new max-lifespan a
  *gene* (temperament genome, per the roadmap's genome-primary decision), add elder
  actuation decline and death-by-age — but only with a two-sided longevity tradeoff, or the
  gene just pins to the cap and there's nothing to watch.
- **C — provisioning / nurture.** Weak young + the existing feed verb *could* let care
  emerge with no care drive. High risk: with only curiosity + homeostasis there is **no
  reward path for an adult to feed a child** (the honest hook is that babies are novel
  moving objects → curiosity magnets → maybe feeding gets stumbled into and transmitted).
  Absence of care is a valid finding; the standing temptation to add a care drive to force
  it is exactly the shaping drift the invariants forbid. Runs only after A shows whether kin
  even share space.

## Invariant check

One persistent world, no episodes/reset (✓). No designer task or fitness function — birth
is conservation-driven physiology, reward untouched (✓). World + brains still checkpoint
together (`gestation` and `age_ticks` serialize with the robot; brains via
`Brain.inherit`) (✓). Sim never waits on learning — birth is a world-tick event,
independent of the learner (✓). Obs/action contract unchanged, OBS_VERSION stays 3 (✓).
