---
proposal: 004
title: finitude — make time scarce and continuation earned, so a survival instinct evolves instead of being taught
date: 2026-07-08
track: anima (world/body change; applies to beta too, but anima is the substrate)
status: proposed
targets_round: anima_02
question: The hibernation attractor is architecture-independent — beta (world-model) and anima (plastic/Hebbian) both converge on solar-powered dormancy — because in this world wasting a whole life costs nothing. Body-neglect death is ~333k ticks away and imperceptibly slow, and the respawn timer tops up hibernators and foragers equally, so nothing selects for living over merely surviving. Can making time FINITE and FELT (sharpened senescence + an interoceptive age channel) and continuation EARNED (endogenous reproduction gated on a thriving body, replacing the respawn timer) cause a survival instinct — fighting to live, not sleeping life away — to *evolve*, with no survival reward and no foraging task?
depends_on: [012, 002]           # builds on anima's first results + the mortality reframe
integrates: [001]                # endogenous reproduction is proposal 001's budding, now load-bearing
arc: mortality/viability (003) → felt finitude + earned reproduction (this doc) → evolved aging as a trait (002 round D)
tags: [mortality, finitude, senescence, aging, reproduction, natural-selection, time-awareness, evolution, emergence, obs-version]
---

# 004 — finitude

## The reframe: survival instinct is *evolved*, not taught — and we removed both things that evolve it

Six rounds of beta and the first anima round all fail the same way, and it is not a brain
failure. Both a world-model planner (Dreamer) and a backprop-free Hebbian gate-learner
(anima) independently converge on the **same** solution: crash energy, go dormant, let the
sun trickle you back up, repeat, and slowly wear out around the ~333k hibernation clock. When
two architectures with nothing in common find the identical strategy, the strategy is in the
**world**, not the mind. No smarter brain and no better reward equation escapes it, because
they would all still notice that napping in the sun works.

**A correction to the earlier "the sun feeds the sleeping" read.** The world is not actually
giving away survival for free. Solar only trickles a dormant body to ~40 energy (`wake_energy`),
but *repairing* the body needs energy above 60 (`repair_threshold`) and costs energy to do
(`repair_energy_per_point`). So a pure hibernator can never repair — it just wears down and
dies. **Eating is already required for a long life.** The world already encodes the user's
intuition: sleep recharges you, but you still need food to keep your body from failing.

So the real defect is sharper, and it is exactly the thing intuition names: **finitude is
neither felt nor consequential.**

1. **Time is not scarce enough to matter.** The body bleeds 0.0003 integrity/dormant-tick —
   imperceptible — and the lethal consequence is ~333,000 ticks (≈70k act-steps) after the
   neglect that causes it. No credit-assignment mechanism — Dreamer's 15-step imagination,
   anima's ~20-step eligibility trace — can bridge that gap. You cannot fear wasting time you
   cannot feel pass, and you cannot learn from a punishment that arrives a lifetime late.
2. **Dying costs the lineage nothing.** In nature, fear of death is not learned in one life;
   it is *bred in* over generations — the animals that did not fight to live left no
   descendants. Our world has no such filter: the respawn timer (`respawn_delay_ticks`) tops
   the population back up regardless of how anyone lived. A hibernator's lineage continues
   exactly as well as a forager's. **No differential reproduction ⇒ no selection for a
   survival instinct.** We have been trying to install in one brain what a billion years of
   dying installed across lineages.

Anima's own data is the tell: valence genes drifted directionally only when they *did*
something (viability_gain +21% in the plastic arm, flat in the frozen null) — so selection
*is* live and responsive here. It just has nothing to grip, because living longer buys no
extra descendants. Give it something to grip.

## The bet: don't teach mortality, make it matter — then let evolution do what it does

Two additions, and the charter test each must pass is the same: **no survival reward, no
foraging task, nothing that tells a brain what to want.** These are properties of the *world*
and the *body*, plus *natural selection* — all explicitly allowed (invariant 2 forbids
designer tasks and fitness functions, not mortality). The brain is never told to survive; we
change what survival *costs* and what continuation *requires*, and let the instinct emerge.

### A. Finite, felt lifespan (make time a scarce, perceivable resource)

The world already has soft senescence (`senescence_halflife: 150000` — repair efficiency
halves every 150k age-ticks). Sharpen it into a genuine finitude and, crucially, **let the
agent perceive it**:

- **A hard-enough lifespan.** Lower `senescence_halflife` (candidate ~80–100k) and/or add an
  age term to baseline wear, so repair efficiency decays toward zero and even a perfectly-fed
  body eventually loses the wear race — death of *old age*, not just neglect. This makes a
  maximum practical lifespan emerge (target the ~500–800k range: long enough to live richly,
  short enough that time is scarce). Death still comes from the body failing — no designer
  clock, no `age > N ⇒ die` rule (that would be a task-shaped cutoff). Charter-clean.
- **Perceivable finitude (the OBS_VERSION 4 decision).** Add one interoceptive channel to
  `proprio`: the body's **senescence state** (normalized age, or equivalently current repair
  efficiency 1→0). This is the literal substrate for "time awareness" — the agent can *feel*
  its life running down, the way it already feels hunger. We never tell it what the channel
  *means*; evolution and learning discover that a high value near an unrepaired body is bad.
  This bumps OBS_VERSION 3→4 (invariant 6, a deliberate versioned decision) and old brain
  checkpoints stop loading — acceptable, since anima_02 is a fresh founder population anyway.

### B. Earned endogenous reproduction (make continuation cost a life well-lived)

Replace the respawn *timer* with **budding gated on a thriving body** (proposal 001's
mechanism, now load-bearing): an agent reproduces when it has *sustained physiological
surplus* — energy held well above the repair threshold for some window, an intact body, past
a minimum age — by spending a chunk of its own energy/integrity to spawn a child that inherits
its mutated genome (and `W_fast` per `inherit_mode`). No fitness function: budding is a
physiological event gated on body state, exactly like real animals, not a score being
maximized. The consequence is the whole point: a hibernator never accumulates the surplus to
bud, so its lineage dwindles; a forager that thrives buds, and its disposition to *live*
propagates. Differential reproduction is the selection pressure that breeds in the instinct.

### C. Curiosity is the "something to live for" (already present)

No change. Being awake and exploring earns learning-progress reward; dormancy is a blank.
Finite, felt time + curiosity = *"I want to experience the limited life I have."* We do not
add this motive — we make time scarce enough that forgone experience finally bites.

## Why anima is the right substrate

This shifts the load from within-life learning onto *evolution*, which needs many fast
generations and a large population to move — precisely anima's regime (cheap Hebbian brains,
48+ agents, ~0.8 ms/act) and precisely what beta's 3 dreamers never had. beta can run the
same world later for a within-life-learning comparison, but the emergence bet lives on anima.

## Predictions (pre-register at launch)

- **P1 — the hibernation attractor loses its subsidy.** Under earned reproduction, the dormant
  fraction of *surviving lineages* falls over generations (not within a life) as hibernator
  lines fail to bud and forager lines take over. Founder-generation dormancy may stay high;
  the signal is the *descendant* generations.
- **P2 — foraging rises across generations, not within a life.** eats/life and the
  sated:hungry ratio improve generation-over-generation as lineages that eat-to-thrive
  out-reproduce. A flat trajectory is the informative null: even earned reproduction can't
  select foraging if the skill is never stumbled into (→ affordances/skill layer next).
- **P3 — the age channel acquires meaning.** Gene drift should favor lineages whose behaviour
  is *conditioned* on the senescence channel (measurable as: do young vs old bodies behave
  differently?). If the channel stays behaviourally inert, felt finitude isn't being used and
  perception alone wasn't enough (→ the pressure has to come from B, not A).
- **P4 — lifespans separate by strategy.** Thriving forager lineages live long and bud;
  hibernator lineages die on the (now shorter) senescence clock without budding. A widening
  gap between lineage-mean lifespans is the selection signal itself.
- **Falsification branch:** if reproduction is earned and time is finite/felt but foraging
  still never emerges, the binding constraint is the *skill* of foraging (multi-step approach
  under sparse food), and an option/skill layer or a denser food geometry moves up — not more
  mortality machinery.

## Invariant check

One persistent world, no episodes/reset — budding and senescence are in-world events, deaths
stay unexperienced (✓). No designer task or fitness function — no survival reward, no forage
directive; finitude is a body property (the body wears out), reproduction is a physiological
event gated on state, and the only optimizer is natural selection, which the charter permits
(✓). World + brains checkpoint together — senescence state serialises with the robot, genome
with the brain (✓). Sim never waits on learning — unchanged; budding is a scheduler event
(✓). **OBS_VERSION 3→4** — a deliberate versioned decision (the senescence proprio channel);
old brain checkpoints stop loading, acceptable for a fresh anima_02 founder population (✓,
with the bump explicitly acknowledged).

## The forks — settled 2026-07-08

1. **Aging → sharpen senescence.** Lower `senescence_halflife` (and, if needed, an age term on
   baseline wear) so repair efficiency decays toward zero and even a well-fed body eventually
   loses the wear race — death of old age *emerges from the body failing*, no `age > N ⇒ die`
   rule. Charter-cleanest.
2. **Reproduction → budding + low respawn floor.** Earned budding drives the normal population;
   a respawn floor fires only if population crashes below a low minimum (extinction guard), so
   budding dominates selection when it works but run 1 can't die outright. Set the floor low so
   it rarely engages.
3. **Perception → bump to OBS_VERSION 4 now, add the senescence channel.** It is the literal
   "time awareness" substrate the hypothesis is about, and anima_02 is a fresh founder
   population so the contract bump costs nothing. Tests the full mechanism; later ablations
   isolate the three levers.
