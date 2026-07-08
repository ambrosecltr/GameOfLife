---
proposal: 003
title: mortality — why viable life must out-value cessation, and a viability drive that makes it so
date: 2026-07-08
status: proposed
targets_round: 012 (two arms — 012a base/cloud, 012b nano/local)
question: The five-round competence wall is not capacity, conditioning, or reachability — it is that in the current reward geometry a mortal life integrates to a *negative* return while cessation is worth exactly zero, so a converged agent correctly prefers to stop. Can a non-telescoping viability drive (distance-to-lethal-boundary, not distance-to-comfort-setpoint), boredom fully decoupled onto its own channel, and a represented terminal value for true death, make self-preservation and a functional sense of mortality *emerge* — without a survival task or fitness function?
depends_on: [011]
arc: mortality/viability (A, this doc) → sense-of-self / interoceptive self-model (B) → evolved valence & aging (C, overlaps proposal 002 anima)
tags: [homeostasis, survival, mortality, reward-geometry, viability, boredom-decoupling, drive, nano, affordances]
---

# 003 — mortality

## The reframe: the beta arc has been fighting the reward's sign, not its reachability

Rounds 004–011 each named a different binding constraint — curiosity collapse (004),
capacity (008), signal conditioning (009), reward reachability (011) — fixed it cleanly,
verified the fix worked mechanically, and watched behavior not move. Four exonerations of
four hypotheses is not four failures. It is evidence that the constraint is upstream of all
of them, in the **geometry of the reward itself**. Here is the claim, stated as arithmetic
over the code we already run.

The homeostatic reward per step (`brain.py:595`) is

```
r_homeo(t) = drive_scale · (d(t-1) − d(t)) − level_penalty · d(t)
```

where `d(t)` is the Keramati–Gutkin drive (convex deficit from setpoints, `_drive_level`).
Sum it over one whole life, birth to death:

```
Σ r_homeo = drive_scale · (d_birth − d_death) − level_penalty · Σ d(t)
```

The reduction term **telescopes**: all the intermediate meals cancel against the decay that
made them necessary, leaving only endpoints. Newborns spawn full (`d_birth` small); mortal
lives end hungry or hurt (`d_death` large). So `(d_birth − d_death) < 0`. The second term is
strictly ≤ 0 always. **Every mortal life earns negative cumulative homeostatic reward, and
the more of it the agent lives, the more negative it gets.** Eating is not net-positive; it
is loss-mitigation on a debt the body keeps re-incurring.

The only strictly positive term in the whole reward is curiosity (`r_cur`, `brain.py:652`).
Boredom (`brain.py:673`) is a *penalty*, i.e. more negative reward.

**This is checkable now, and it checks out — with one correction to the clean story.** On
beta_10's logs, mean `reward_homeostasis` is negative in every 400k window (−0.0043 →
−0.0065, trending *more* negative over the run): the homeostatic stream is a net drag on
being alive, confirmed empirically and by the telescoping arithmetic above. But the naive
conclusion — "so value goes below the 0 of dormancy and the critic turns suicidal" — is
**not** what the data shows, and honesty demands the amendment. Median critic `value` is
~212 and, tellingly, **identical whether the agent is dormant or awake** (212 vs 212 across
70k dormant and 9k awake samples). Value stays positive because curiosity, under 009's
anchoring, did *not* collapse in beta_10 — `curiosity_scaled` rose 0.20 → 0.93 across the
run. (The "curiosity → 0" collapse was beta_08's unanchored regime; anchoring fixed it, as
intended.)

So the real geometry is subtler and, if anything, a cleaner target: the agents are not
suicidal, they are **indifferent to their own aliveness**. Homeostasis supplies *no positive
reason to be awake* and a small standing reason not to be; curiosity — an exploration signal
that is earned in imagination and pays the same whether the body lives or sleeps — is the
only thing holding value up, and it rates dormancy exactly as good as activity. The
hibernation attractor is not "death beats life"; it is "sleep is free and being awake costs,
and nothing values the difference." Martin, Everitt & Hutter (2016), *Death and Suicide in
Universal AI*, still supplies the governing principle — cessation sits at an implicit reward
of 0, and self-preservation requires the *lived* stream to clear that bar by a margin — but
our regime clears it only via a survival-agnostic curiosity term, while the one drive that
is *about* the body clears it by a **negative** margin. There is no signal in the system for
which "this body, alive and far from the boundary" is worth more than "this body asleep, or
gone." That absence is what proposal 003 adds.

This is not a bug in reachability or capacity — those were all real and all fixed. It is that
the reward geometry gives a mind no positive stake in its own continued viability. The
replay-subsidy census (011) found one artifact *paying* the attractor; this is why it was an
attractor for the artifact to feed.

**Testable, falsifiable form (partly pre-confirmed):** cumulative homeostatic return over a
dreamer life is negative and worsens with lifespan (✓ on beta_10); critic value is
insensitive to dormancy-vs-activity (✓, 212≈212); both should reverse under the intervention
below — value should become *strictly higher* for viable-and-active than for dormant. The
remaining pre-work is to confirm the sign and the indifference replicate on beta_08/09 and to
measure the per-life cumulative directly (see Pre-work).

## What the literature offers, and where we go past it

- **HRRL (Keramati–Gutkin; Laurençon et al. continuous HRRL, 2021/2024)** gives us
  need-relative valence and we keep it — a starving agent's meal *should* outweigh a snack.
  What HRRL never claims is that drive-reduction alone yields self-preservation; it assumes
  the value function integrates anticipation, and in an *episodic* benchmark the negative
  telescoping never bites because lives are reset before the debt compounds. We run
  non-episodic (invariant #1), so we see what episodic HRRL structurally cannot.
- **Death & Suicide in Universal AI (Martin/Everitt/Hutter 2016)** gives the sign law above.
  Its lesson for us: you cannot get self-preservation by *reaching* a reward better; you get
  it by making the lived stream net-positive relative to the 0 of cessation — an additive,
  geometric property, not a density property.
- **Active inference / FEP (Friston; Da Costa et al. 2020)** frames survival as
  self-evidencing: acting to maximise evidence for one's own continued existence, with a
  *viability boundary* encoded as prior preference. This is the missing primitive — beta has
  a comfort setpoint but no represented boundary of annihilation. Our contribution is to add
  a boundary term inside a world-model/actor-critic agent (not a full active-inference
  rewrite — that is proposal 002's neighbouring track), and to test whether mortality-salience
  emerges from it.

We were explicitly invited to invent here rather than replicate. The core new idea is small
and, as far as the above shows, un-tried in a persistent world-model agent:

## The one new mechanism: a viability drive (boundary-distance, not setpoint-distance)

Replace *nothing*; **add** a second homeostatic term with a deliberately different geometry.
Keramati–Gutkin drive `d` stays — it is comfort-seeking, pulling toward the 0.85/1.0/1.0
setpoints. Add a **viability potential** `V` that is annihilation-*avoiding*, a log-barrier
on the distance from each survival-critical variable to its lethal floor:

```
V(t) = Σ_i  w_i · ( −log( (x_i(t) − lethal_i) / (safe_i − lethal_i) ) )₊     # 0 above safe_i, ↑∞ toward lethal_i
r_via(t) = viability_scale · ( V(t-1) − V(t) )   −   viability_floor · V(t)
```

for the survival variables (energy, integrity), with `lethal_i` the death floor and `safe_i`
a comfortable margin above it. Three properties, each targeting a specific failure:

1. **It does not telescope to a loss.** Far from the boundary `V ≈ 0`; the return of a life
   spent *staying* far from death is bounded near zero from below and dominated by the
   reduction spikes when you claw back from an excursion — the integral of viable living is
   ≈ 0⁺, not the strongly-negative of the current drive. Combined with any residual positive
   curiosity, the lived stream clears the 0 of cessation. This is the sign-fix the death
   theorem demands, achieved through geometry rather than a bolted-on "+1 alive" (which would
   be a survival task — invariant #2).
2. **The marginal value of a calorie explodes as you starve.** `−log` → ∞ near the floor, so
   eating-while-hungry is worth *unboundedly more* than eating-while-full — a far steeper
   asymmetry than the convex setpoint drive gives. This is exactly the "eat to survive ≫ eat
   for fun" gradient round 011 measured the absence of (104 sated meals : 1 hungry). And it is
   dense and low-frequency — no reliance on the twohot head catching rare spikes, so it is
   *cheaper for a nano brain* than sparse-meal reachability was.
3. **It is not a task.** It rewards no action and names no behaviour; it is a pure function of
   internal state, like every other drive. Moving away from the boundary by any means —
   eating, resting, being fed by a peer — relieves it. Charter-clean.

`viability_scale = 0` recovers beta_10 exactly (ablation switch). The whole proposal is one
config-flagged additive term plus the two decouplings below.

## Decoupling boredom from hunger (the user's structural concern, made concrete)

Today the two are coupled twice: (a) summed into one scalar the actor maximises, and (b) the
boredom gate literally reads drive — `calm = (1 − drive/thr)₊` (`brain.py:659`), the
"an agent in need is never bored" rule. The user's point stands: that rule bakes in the
assumption that survival always dominates, yet behaviourally boredom is the *only* drive that
ever produced foraging, and one can absolutely eat out of boredom. The coupling is why we can
never tell "ate to live" from "ate to pass the time" — and 011 says it's almost always the
latter.

Decouple on two axes, both flagged:

- **Gate boredom on viability, not on comfort-drive.** `calm` should read "am I safe from
  annihilation" (`1 − V/thr`), not "am I at my comfort setpoint." An agent can then be
  simultaneously well-away-from-death *and* bored — which is the honest state most of a fed
  life is in, and the state in which boredom-foraging is legitimately just play. Hunger
  (approaching the boundary) still shuts the gate, but *comfort deficit* no longer does.
- **Report the two channels separately end-to-end** (they are already summed only at the last
  line of `_imagination_reward`). Log `r_via`, `r_homeo`, `r_cur`, `bored` as distinct return
  contributions along real trajectories so we can finally read *which drive paid for each
  eat*. This is the decision-forensics arm 011's Next asked for, and it is a prerequisite for
  believing any positive result: "eating rose" means nothing until we can attribute it to
  `r_via` rather than `bored`.

## Mortality proper: give true death a represented terminal value

Beta_10 made the dormancy blackout *priced but non-terminal* (`cont → 1`) so imagination
could plan across the recoverable crash. Correct for reachability, but it also erased the one
mortality signal in the system, and the death theorem says an agent that never represents an
end-state cannot value avoiding it. Split the two events that beta currently blurs:

- **Recoverable dormancy** (energy floor, solar-recoverable): stays priced, stays
  `cont → 1`. The viability barrier already makes *approaching* it steeply aversive, which is
  the anticipatory signal we want; the agent should fear the slope, not the sleep.
- **True death** (integrity → 0, the permanent one): should terminate the imagined stream
  (`cont → 0` at the lethal integrity floor) so its value backs up through the critic as the
  absorbing 0-return state the theorem requires. Death stays *unexperienced* in the real
  stream (invariant — no episodes), but it becomes *imaginable*: the world model, having seen
  integrity decay, can roll forward into the boundary in the dream, and the barrier + terminal
  cont together give that rollout a value strictly below any viable trajectory. That is a
  functional fear of death that emerges from prediction, not from having died. This is the
  narrow, load-bearing sense in which a "sense of mortality" is buildable here.

(A dormancy-duration or "time-to-lethal" interoceptive channel would sharpen this but is an
OBS_VERSION 4 decision, deferred exactly as in 011. The barrier reads only existing proprio.)

## The two arms (012a / 012b) and the affordance question

The user's arm split doubles as the capacity control the whole beta line has wanted:

- **012a — base preset, cloud, paced.** The clean test of the mechanism at a capacity where
  the world model is known to converge (008). If mortality-salience emerges anywhere, here.
- **012b — nano preset, local M1.** The real question: *is eat-to-survive a basic function
  that shouldn't need a large brain?* The barrier drive is designed to make this winnable —
  dense, low-frequency, no sparse-spike head dependence — but 008's hard finding stands: nano
  never converged the obs-v3 model, and if the world model can't predict the approach to the
  boundary, imagination can't fear it. So 012b must be **paired with the affordance change**
  (below) and judged on a *converged-enough* criterion, not against 012a's absolute behaviour.
  A null on 012b with a win on 012a is itself a clean, publishable result: survival-salience
  has a capacity floor. A win on 012b is the bigger prize — it frees base brains to host
  communication/social emergence on top of a survival substrate that already works.

**Affordances (raise food production).** 011 found scripted foragers — a *perfect* policy —
starving in generation-scale swings on spawn luck alone (intake 4–8× between forager
generations with 300+ ripe bushes standing). That means meal geometry is patchy enough to gate
*any* policy, which would confound a survival-drive test: an agent that correctly wants to eat
but can't find food looks identical to one that doesn't want to. So raise bush
density/regrowth (candidate ~2×) as a **precondition**, not a treatment — enough that a
competent policy can reliably stay fed, so that failure to stay fed is attributable to the
mind and not the map. Titrate back down in a later round once the survival loop demonstrably
closes. This is a world-config change (`configs/world/`), ablatable, and touches no invariant.

## Predictions (to pre-register at launch, not before)

- **P1 — the indifference breaks.** Today critic value is ≈equal for dormant and awake
  bodies (212≈212 on beta_10) and homeostatic return is net-negative. After the viability
  drive, value should be *strictly higher* for a viable-and-active body than for a dormant
  one, and per-life cumulative homeostatic+viability return should clear 0. If it doesn't,
  the viability geometry is miscalibrated (scale/floor/lethal), not the theory — retune the
  barrier before concluding anything about behaviour.
- **P2 — eating decouples from boredom.** The forensic channels show a rising share of eats
  paid for by `r_via` while hungry (the 104:1 ratio moves), not by `bored` while sated.
- **P3 — dormancy stops being an attractor.** Hibernation-ledger death share falls below
  011's 23/24; death ages leave the ~347k clock; late-run dormant fraction falls instead of
  rising.
- **P4 — mortality is represented.** Imagined value on trajectories approaching the lethal
  integrity floor turns sharply negative (introspection); agents show anticipatory avoidance
  of the boundary (course-correct before the crash, not after).
- **P5 — the capacity fork resolves.** 012a and 012b diverge or don't. Either answer sizes
  the survival substrate for the social-emergence rounds.
- **Falsification branch:** if the sign flips (P1) and mortality is represented (P4) but
  eating-while-hungry still fails (P2/P3), the remaining suspect is the *actor's* ability to
  execute a multi-step approach-food policy under a barrier gradient — i.e. skill/credit, not
  motivation — and the anima track (proposal 002) or an explicit option/skill layer moves up.

## Pre-work before any code (cheap, decisive, do first)

1. **Confirm the reframe on existing data. ✓ DONE 2026-07-08 — confirmed on all three runs.**
   Mean `reward_homeostasis` is negative in every 400k window of beta_08, beta_09 *and*
   beta_10 (−0.004 to −0.006, worsening start→end in each): the body-drive is a net drag on
   living, universally, across capacity/conditioning/reachability regimes. And the linchpin:
   median critic **value is identical dormant vs awake in all three runs** — 422≈433 (08),
   494≈495 (09), 212≈212 (10). The indifference-to-aliveness is a robust three-run invariant,
   not a beta_10 artifact; value tracks curiosity magnitude (08/09 curiosity higher → value
   ~420–495; 10 lower → 212), confirming curiosity is the sole prop holding value above the
   cessation floor. Deaths deep in deficit (16/19, 26/26, 23/24 hibernation-main) confirm
   `d_death ≫ d_birth`, i.e. the telescoping endpoint is negative. Caveat: `reward_homeostasis`
   logs as a per-update batch mean, so the per-life integral is inferred from sign+telescoping,
   not summed exactly — a per-tick reward log would make it exact (cheap add for 012). **The
   intervention now has a precise measurable target: break the `value_dormant == value_awake`
   identity.**
2. **Calibrate the barrier offline.** The conditioning gym (docs/training-ops.md) can replay
   dreamer_042's buffer (011's designated screen blob) through a candidate `viability_scale /
   floor / lethal / safe` and check that (a) the lived-return sign flips and (b) the hungry
   marginal value dominates — before committing a launch.

## Deliberately not in scope

Reproduction/budding (proposal 001), the plastic-valence architecture (002 — neighbouring
track, would consume this drive), OBS_VERSION 4 interoception, Stage-2 camera vision, and any
"+reward for staying alive" survival bonus (a task; forbidden). The viability drive is the
minimum that could make self-preservation *emerge*, and it names its sequels rather than
bundling them.
