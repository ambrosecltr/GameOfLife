---
round: 012
title: the mortality round — a viability drive, and a body worth keeping
date: 2026-07-08            # pre-registered at staging; NOT yet launched
status: planned
question: The five-round competence wall is not capacity, conditioning, or reachability — it is that the HRRL comfort drive telescopes to a NEGATIVE return over any mortal life, so the reward gives an agent no positive stake in its own survival and the critic is mortality-blind (beta_10, offline: value flat across integrity). Does an added viability drive — a standing danger tax on distance to the LETHAL floor — install a mortality gradient the actor can act on, so eating finally rises while hungry and lifespans leave the hibernation clock, without a survival task or fitness function?
headline: ""
runs:
  - save: saves/beta_11
    config: configs/run/beta_11_mortality.yaml
    brain: configs/brain/beta_11_dreamer.yaml
    commit: pending
    ticks: pending
    role: experiment (arm A — base/cloud, the verdict)
  - save: saves/beta_11_nano
    config: configs/run/beta_11_nano_mortality.yaml
    brain: configs/brain/beta_11_nano_dreamer.yaml
    commit: pending
    ticks: pending
    role: side-quest (arm B — nano/local, the capacity question)
baselines: [011, 009]
tags: [mortality, viability, reward-geometry, homeostasis, boredom-decoupling, affordances, nano]
---

# 012 — the mortality round

## Why this round (the reframe)

Rounds 004–011 each named a binding constraint — curiosity collapse (004),
capacity (008), signal conditioning (009), reward reachability (011) — fixed
it, verified the fix worked *mechanically*, and watched behaviour not move.
Four clean exonerations is evidence the constraint is upstream of all of them,
in the **geometry of the reward**. The proof is arithmetic over the code we
run: the HRRL comfort reward `scale·(d(t-1)−d(t)) − penalty·d(t)` telescopes
over a life to `scale·(d_birth − d_death) − penalty·Σd`. Newborns spawn full
(d_birth small); mortal lives end hungry or hurt (d_death large); so the first
term is negative and the second always ≤ 0. **Every mortal life earns negative
cumulative homeostatic reward.** Eating is loss-mitigation on a debt the body
keeps re-incurring, not a net gain. The only positive term is curiosity, which
is survival-agnostic. So the reward gives a mind **no positive stake in its own
body's survival**.

Confirmed on existing data (2026-07-08): `reward_homeostasis` is negative in
every 400k window of beta_08, beta_09, and beta_10. And offline, beta_10's
critic is **mortality-blind**: trained over a recorded life and read on real
states binned by integrity (the lethal variable — energy→0 is recoverable
dormancy; integrity→0 is the only true death), it values a body at 5% integrity
the same as a healthy one (value gap −0.3). Martin, Everitt & Hutter (2016)
supply the principle: cessation sits at an implicit reward of 0, and
self-preservation requires the lived stream to clear that bar by a margin —
which the one drive that is about the body clears by a *negative* margin.

(Retraction carried from proposal 003: an earlier "value identical dormant vs
awake, 212≈212" claim was a metric artifact — `value` is state-unconditioned
and dormant states aren't in the buffer. The telescoping-sign result stands on
its own; the indifference framing did not, and is replaced by the
value-vs-integrity measurement above.)

## What changed vs beta_10 (the only brain knob)

The A/B control is **beta_10 itself**: `beta_11_dreamer.yaml` is
`beta_10_dreamer.yaml` plus one added term. Everything else — the conditioning
stack, the capacity bundle, and beta_10's three reachability knobs (blackout
priced, prioritize reward, spike_loss_weight 4) — is carried byte-identical.

**`reward.viability`** — a log-barrier on distance to the LETHAL floor, added
to (not replacing) the comfort drive. Where the comfort drive is convex
distance to a comfort SETPOINT, viability is unbounded distance to a BOUNDARY:
`V = Σ w_i·(−log((x_i − lethal_i)/(safe_i − lethal_i)))₊`, capped, over energy
(recoverable-dormancy floor) and integrity (true-death floor). It rewards no
action and names no behaviour — a pure function of internal state, charter-clean.

Offline calibration (dreamer_042, `value_vs_energy.py`, 2026-07-08) chose the
form and set the magnitude:

- The **reduction** part (`scale·(V(t-1)−V(t))`, HRRL movement away from the
  floor) reproduced the hibernation attractor in *value* space — escaping the
  floor pays, so the floor becomes a valuable launchpad. It bent value the
  wrong way. **Left at 0.**
- The **standing tax** (`floor·V`, a cost of *being* near the floor) is what
  installs the gradient. At `floor 1.0` it flipped the value-vs-integrity gap
  from beta_10's **−0.3** (blind) to **+4.5** (safe out-values dying) — the
  first mortality gradient in the project — while correctly keeping recoverable
  low-energy states valued (energy gap stayed negative). `floor 2.0` over-taxes
  and crushes the value scale, so **1.0 is the operating point.**

Two composed knobs, same constraint:

- **`death_terminal: true`** — true death (integrity → lethal) terminates the
  imagined stream, so its ~0 return backs up through the critic: a fear of
  death from prediction, though the body is never experienced dying (invariant:
  no episodes). Composes with `blackout: priced` via `cont = integrity>lethal`,
  so the recoverable energy collapse stays non-terminal and only integrity death
  terminates. The terminal targets need a delivery path: a dying body is never
  observable from inside (dormant bodies don't act, and the death tick removes
  the robot before sensing), so without one the cont head would train "continue"
  everywhere and the knob would be inert (pre-launch review). The runtime
  therefore hands the dead body's last observation to its brain
  (`Brain.record_death`, non-blocking — the sim never waits), and the brain
  records it with the vitals at the state the world actually reached: integrity
  0, energy 0 too for a hibernation death. One real sample per death, at the
  floor, priced like any lived transition.
- **`boredom.gate: viability`** — the round-011 decoupling: an agent far from
  the lethal floor can be safely bored (honest play) while merely peckish;
  only true danger shuts the boredom gate.

New instrumentation: `reward_viability` / `viability_level` (the channel, kept
separate from `reward_homeostasis` so we can read which drive paid for each
transition), and **`life_return_homeo` / `life_return_via`** — the exact
per-life realized return, accumulated per lived tick in `act()`, which makes
the telescoping-sign claim measurable directly instead of inferred.

Replay salience under the staged form: the barrier's priority signal is
`|scale·ΔV| + floor·V` — the standing tax, not just the delta. With the
reduction at 0 the delta term contributes nothing, and near-floor drift is
slow, so per-step deltas are small anyway; it's the tax level that makes
near-death excursions loud to reward-aware replay, exactly as they are to the
reward head.

## The affordance precondition (world)

`configs/world/beta_11_2x_food.yaml`: bush density 0.012 → **0.024** (2×). Round
011 found even a perfect policy — scripted foragers — starved in generation-
scale swings (intake 4–8× on spawn luck with 300+ ripe bushes standing), so
meal geometry is patchy enough to gate *any* policy and would confound a
survival-drive test. This is a **precondition, not a treatment**: enough that a
competent policy can reliably stay fed, so failure to stay fed is attributable
to the mind, not the map. Single-lever and ablatable (revert to 0.012 for the
1× control); the bush budget is conserved after generation, so doubling density
doubles the standing food map at all times. Titrate down once the survival loop
closes. Foragers remain the cross-run anchor; their intake in the 2× world is
itself the check that the affordance change did its job.

## The two arms

- **012a — `saves/beta_11`, base preset, cloud, paced.** The verdict. Tests the
  mechanism at a capacity where the world model is known to converge (008), so
  imagination can predict the approach to the boundary and thus fear it.
- **012b — `saves/beta_11_nano`, nano, local M1, background.** The capacity
  question: was brain *size* ever the problem, or was it always our approach?
  Paired with the 2× food and judged on a converged-enough criterion, not
  against 012a's absolute behaviour (round 008: nano never converged the obs-v3
  model). A null here with a win on 012a is a clean result — "survival has a
  capacity floor"; a win here is the bigger prize, freeing base brains for
  social/communication emergence on a working survival substrate.

## Predictions (to finalize at launch; written at staging)

- **P1 — value acquires a mortality gradient.** beta_10's critic is flat in
  integrity (offline gap −0.3); under the viability drive, value should fall as
  integrity → the lethal floor and rise into safety (offline reached +4.5). Live
  P1 is the closed-loop version the offline probe could not settle: read `value`
  at low-integrity states via introspection; per-life `life_return_via` should
  price the danger excursions while `life_return_homeo` still integrates
  negative. If value stays flat with the barrier on, the geometry is
  miscalibrated (floor/safe/cap) — retune before concluding on behaviour.
- **P2 — eating decouples from boredom and rises while hungry.** The win
  condition round 011 failed (104 sated meals : 1 hungry). The forensic channels
  should show a rising share of dreamer meals eaten at low energy, paid by the
  survival gradient, not the bored-sated binge. Poisoned-meal fraction finally
  moves off ~24% once avoiding death has a value gradient.
- **P3 — dormancy stops being an attractor.** Hibernation-ledger death share
  falls below beta_10's 23/24; death ages leave the ~347k clock; the dormant
  fraction falls or consolidates instead of the chronic crash cycle. (This also
  cuts sleep-learning headroom — a pacing signal to watch, not a failure.)
- **P4 — mortality is represented.** Imagined value on trajectories approaching
  the lethal integrity floor turns sharply negative; agents show anticipatory
  avoidance (wake and repair before the crash, not after). `death_terminal` has
  real terminal states in the lineage buffer to learn from.
- **P5 — the capacity fork resolves (012a vs 012b).** Either the nano arm shows
  the same mortality behaviour (survival is a basic function, size wasn't the
  wall) or it doesn't (survival has a capacity floor). Both answers size the
  survival substrate for the social-emergence rounds.
- **Falsification branch:** if value acquires the gradient (P1) and collapse
  stops paying (P3) but eating-while-hungry still fails (P2), the remaining
  suspect is the actor's ability to execute a multi-step approach-food policy
  under a barrier gradient — skill/credit, not motivation — and the plastic
  track (proposal 002) or an explicit skill layer moves up.
- Free riders to watch: whether the 2× food alone shifts the forager anchor and
  the ecology (toxic share, overgrazing); whether the viability boredom-gate
  changes the pressure equilibrium (beta_10 ~0.55 under the drive gate); whether
  terraforming (still dreamer-only) tracks the survival drive or only boredom.

## Operations

STAGED, NOT LAUNCHED (2026-07-08). Before any pod: bench on the round's box
(`bench_learn.py --devices cuda` + the 3-thread contention probe), set launch
speed from measured `learn_seconds`, verify `updates == act_steps − warmup` at
first checkpoint. Ports 7302 (012a) / 7303 (012b), no collision with beta_10's
7301. Run PACED, never headless; rerun OFF (008 disk lesson). Pull ≥1 dreamer
blob before any pod dies (the beta_09 lesson; 011 already cost two screening
lives). Note the pacing tension: a working mortality drive REDUCES dormancy
(the point), which cuts sleep-learning catch-up headroom — and 2× food may raise
the awake fraction from the start — so budget launch speed against a
higher-awake world than beta_10's and drop speed if `train_ratio_eff` slips.

## Results

*(pending launch)*

## Interpretation

*(pending)*

## Next

*(pending)*
