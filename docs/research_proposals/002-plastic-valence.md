---
proposal: 002
title: plastic-valence — a backprop-free, world-model-free brain that learns by feeling
date: 2026-07-07
revised: 2026-07-08          # folded in the round 011/012 mortality reframe + viability drive
track: anima                 # provisional name for the plastic-neuroevolution track (vs beta = world-model)
status: closed               # bet falsified after 7 rounds (anima_01–07); see research_journal/anima/007-centered-valence.md
targets_round: anima_01      # first round of a new track; forks beta's base, runs local on M1
question: Can a brain with no gradient descent and no world model — a recurrent net whose fast weights adapt online via neuromodulated Hebbian plasticity, gated by an evolved homeostatic valence signal — keep itself alive in beta's world, and does the evolved valence map diverge across lineages into persistent individuality?
depends_on: [009, 011, 012]  # forks beta's drive block AND inherits the viability drive + 2x-food world from 012; does NOT wait for 012's live verdict
independent_of: [001]        # uses the existing respawn+inherit+mutate loop; budding is an optional later integration
arc: plastic-valence brain (A, this doc) → self-model / introspection (B) → dreaming / offline consolidation (C) → evolved aging / plastic senescence (D)
sibling_track: active-inference (a parallel comparison track, not a sequel in this arc)
tags: [architecture, neuroevolution, plasticity, neuromodulation, affect, homeostasis, individuality, continual-learning, viability, mortality]
---

# 002 — plastic-valence

> **CLOSED (2026-07-10).** The track ran 7 rounds (anima_01–07) and the founding bet —
> that feeling alone, with no learned predictor, can gate Hebbian consolidation into
> survival competence — is falsified: a three-factor rule needs a ~zero-mean modulator,
> and every unlearned geometry of felt-state valence (reduction, level, level−EMA)
> integrates negative over a mortal, never-thriving life. The fix (centering against a
> learned expectation, i.e. a critic/TD error) is outside this family's charter by
> definition. Full verdict: `docs/research_journal/anima/007-centered-valence.md`.
> Note: the "structural immunity" argument below (that anima escapes beta's
> telescoping-negative return) was tested and is WRONG in its strong form — the
> telescoping reappears inside the modulator itself (anima_05/06/07).

This is the first round of a **new track**, not the next beta round. Every round from 004
to 011 asked the same question with the same architecture — can *this one Dreamer* learn
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

> **Revision note (2026-07-08).** This doc was first written the day before the round-012
> mortality reframe (proposal 003) landed. That reframe changed what "beta's feeling signal"
> *is*, and it does so in a way that turns out to matter more for anima than for beta. The
> two big edits below: (1) the feeling signal `M` is no longer bare drive-reduction — it now
> carries the **viability drive** (a log-barrier on distance to the lethal floor), because
> plain drive-reduction was proven *mortality-blind*; and (2) anima is argued to be
> **structurally immune to the exact failure** (telescoping-negative return) that the beta
> arc has spent five rounds fighting — while paying a different, honestly-stated price for
> that immunity. The affordance precondition (2× food) and the per-life return
> instrumentation from 012 are adopted wholesale.

## The one new idea: decouple feeling from maximization

beta already has feeling. Its homeostasis is HRRL drive-reduction (Keramati & Gutkin):
`reward = movement of internal state toward setpoints`, so the meal that saves a starving
agent outweighs a snack at satiety. That valence signal is real and it is good. What beta
does *with* it is the thing worth challenging: valence becomes a **reward an actor-critic
maximizes inside an imagined world-model rollout**, via backprop. Feeling → reward →
planner → gradient.

This track keeps the *identical* feeling signal and strips out everything downstream. Let
`M` = beta's valence signal. In beta, `M` is a reward to be maximized. Here, `M` is a
**neuromodulator that gates plasticity**: it decides which of the synapses that just fired
get consolidated, and nothing performs `argmax` over expected `M`. There is no planner
reaching for pleasure; pleasure just decides what sticks.

That is the whole bet, and the two tracks answer it head-to-head in the same world, same
obs contract, same drive definitions: **does feeling have to be planned-toward to shape a
life, or is it enough for feeling to decide what is learned?** beta gives one answer; a
nervous system gives the other.

## Why this is *sharper* after the mortality reframe (the load-bearing new argument)

Rounds 008/009/011 each named a binding constraint (capacity, conditioning, reachability),
fixed it, verified the fix worked *mechanically*, and watched behaviour not move — three
clean exonerations. Proposal 003 then showed why: the constraint was upstream of all of
them, in the **geometry of the reward the actor maximizes**. Summed over a mortal life,
beta's homeostatic reward telescopes to

```
Σ r_homeo = drive_scale·(d_birth − d_death) − level_penalty·Σ d(t)
```

Newborns spawn full (`d_birth` small); mortal lives end hungry or hurt (`d_death` large);
the second term is ≤ 0 always. **Every mortal life earns negative cumulative homeostatic
reward, and living more makes it more negative.** Confirmed empirically:
`reward_homeostasis` is negative in every 400k window of beta_08/09/10. And offline,
beta_10's critic is *mortality-blind* — trained over a recorded life and read on states
binned by integrity (the lethal variable), it values a body at 5% integrity the same as a
healthy one (value-vs-integrity gap **−0.3**). Martin/Everitt/Hutter (2016) supply the law:
cessation sits at reward 0, and self-preservation needs the *lived* stream to clear that bar
— which the one drive that is about the body clears by a *negative* margin.

**Here is the point for anima, and it is the strongest form of this track's thesis.** That
failure is a failure of **maximization**, not of feeling. It requires a machine that (a)
sums `M` over time into a return and (b) chooses actions to maximize that return; only then
does "the integral is negative and cessation is worth 0" make stopping attractive. *anima
has neither.* Its update is

```
ΔW_fast = M · α · trace − decay · W_fast
```

Nothing sums `M` over a life. Nothing does `argmax` over expected `Σ M`. There is no value
function for cessation to look attractive *inside*. So the telescoping-negative-return
pathology — the thing the entire beta arc has been fighting — **cannot arise in anima**.
This is not a tuning claim; it is structural. If the reframe is right that the wall is
maximization-of-a-net-negative-signal, then removing maximization removes the wall.

**The honest price of that immunity — state it up front.** Removing maximization also
removes the machine that could *represent* death before experiencing it. beta_12's
`death_terminal` makes imagination fear a death never lived, by backing an absorbing ~0
return up through the critic. anima has no critic and no imagination, so it has **no
anticipatory death representation at all**. Its only mortality mechanism is *reactive*:
neuromodulated plasticity consolidating whatever it did on a near-death excursion it
actually lived through and escaped. So the two tracks are not "same fix, different brain" —
they are opposite bets on where mortality-competence comes from: beta bets on *imagined*
terminal value; anima bets on *felt, lived* survival gating. Comparing them is the point.

## The feeling signal `M`, after 012: comfort drive **plus** viability barrier

Proposal 002's first draft set `M` = beta's bare drive-reduction. The reframe proved that
signal is the wrong feeling to learn from near death: it is convex distance to a *comfort
setpoint* (0.85 energy is nice), it is flat as you approach the lethal floor, and it is
mortality-blind. 012 fixed this for beta by *adding* a second homeostatic term with a
deliberately different geometry — the **viability barrier**, already implemented and
offline-validated (`brain.py:643` `_viability`, `brain.py:661` `_viability_reward`):

```
V(t) = Σ_i  w_i · ( −log( (x_i(t) − lethal_i) / (safe_i − lethal_i) ) )₊     # capped at barrier_cap
```

over the survival-critical variables (energy → recoverable-dormancy floor; integrity → true
death). Its defining property: **the marginal cost of a lost unit explodes as the floor
nears** — a calorie when starving is worth far more than at satiety, the "survive ≫ pass the
time" asymmetry round 011 measured absent (104 sated meals : 1 hungry). Offline on
dreamer_042 it flipped beta's value-vs-integrity gap from −0.3 (blind) to **+4.5** (safe
out-values dying) — the first mortality gradient in the project.

anima inherits `V`. But *how* it uses `V` diverges from beta, and the divergence falls
straight out of the immunity argument above:

- **beta needs the standing tax and rejects the reduction.** Offline calibration showed the
  reduction form (`scale·ΔV`, reward for *moving away* from the floor) reproduced the
  hibernation attractor **in value space** — escaping the floor pays, so the floor becomes a
  high-value launchpad. That launchpad only exists because a value function integrates
  future escape-reward. So beta uses `scale 0, floor 1` (standing danger tax only,
  `beta_11_dreamer.yaml:96`).
- **anima wants the reduction and the standing tax is nearly meaningless.** anima has no
  value function, so *there is no launchpad to create* — the failure beta rejected the
  reduction to avoid cannot occur here. And for a plasticity *gate*, the reduction is
  exactly the right shape: escaping the lethal floor produces a large positive `M` →
  **strong consolidation of the behaviour that just saved the agent**, with the magnitude
  scaled by how close to death it was (the `−log`). Approaching the floor produces a large
  negative `M` → anti-Hebbian suppression of the behaviour that endangered it. A *standing*
  tax, by contrast, is a roughly-constant negative gate whenever near the floor: it would
  anti-consolidate *whatever the agent happens to be doing* while in danger, uniformly —
  a death-spiral eraser, not a survival teacher. So anima's founder mean is the **mirror of
  beta's**: viability `scale` ON (reduction as the gate), `floor` ≈ 0.

  This is a genuine, non-obvious consequence of the two architectures, not a free parameter:
  *the same barrier term wants opposite emphases in a value-maximizer versus a plasticity
  gate, and the reason is precisely the value function beta has and anima lacks.* It is a
  founder hypothesis, an evolvable gene (both coefficients live in the genome), and P5 below
  pre-registers it as falsifiable.

So `M`, evolved per-lineage from the same interoceptive `proprio`:

```
M(t) = comfort_gain · (d(t-1) − d(t))              # HRRL comfort-reduction (signed)
     + viability_gain · (V(t-1) − V(t))            # escape-death consolidation, magnitude-scaled by −log
     − standing_gain · V(t)                        # optional standing danger tax (founder ≈ 0; an ablation)
```

Positive `M` (ate while hungry, pulled back from the lethal floor) consolidates what just
fired; negative `M` (took damage, slid toward death) is anti-Hebbian and suppresses it.
Pleasure and pain are the learning gate, literally — and now the gate *screams* exactly when
survival is on the line, which is where a mortality-blind comfort signal was silent.

The interoception→`M` mapping is **genome-encoded and evolved**, not designed. We give the
setpoints and lethal/safe floors (as beta does) and let lineages evolve how sharply their
bodies feel each deviation and each brush with the boundary. We never write "food is good" —
an agent discovers that eating restores energy and pulls it off the barrier, and its evolved
valence marks that as worth keeping.

**Where exploration comes from.** In a no-reward architecture there is no policy being
optimized toward anything, so movement can't come from maximizing a curiosity reward (as it
does in beta). It comes from intrinsic motor activity — a heritable **restlessness** in the
motor output (action-space noise whose scale is a gene) plus whatever the recurrent dynamics
produce. Evolution shapes restlessness against consolidation: too little and the agent never
stumbles into food; too much and it never settles on what worked. A natural refinement worth
flagging (and cheaply gate-able on `V`, mirroring 012's `boredom.gate: viability`): let
restlessness *fall* as `V` rises, so an agent near death stops wandering and exploits the
behaviour that is keeping it alive. Founder-optional; whether it is *needed* is part of P1.

## The genome, and two ways to inherit

The simplified digital genome (reuses proposal 001's genome-primary decision and the
existing `Brain.inherit` path, `base.py:55`; the mutate-on-inherit machinery already exists
for temperament, `brain.py:1350`):

| gene group | what it sets |
|---|---|
| valence map | interoception → `M`: comfort weights, viability weights, and the `comfort_gain` / `viability_gain` / `standing_gain` split |
| plasticity | per-layer `α`, eligibility `τ`, `decay` |
| modulator | gain / sign / baseline of `M` |
| restlessness | motor-noise scale (the exploration drive); optional viability-gated decay |
| innate wiring | `W_slow` init seed / scale |
| temperament | the existing heritable multipliers, carried over |

Inheritance has a clean, cheap ablation that maps onto the *existing* `inherit_weights` flag
and directly probes research-question 4 (cultural transmission):

- **Darwinian (`genome`)** — child inherits the mutated genome only; `W_fast` reinitialised.
  Pure genetic evolution: no learned experience crosses the generation gap.
- **Lamarckian (`lineage` / `random_living`)** — child also inherits the parent's *learned*
  `W_fast`. Experience passes on, as beta's weight-inheritance does today.

Running both and comparing separates "the lineage evolved a better innate brain" from "the
lineage carried forward a well-trained one" — the same confound proposal 001 flags (P3), but
here it is a first-class arm because the genome and the learned weights are physically
separate tensors.

## Modelling the fundamentals of life (mapped to the real contract)

Everything routes through the existing `Observation`/`Action` contract — **no OBS_VERSION
bump**, invariant 6 untouched:

- **See / hear** — `rays` (color vision + gaze), `sound`, as-is.
- **Feeling (pleasure / pain)** — `M`, the evolved valence over interoceptive `proprio` +
  `events` (ate, took_damage), now carrying the viability barrier. Native, and the *engine*
  of this round.
- **Energy / body / homeostasis** — the drive setpoints and lethal/safe floors inherited
  from the beta_11 block.
- **Mortality** — reactive only, and deliberately so: the barrier's `−log` makes near-death
  transitions the loudest learning signal the agent ever gets, so escapes consolidate hard.
  There is no imagined terminal value (beta's `death_terminal` has no home without a critic);
  anticipatory fear-of-death is explicitly deferred to round B, where a self-prediction head
  could let the barrier gradient be *predicted* rather than only felt on arrival.
- **Identity** — the slow genome + `W_slow` + the un-consolidated net: the part that persists
  when the fast weights churn.
- **Introspection / self-model (round B)** — a small head that predicts the agent's own next
  `proprio`/`M`; its error is a felt "surprise about myself" and the natural second
  modulator — the novelty/curiosity signal this architecture can compute without a world
  model, and the first thing that could make mortality *anticipatory* here.
- **Dreaming (round C)** — during the night dormancy the world already has, replay the
  eligibility traces through the plastic rule: offline consolidation, nearly free.
- **Aging (round D)** — let `α` decay with `age_ticks` (a biological critical period).

## The world: adopt the 012 affordance precondition

anima runs in **`configs/world/beta_11_2x_food.yaml`** (bush density 0.012 → 0.024), not
`default.yaml`. This is a change from the first draft and it is not optional for anima —
it is the same precondition 012 adopted, and anima needs it *more* than beta does. Round 011
found even a *perfect* policy (scripted foragers) starving in generation-scale swings on
spawn luck alone (intake 4–8× between forager generations with 300+ ripe bushes standing):
meal geometry is patchy enough to gate *any* policy. A brain whose only exploration is
restlessness and whose only learning is Hebbian consolidation is *maximally* exposed to that
confound — an anima agent that correctly wants to eat but can't find food looks identical to
one that never learned to. So the 2× food world is how we keep P1 ("does it forage at all?")
attributable to the brain and not the map. Ablatable back to 0.012 for a 1× control once the
survival loop demonstrably closes.

Comparability note: beta_11 also runs in the 2× world, so anima↔beta stays a clean
same-world comparison. When we later compare against the pre-012 beta runs (008–010, which
ran 1× food), the food density is a known confound to control for, not a hidden one.

## Config surface (illustrative — sized in the calibration pass, not final)

New brain family, forked from beta_11's drive + viability blocks (the valence source) with
the world-model / actor-critic / replay / training / curiosity blocks removed:

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
# --- the feeling signal M, forked from beta_11 (comfort drive + viability barrier) ---
valence:
  # comfort drive (HRRL, verbatim from beta_11_dreamer.yaml drive block) — signed reduction
  comfort_gain: 1.0
  drive:
    scale: 3.0
    level_penalty: 0.01        # as a gate this is a small standing anti-Hebbian bias; keep low
    pow_m: 3.0
    pow_n: 2.0
    energy_setpoint: 0.85
    energy_weight: 1.0
    integrity_setpoint: 1.0
    integrity_weight: 1.0
    rested_setpoint: 1.0
    rest_weight: 0.5
  # viability barrier (from beta_11) — but MIRRORED for a plasticity gate:
  # reduction ON (escape-death consolidates), standing tax ~0 (see the argument above).
  viability:
    viability_gain: 1.0        # the reduction gate (beta uses scale 0; anima uses it as the survival teacher)
    standing_gain: 0.0         # the standing tax (beta's operating point; ≈0 here — an ablation)
    barrier_cap: 4.0
    energy_lethal: 0.0
    energy_safe: 0.25
    integrity_lethal: 0.0
    integrity_safe: 0.5
    energy_weight: 1.0
    integrity_weight: 1.0
genome:
  enabled: true
  sigma: 0.25                  # founder diversity (log-stddev over gene multipliers)
  mutation_sigma: 0.1          # drift applied on inherit
```

Run config, forked from `beta_11_mortality.yaml` but retargeted to the M1 (cpu) and to a
**large cheap founder population** — no cuda, no pacing math (no learner-thread gradient
step to pace against; plasticity is a per-act-step local update):

```yaml
# configs/run/anima_01.yaml
world_config: configs/world/beta_11_2x_food.yaml   # the 012 affordance precondition
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

- **`gol_brains/plastic/`** — new family: a GRU core, the three-factor plastic linear layer,
  the genome dataclass, and the `M` computation. **Lift `_drive_level` and `_viability` out
  of `dreamer/brain.py` into a shared `gol_brains/feeling.py`** so both tracks compute
  interoception→feeling from one definition (the reframe made this refactor worth doing now:
  the barrier is subtle and we do not want two copies drifting). The plastic brain imports
  the same functions; only the *use* (gate vs reward) differs.
- **`registry.py`** — add `kind: plastic` to `build_brain`'s dispatch (`registry.py:38`) and
  to `is_learning_kind` if the learner thread should ignore it (it should — see below).
- **`base.py` methods** — `act` runs the forward pass + applies the plastic update inline
  (learning is *in the act step*, so `learn()` stays a no-op and `target_train_ratio()`
  returns 0 — the learner thread never schedules this brain, and invariant 5 is satisfied
  trivially). `introspect()` surfaces `M` and its split (`M_comfort` / `M_viability`), mean
  `|W_fast|`, and trace magnitude. `state_dict()` serialises genome + `W_slow` + `W_fast`.
  `inherit()` mutates the genome and, depending on the flag, reinitialises or copies
  `W_fast`. `reset_stream()` zeroes the eligibility trace and live recurrent state on a
  respawn (a newborn must not consolidate on a trace it did not live — the same stream-break
  hygiene 011 added for the reward reduction, `brain.py:1139`). `wake()` and `record_death()`
  are available (`base.py:72`, `base.py:83`) but largely inert for anima: with no critic
  there is no terminal value to deliver, and the dormancy blackout is simply a stream break.
- **No world.py / scheduler.py changes** — this round uses the existing respawn+inherit loop;
  only a new `inherit_weights: genome` mode (reinit fast weights) is added. Endogenous budding
  (001) is an *optional later* integration, not a dependency.
- **`gol-stats` / metrics** — valence-map census across lineages (does interoception→`M`
  diverge?); per-agent `M`, its comfort/viability split, and plasticity traces; **per-life
  realized `M` integral split into comfort vs viability contributions** — the direct analogue
  of 012's `life_return_homeo` / `life_return_via` (`brain.py:557`, `:574`), so we can read
  the geometry a plastic life actually feels; foraging rate vs the forager anchor and vs beta;
  per-tick wall cost (the "runs quick / fits on M1" claim, measured).

## Calibration first (the M2 / M1-fit pattern)

Before the real run, size the population to the hardware, exactly as the M2 economy was
calibrated on scripted foragers first:

1. **Measure per-tick cost** of one `plastic` brain on the M1 (cpu). It should be far below a
   dreamer's ~500 ms/update — forward pass + one outer product.
2. **Scale `target`** up until the world holds a usable tick rate (aim to keep speed-1
   real-time-ish, ~20 t/s; more agents is better for emergence, so push it as far as the tick
   rate tolerates). Record the number; that is the round's founder population.
3. **Screen the two viability-gate forms offline before launch (decided 2026-07-08).** The
   reduction-gate-vs-standing-tax bet (above) is an untested architectural claim, and it is
   cheaper to settle before spending M1 time than to unwind after. Reuse the 012 method:
   replay a recorded forager/dreamer life through the *feeling* module under **both** forms —
   `viability_gain` ON / `standing_gain` 0 (the mirror bet) and `viability_gain` 0 /
   `standing_gain` ON (beta's form, as a gate) — and inspect which produces the saner
   consolidation signal: does the reduction form spike positively on lived near-death
   *escapes* (the behaviour we want consolidated) while the tax form merely suppresses
   in-danger behaviour uniformly? Also confirm `M`'s viability component is quiet at satiety
   and `viability_level` stays well under `barrier_cap`. This chooses the founder form the way
   `value_vs_energy.py` chose beta's, and it de-risks P5 before launch.
4. **Lock it, then run.** Separates "the population is too big for the M1" from "plastic
   brains can't forage."

## Predictions (pre-registered — including the failure branches)

- **P1 — does it forage at all? (the chicken-and-egg, first thing to check.)** A brain with
  only restlessness for exploration and only Hebbian consolidation for learning may never
  stumble into enough meals to consolidate foraging before dying. The 2× food world is the
  precondition that makes this a fair test (011: even a perfect policy starved on spawn luck
  at 1× food). If forage rate still sits at noise, it is a clean, informative null: *this
  world needs planned-toward reward, not just felt reward.* Non-cheating mitigations: the
  large founder population (some survive on luck and seed selection) and the frozen-net
  control below.
- **P2 — the evolved valence map diverges into individuality.** Across generations, the census
  of interoception→`M` genes should spread and drift *directionally* (toward whatever keeps
  agents fed and off the barrier), not stay at founder diversity. This is beta_07's
  individuality finding, but now the thing carrying the identity is an explicit, inspectable
  gene vector — and it should include the comfort/viability *balance* drifting, not just the
  weights.
- **P3 — plasticity earns its keep (the frozen-net control) — the first-sitting arms (decided
  2026-07-08).** anima_01 launches as a **plastic + frozen-net pair**: the flagship (plastic,
  `viability_gain` ON, Darwinian inherit) and an `alpha: 0` frozen control — a pure evolved
  reflex agent, `W_fast` frozen at zero — sharing seed protocol and world. If the plastic arm
  doesn't out-forage the frozen arm, within-life learning is contributing nothing and all
  adaptation is genetic — itself a strong, publishable result about this world, and the most
  load-bearing control to run before spending on the Darwinian/Lamarckian split (P4), which
  moves to a later sitting.
- **P4 — Darwinian vs Lamarckian.** The `genome` (reinit fast weights) and `lineage` (carry
  fast weights) arms should diverge: Lamarckian should ratchet competence faster early
  (inherited training) but may homogenise; Darwinian isolates *evolved innate* competence. The
  gap between them is the cultural-transmission signal (Q4), cleanly separated because genome
  and learned weights are separate tensors.
- **P5 — the viability barrier teaches survival *as a gate* (the mortality prediction, and the
  head-to-head with beta).** With `viability_gain` ON, near-death escapes should consolidate:
  per-life `M_viability` should show large positive spikes on lived recoveries, and — the
  behavioural payoff — the sated:hungry eat ratio that beta could never move (104:1 in 011)
  should tilt toward hungry eating, because the loudest learning signal an anima agent gets
  is *"I just pulled myself off the floor, do that again."* If eating stays sated-only with
  the barrier gating hard, the reactive-only limit is real and the anticipatory self-model
  (round B) is required — a clean result that sizes what felt-but-unpredicted mortality can
  buy. This is also where the standing-tax-vs-reduction claim is tested: an ablation with
  `standing_gain` ON, `viability_gain` OFF should *under*-teach survival relative to the
  reduction gate (it uniformly suppresses in-danger behaviour rather than crediting escape) —
  the mirror of beta's finding, and a falsifiable prediction of the architecture argument.
- **vs beta (the headline comparison).** Does a backprop-free, world-model-free brain reach a
  foraging *and* survival competence in the same ballpark as beta's Dreamer, in the same
  2× food world, at a fraction of the compute and with a much larger population? Either answer
  is a finding: parity would be a strong claim about how little machinery embodied survival
  needs — and, given the immunity argument, a demonstration that *removing maximization
  removed the five-round wall*; a large gap quantifies what the world model and its imagined
  terminal value buy that felt-only survival cannot.
- **Population dynamics as a result in themselves.** With ~24 cheap agents in one world,
  crowding, competition over bushes (now doubled), and mutual perceptual salience ("alive"
  ray-kind) may produce dynamics the 3-dreamer runs never could. Any shape — stable,
  oscillating, clustering — is a finding.

## What this round can and cannot claim

- **Can:** whether felt-but-not-planned-toward valence can sustain a life (P1); whether an
  evolved valence map produces inspectable individuality (P2); whether plasticity beats a
  frozen reflex (P3); the Darwinian/Lamarckian gap (P4); whether a viability barrier used as
  a *gate* tilts eating toward survival where beta's value-maximizer could not (P5); a
  compute-matched competence comparison against beta in the same world.
- **Cannot (needs sequels):** *anticipatory* fear of death — anima's mortality is reactive by
  construction; the self-prediction head that could make the barrier gradient predicted, not
  only felt, is round B. Also out: dreaming / consolidation (round C); aging as an evolved
  trait (round D); and the interiority-vs-evolution question the parallel **active-inference**
  track is meant to answer.
- **Inherited caveats:** single run per arm (006 measured ~40% forager variance between
  identical-config runs — trust trajectory shapes and directional trends, not fine levels);
  the temperament↔weights confound is the same one 001/007 carry, here made addressable by the
  genome/lineage split; and the 2× food world means comparisons to pre-012 beta runs (008–010)
  must control for food density.

## The sequel arc (named, not built here)

- **B — self-model / introspection.** A head predicting the agent's own next state; its error
  is a felt surprise about oneself and the second modulator — the curiosity this architecture
  can compute without a world model, **and the first mechanism that could make mortality
  anticipatory here** (predict the barrier gradient before arriving at it, closing the gap
  with beta's imagined `death_terminal` without importing a critic).
- **C — dreaming / offline consolidation.** Replay eligibility traces through the plastic rule
  during night dormancy. Cheap here; a genuine memory-consolidation phase to instrument.
- **D — evolved aging / plastic senescence.** `α` decays with age; the young learn, the old
  exploit, and knowledge must transmit (B/C) or die with the body — the engine behind rearing
  and culture, and the tie-in to proposal 001's senescence sequel.
- **Parallel: the active-inference track** (its own prefix, `kind: active_inference`). Richer
  for pure interiority — free-energy, preference, self-model as native math — run *alongside*
  anima, not after it, so the two architectures can be compared on the same world.

## Invariant check

One persistent world, no episodes/reset (✓). No designer task or fitness function — `M` is
homeostatic valence + a viability barrier (both allowed intrinsic drives; the barrier rewards
no action and names no behaviour, `brain.py:240`) that *gates plasticity*; nothing performs
`argmax` over expected valence, and the valence map is evolved, not authored (✓). World +
brains still checkpoint together — genome + `W_slow` + `W_fast` serialise with the brain,
world state as today (✓). Sim never waits on learning — plasticity is an in-`act` local
update, the learner thread is never scheduled for this family (✓). Obs/action contract
unchanged, OBS_VERSION stays 3, so anima↔beta comparisons are valid (✓).
