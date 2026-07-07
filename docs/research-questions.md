# Research Questions

This platform exists to probe questions the episodic RL literature mostly doesn't touch.
Every long run should be attributable to at least one of these. When a run produces
evidence (or interesting absence of evidence), write it up in the
[research journal](research_journal/).

## 1. Lifelong, non-episodic learning — and which architectures can host it

Can an agent learn continuously through one unbroken life — no resets, no episode
boundaries — in a world that other agents are simultaneously changing? And *which brain
architectures* can sustain that: world-models are the first bet, but model-free,
predictive-coding, evolved-plasticity, and other families are all in scope, run side by
side in the same world through the architecture-agnostic `Brain` interface.

- World-model literature (DreamerV3, TD-MPC2, Plan2Explore) is almost entirely episodic:
  the agent is reset thousands of times into a stationary MDP. Here, an agent has one
  life in a nonstationary world.
- The architecture itself is a variable, not a fixed choice. A world-model may not be the
  best substrate for lifelong embodied learning (or for the interiority questions below —
  self-modeling, valence, identity); the platform exists partly to find out by trying
  families against each other, including our own mechanism/param exploration rather than
  only replicating published results.
- Watch for: plasticity loss (prediction error plateauing then rising), catastrophic
  forgetting of distant regions, whether long replay buffers act as sufficient ballast.
- Instruments: per-agent prediction-error trend (first-class metric), spatial revisit
  error (does error spike in places the agent hasn't seen for a sim-day?).

## 2. Intrinsic motivation in a shared world

What happens when curiosity-driven agents are each other's most unpredictable objects?

- Plan2Explore disagreement is epistemic (noise-robust), but other learning agents are
  a moving target: epistemic uncertainty about them never fully resolves.
- Hypotheses to watch: mutual fascination (agents orbiting each other), social
  avoidance once modeled, curiosity-driven chasing.
- Ablation: `curiosity_mask_agents` — mask other-robot ray classes out of the curiosity
  target. Compare social behavior with and without.

## 3. Emergent communication

Does a free 2-channel continuous signal broadcast acquire meaning under survival
pressure?

- The signal costs (a little) energy and has no built-in semantics. Neighbors hear it
  distance-weighted with bearing.
- Watch for: signal bursts correlated with food discovery, distress signaling near
  death, silence at night.
- Instruments: `events.ndjson` signal-burst events with positions + nearby-agent
  responses; mutual information between signal emissions and world events.

## 4. Cultural transmission

Does warm-starting newborns from living agents' weights propagate behaviors?

- Default: fresh brains on respawn. Flag: `inherit_weights: random_living` copies a
  living agent's weights (no fitness selection — a random living agent, which is only
  survivorship-conditioned).
- Watch for: behavioral lineages (foraging routes, hoarding sites persisting across
  "generations"), divergence between inherit-on and inherit-off worlds under the same
  seed.

## System-level questions

The four above are about what an agent *learns and does*. The long runs kept surfacing a
second class of question — about what the *system* becomes over time — that we didn't
write down in advance. These earn their place by having already produced evidence (the
round they came from is noted); they are the ones that most directly answer "what emerges
in a world with no goals."

## 5. What keeps a mind motivated for a whole lifetime?

This has become the project's central question, and the runs discovered it — it was not
planned. Curiosity is self-extinguishing: the better the world model gets, the less
surprise there is to feed on. How does an intrinsic motivation system stay *alive* across
one unbroken life, handing off between drives as each satisfies itself?

- Round 004 is the crux: cross-lifetime learning worked, and *that was the problem* —
  curiosity collapsed 20× as the world became predictable, homeostasis was ~1000× too
  quiet to take over, and behavior decayed to aimless wandering in a food-rich world.
- Rounds 005–008 are successive attacks on it: a louder body (005), richer senses (006),
  and the full gratification stack — learning-progress curiosity, drive reduction,
  boredom, temperament (007) — each testing whether motivation can be made to persist and
  hand off rather than decay.
- The episodic literature never meets this: agents are reset long before they could get
  bored of a mastered world.

## 6. Where does individuality come from?

Do identical minds in a shared world necessarily diverge into persistent individuals, and
what carries that identity — the seed, the life history, the lineage?

- Round 007: three lineages with identical architecture and identical drive settings
  developed distinct, persistent behavioral profiles (forager / social / mixed), and
  individuality arrived *before* survival competence.
- Instruments: `gol-stats --interests` (per-agent activity profiles over time windows —
  do agents differ, and do the differences persist rather than being noise?).

## 7. Does the world itself evolve under the population?

A persistent world shaped by adaptive agents starts generating its own selection pressures
back at them. Does agent–world coevolution appear without any ecology being designed in?

- Round 003's toxic ratchet: the better the population avoided poison bushes, the more
  poisoned the world became — plants effectively evolved defenses under grazing pressure,
  from nothing but the regrow rule and the agents' avoidance.
- Watch for: feedback loops where a behavior reshapes the world in a way that reshapes the
  behavior, on timescales longer than a single life.

## 8. Can natural selection arise without a fitness function?

The prospective one (see [budding proposal](research_proposals/001-budding.md)). If
reproduction becomes a bodily process — funded by an agent's own energy surplus, passing
on mutated temperament — does a population under *no* fitness function develop real
selection on temperament and a cross-generational ratchet in competence?

- Selection would act only through who stays in surplus, i.e. who forages and survives
  well — the opposite of a designer-assigned fitness function (which stays a non-goal).
- It is also the answer to a pattern in Q5–Q7: five rounds each hinged on *one* brain's
  learning speed; selection over an inheriting population is the one lever that doesn't.

## Non-goals

- Benchmark scores, sample-efficiency comparisons, task success rates. There are no
  tasks. If a proposed experiment needs a task or a reset, it belongs in a different
  project.
