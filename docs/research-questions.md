# Research Questions

This platform exists to probe questions the episodic RL literature mostly doesn't touch.
Every long run should be attributable to at least one of these. When a run produces
evidence (or interesting absence of evidence), write it up in [journal.md](journal.md).

## 1. Lifelong, non-episodic world-model learning

Can a Dreamer-class agent learn continuously through one unbroken life — no resets, no
episode boundaries — in a world that other agents are simultaneously changing?

- World-model literature (DreamerV3, TD-MPC2, Plan2Explore) is almost entirely episodic:
  the agent is reset thousands of times into a stationary MDP. Here, an agent has one
  life in a nonstationary world.
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

## Non-goals

- Benchmark scores, sample-efficiency comparisons, task success rates. There are no
  tasks. If a proposed experiment needs a task or a reset, it belongs in a different
  project.
