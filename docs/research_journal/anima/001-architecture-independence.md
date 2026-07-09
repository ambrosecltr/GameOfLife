---
round: anima-001
title: first sitting — the attractor is architecture-independent
date: 2026-07-08
status: complete
question: does a backprop-free plastic-valence brain escape beta's hibernation attractor, and does within-life plasticity beat a frozen evolved reflex?
headline: "The hibernation attractor is ARCHITECTURE-INDEPENDENT — a backprop-free Hebbian gate-learner converged on the same attractor as five rounds of world-model planners (~82% dormant, deaths on the ~330–350k hibernation clock, 589 sated eats to 0 hungry), and plasticity did not beat the frozen evolved reflex (P3 null); the elegant result is P2: valence genes came under strong directional selection only in the plastic arm where M has a phenotype (viability_gain +21% vs −3% frozen, alpha drifting identically in both) — proof the valence-map evolution is real selection, not drift — while the reduction viability gate integrated net-negative over these lives (P5 suspect)."
runs:
  - save: saves/archive/anima_01/anima_01
    config: configs/run/anima_01.yaml
    brain: configs/brain/anima_01_plastic.yaml
    commit: 0e684ef
    ticks: 721200
    role: experiment
  - save: saves/archive/anima_01/anima_01_frozen
    config: configs/run/anima_01_frozen.yaml
    brain: configs/brain/anima_01_frozen.yaml
    commit: 0e684ef
    ticks: 642300
    role: control
baselines: [011, 012]
tags: [motivation, valence, plasticity, mortality, architecture, individuality]
---

# anima 001 — first sitting — the attractor is architecture-independent

*(Backfilled 2026-07-09 from the retained saves — `anima_stats.py` was re-run on both
archive dirs for this entry, so the numbers below are fresh reads, not recollection.)*

## Why this round

Proposal 002 (`docs/research_proposals/002-plastic-valence.md`) opened the second brain
track: **anima**, `kind: plastic` — a GRU with fast weights adapted online by a
three-factor neuromodulated Hebbian rule `ΔW_fast = M·α·trace − decay·W_fast`, gated by
an evolved homeostatic valence `M`. No backprop, no world model, no critic, no return.
The headline bet is the **immunity argument** from the 011/012 mortality reframe: beta's
five-round competence wall is a failure of *maximizing a net-negative return*
(telescoping homeostatic reward + cessation at zero). anima never sums M into a return
and never does argmax, so it is structurally immune to that exact pathology. The price:
no critic ⇒ no imagined `death_terminal` ⇒ mortality is reactive-only — Hebbian
consolidation of *lived* near-death escapes. Opposite bets, run in the same world.

Two launch decisions were settled before the sitting:

1. **First sitting = plastic + frozen-net pair** (P3, the most load-bearing control):
   the flagship (plastic, `viability_gain` ON, Darwinian inherit) against a frozen
   control (plasticity disabled, `W_fast` pinned at zero — a pure evolved-reflex agent)
   sharing world and seed protocol. Darwinian/Lamarckian (P4) deferred.
2. **Offline-screen the viability-gate form first.** On dreamer_042's 21k-step life
   (`scripts/anima_viability_screen.py`), the REDUCTION gate credited escapes **+11.06**,
   suppressed approaches −0.46, stayed silent when settled-safe; the STANDING TAX
   scored escapes **−0.49** and approaches −2.42 — it only suppresses. So anima launched
   with the reduction form (`viability_gain` ON, `standing_gain ≈ 0`), the mirror of
   beta's choice, pre-registered as P5.

Calibration locked the founder population at **42 plastic + 6 forager = 48**: per-act
cost 0.8 ms on M1 (1 thread), world stepping — not brains — the binding cost,
real-time ceiling ≈ 96 robots, pop 48 ≈ 2× real-time headroom.

## What changed

Everything — this is the first run of a new brain family (commit 0e684ef):

- **`gol_brains/plastic/`** — `network.py` (PlasticLinear, PlasticGRUCell, PlasticNet;
  fully plastic: encoder + GRU candidate + readout all carry `W_slow`+`W_fast`, GRU
  sigmoid gates innate) + `brain.py` (genome multipliers, M via the shared
  `feeling.py`, online act/consolidate, inherit/checkpoint). Stability guards default:
  M clip 5, per-layer decay, `W_fast` clip 2. One-step-delayed credit (consolidate
  `M_t` on the prior-step trace, then forward). Discrete EAT credit = three-factor on
  the taken gripper mode + restlessness-scaled ε-floor.
- **Registry `kind: plastic`**, deliberately NOT in `is_learning_kind` — it learns
  in-act; the learner thread never schedules it.
- **Inheritance Darwinian**: `genome` mode reinits `W_fast`; `W_slow` is an evolvable
  stored tensor (log-normal jitter); `inherit_weights: random_living` warm-start.
- **World**: `beta_11_2x_food` (the 012 2×-food precondition — 011 showed even a
  perfect policy starves on spawn luck at 1× food). Obs v3, senescence halflife 150k,
  reproduction = respawn timer. (Budding and the OBS v4 senescence channel are
  proposal 004, i.e. anima_02 — not in this round.)
- **Frozen arm** (`configs/brain/anima_01_frozen.yaml`): `plasticity.enabled: false`,
  everything else identical; port 7312 vs 7311.
- **Rerun sliding-window memory** (`rerun_memory_limit`, default 2GB) added so the
  live viewer stays bounded at anima's ~128 KB/tick data rate.

## Results

Both arms ran paced, headless, on the M1; both populations stayed at the 48 cap
throughout. Flagship stopped at **721,200** ticks (~30 sim-days), frozen at **642,300**
(~27) — an 80k gap; see Caveats. `anima_stats.py` on the archived saves:

| metric | anima_01 (plastic) | anima_01_frozen (control) |
|---|---|---|
| ticks | 721,200 | 642,300 |
| population | 42 plastic + 6 forager, stable | same |
| plastic eats per 100k ticks (per-window) | 41–133, mean ≈ 74 | 49–154, mean ≈ 90 |
| eating-while-hungry (E<40) | **0 hungry / 589 sated** | 2 hungry / 627 sated |
| dormant fraction (final point-sample) | 0.88 | 1.00 |
| dormant fraction (time-averaged at close) | ~0.82 | ~0.83 |
| plastic deaths | 82 | 68 |
| death age median (p10 / p90) | 340,355 (245,562 / 352,780) | 312,155 (222,536 / 358,130) |
| dominant death cause | hibernation 76, poison 6 | hibernation 58, poison 10 |
| `w_fast_norm` (recent mean) | 0.0668 (0.04–0.08 over the run) | 0.0000 |
| `m_viability` (recent mean) | −0.0184 | −0.0131 |
| `life_return_via` | −10.75 | −8.30 |
| mean plastic age at close | 99,033 | 159,829 |

Foragers (the 6-robot scripted anchor) ate 98–1305 per 100k per window (flagship) and
0–1476 (frozen) — the same spawn-luck-dominated 4–8× swings 011 measured; per capita
they out-eat plastic brains by well over an order of magnitude.

**Gene drift** (founder census <100k → recent, the selection signal):

| gene | plastic | frozen |
|---|---|---|
| viability_gain | **+20.8%** | **−3.0%** |
| integrity_weight | **+15.3%** | +2.1% |
| comfort_gain | +4.8% | +5.8% |
| restlessness | +4.6% | −0.4% |
| alpha | +5.8% | +5.9% |
| via_integrity_weight | +0.9% | +5.9% |

## Interpretation

**1. The hibernation attractor is architecture-independent — the round's headline.**
P1 first: the brain does forage above noise (~74 eats/100k, not zero — restlessness +
Hebbian consolidation is enough to *live*), so the chicken-and-egg null didn't fire.
But the population converged on exactly beta's attractor: ~82% dormant, deaths on the
~330–350k hibernation clock (76 of 82 last-ledger causes), and eating **only while
sated** — 589 sated : 0 hungry, where beta_10 managed 104 : 1. A backprop-free Hebbian
gate-learner with no world model, no critic, and no return lands on the same attractor
as five rounds of Dreamer planners. The attractor is therefore a property of the
**reward geometry / world economy**, not a planning or gradient artifact — the
strongest cross-architecture evidence yet for the 003 mortality-reframe thesis.

**2. P3 null: within-life plasticity does not beat a frozen evolved reflex.** Plastic ≈
frozen on every behavioral metric — eats/tick (frozen slightly *more*, ~90 vs ~74),
dormancy (0.82 vs 0.83 time-averaged), eating-while-hungry (both ≈0), death clock. All
the competence in this world is carried by the evolved reflex (genome + `W_slow`);
`W_fast` contributes nothing behavioral. The mechanism itself is healthy —
`w_fast_norm` stable and bounded at 0.04–0.08, never exploding, never dying — it is
simply learning to hibernate, faithfully.

**3. P2 confirmed WITH a proper null — the elegant result.** Valence genes are under
strong directional selection in the plastic arm (viability_gain +20.8%,
integrity_weight +15.3%) and **flat in the frozen arm** (viability_gain −3.0%), while
`alpha` — a pure plasticity gene — drifts identically in both (+5.8% / +5.9%),
providing a shared drift baseline. Mechanism: in the frozen arm M is inert, so valence
genes have no phenotype and are selectively neutral (no phenotype → no directional
movement); in the plastic arm M gates plasticity, so the same genes have a phenotype
and selection pushes toward mortality-sensing. The frozen control turns "genes moved"
from a drift anecdote into demonstrated selection — the valence map is genuinely
evolving, even while the behavior it gates stays trapped.

**4. P5 mechanistic suspect: the reduction gate integrates net-negative.** Per-step
`m_viability` sat at −0.01 to −0.03 and per-life `life_return_via` at ≈ −10.75: the
modulator budget is dominated by many mildly-negative slow-APPROACH steps (each
anti-Hebbian), not by the rare large ESCAPE spikes (+11) the offline screen was chosen
for — because this population almost never escapes; it slides monotonically into
terminal hibernation. The screen couldn't catch this: dreamer_042's life was
recovery-heavy, this population's lives are decline-shaped. A net-suppressing gate is
the leading candidate for *why* plasticity doesn't help: awake behavior — including
food-seeking — is on net anti-consolidated.

Observation vs inference: the attractor metrics, the plastic/frozen parity, the gene
drift split, and the negative modulator integrals are measured. That the
net-suppressing gate is what neutralizes plasticity is inference — flagged for a
rectified/asymmetric-gate follow-up.

## Caveats

- **Single run per arm** — 006 measured ~40% between-run forager variance; treat all
  magnitudes accordingly. The P2 split (+21% vs −3% on the same gene, with a matched
  drift baseline) is the claim most robust to this.
- **Frozen ran 80k ticks fewer** (642k vs 721k): death-age and eat totals are not
  tick-matched; per-100k rates and fractions are used where it matters.
- The frozen arm's final dormant point-sample of 1.00 is a synchronized-sleep snapshot
  (009 saw whole populations sleep in sync); the ~0.83 time-averaged figure is the
  honest number.
- "Hungry" uses E<40 at eat — the same threshold anima-002 later showed is generous
  (wake_energy is exactly 40). The 0-hungry result here can only get *more* extreme
  under the stricter exposure-normalized analysis 002 introduced.
- Gene drift is measured over a founder-vs-recent census across relatively few
  generations of respawn-timer turnover; directions are meaningful, magnitudes soft.
- The offline viability screen that chose the reduction gate was run on a single
  recovery-heavy dreamer life — a source bias this round exposed (see P5).

## Next

- **anima_02 — the finitude round (proposal 004).** If the attractor is
  reward-geometry and the respawn timer makes death free, make time finite and felt
  (senescence 150k → 90k + OBS v4 senescence proprio channel) and make reproduction
  *earned* (budding gated on a thriving body, replacing the respawn timer) — let a
  survival instinct evolve rather than be taught. Built as d6e8755; ran as
  [anima 002](002-hunger-deafness.md), which also delivered the verdict on this
  round's P5 suspicion (confirmed at scale, one of four stacked blockers).
- **P5 follow-up: rectify the gate** — credit escapes, don't bleed on approach (or
  rebalance the standing tax back in). Shipped as `valence.viability.rectified` in the
  anima_03 staging.
- P4 (Darwinian vs Lamarckian) stays deferred until there is within-life learning
  worth inheriting — this round showed there currently isn't.
- Housekeeping: this entry is the anima/001 backfill the 002 entry flagged; journal is
  now organized per-track (`beta/`, `anima/`) with per-track numbering matching save
  prefixes.
