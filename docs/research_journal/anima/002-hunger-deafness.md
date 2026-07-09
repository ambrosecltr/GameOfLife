---
round: anima-002
title: finitude, and the hunger-deafness verdict
date: 2026-07-09
status: complete
question: does finite felt time + earned reproduction evolve a survival instinct — and is the plastic brain listening to its hunger feeling at all?
headline: "The inversion probe delivered a clean verdict: comfort valence is behaviorally INERT — flipping the felt hunger sign changed nothing, because the teaching event (eating while hungry) is physically near-unreachable, the meal engram decays before hunger recurs, credit never spans the approach, and the unrectified viability gate net-punishes awake life while the recovery hides behind the dormancy stream reset."
runs:
  - save: saves/anima_02
    config: configs/run/anima_02.yaml
    brain: configs/brain/anima_01_plastic.yaml
    commit: d6e8755
    ticks: 6946200
    role: experiment
  - save: saves/anima_02_thrive50
    config: configs/run/anima_02.yaml (thrive_energy 50)
    brain: configs/brain/anima_01_plastic.yaml
    commit: d82e6ae
    ticks: 1854800
    role: experiment
  - save: saves/smoke_anima_inverted_thrive50
    config: not retained (off-books probe, deleted after run)
    brain: anima_01_plastic + valence.invert_comfort true (uncommitted)
    commit: d82e6ae (+ probe diff, deleted)
    ticks: 2261300
    role: control (sign test)
baselines: [anima-001, 011, 012]
tags: [motivation, valence, plasticity, mortality, reproduction, finitude]
---

# anima 002 — finitude, and the hunger-deafness verdict

## Why this round

anima_02 is the finitude round (proposal 004): sharpened senescence (halflife
150k → 90k) + the OBS_VERSION-4 senescence proprio channel make time finite and
felt; budding (earned reproduction, replacing the respawn timer) lets natural
selection reward lineages that live awake. The bet: a survival instinct
*evolves* instead of being taught. anima_01 had already shown the hibernation
attractor is architecture-independent and that plasticity didn't beat a frozen
reflex; this round tested whether selection pressure on finite lives changes
that. Mid-round, eating-while-sated persisted so stubbornly that we ran an
off-books **sign-inversion probe**: flip the felt hunger/comfort valence
(`m_comfort` sign only; viability untouched) and see if behavior changes. If
the brains listen to hunger, inverted brains should behave measurably
differently. They didn't.

## What changed

- `anima_02` — proposal 004 as committed (d6e8755): world
  `configs/world/anima_02.yaml` (beta_11_2x_food + senescence 90k),
  reproduction.mode budding (thrive E≥75 I≥70, min age 20k, cooldown 15k, cost
  40E/5I, floor 4), 42 plastic + 6 forager anchor.
- `anima_02_thrive50` — one knob: thrive_energy 75 → 50 (is the budding bar the
  bottleneck?).
- `smoke_anima_inverted_thrive50` — thrive50 plus `valence.invert_comfort:
  true` (sign of `m_comfort` flipped in the brain; `m_via` untouched). **Config
  confound:** this arm ran on `beta_11_2x_food.yaml` directly, i.e. senescence
  halflife 150k not 90k — explains its longer lives and larger population, not
  its eating pattern. Probe code deleted after the run (was never committed).

## Results

**The inversion took effect, and nothing behavioral changed.**
`life_return_comfort` (per-life integral of felt hunger valence): normal runs
mean −0.94 / −1.09 with 98%/96% of robots negative; inverted run **+1.20 with
98% positive**. The felt signal genuinely flipped. Every behavioral fingerprint
below is the same in all three runs.

**Eat rate rises WITH energy (anti-hunger), exposure- and opportunity-normalized.**
Eats per 100k awake robot-ticks, restricted to samples with bushes within
radius 8 (near_bushes > 0):

| run | E 20–40 | E 40–60 | E 60–80 | E 80–100 | median E at eat |
|---|---|---|---|---|---|
| anima_02 | 1.8 | 16.6 | 40.1 | 142.3 | 97.4 |
| anima_02_thrive50 | 2.2 | 12.8 | 38.4 | 137.1 | 98.7 |
| inverted_thrive50 | 3.3 | 35.1 | 57.4 | 118.5 | 98.2 |

Eating-while-hungry (E<40) over the whole runs: **27 / 28 / 43 events** vs
2025 / 1431 / 779 sated. A hungry agent standing next to food eats at ~1/70th
the rate of a sated one.

**Eating is camping + spawn luck, not policy.** 37–39% of inter-eat intervals
are <500 ticks (bursts on a bush clump; median interval 885–1065). Per-robot
eat rate is uncorrelated with `gene_comfort_gain` (r = −0.04 / +0.02 / −0.18)
and inconsistently with `gene_restlessness` (r = −0.02 / −0.30 / +0.07),
n = 174/99/107 robots with ≥50k ticks observed.

**The dominant life is wake → starve → re-hibernate.** After a wake event,
**81% / 75% / 80%** of wakes re-hibernate before a single eat; median wake→eat
for the lucky minority is 789 / 842 / 652 ticks. The awake corridor
(wake_energy 40 → brownout 25 at ~0.0065/tick drain) is only ~2.3k ticks.
Awake exposure sits 57–73% below E 40.

**Mortality and population.** Deaths 169 / 96 / 95; dominant last-ledger cause
hibernation (107 / 51 / 75) then poison (62 / 45 / 19); median death age 258k /
238k / 298k (~2.9 / 2.6 / 2.0 senescence half-lives — they age out dormant).
Population collapsed 42 → 7 / 5 / 16 plastic vs floor 4 (inverted's 16 is the
150k-halflife confound). Dormant fraction now: 0.71 / 1.00 / 0.81.

**The budding/selection channel starved.** Buds 66 / 42 / 57 vs spawns
(mostly floor respawns) 267 / 129 / 142; thriving-now 1 / 0 / 0 — even at
thrive_energy 50, because thriving requires being *awake*. Gene drift: anima_02
(6.9M ticks) shows real directional signal (viability_gain +176%, comfort_gain
+49%) but the inverted run's drift is ≈0 at 2.3M — with floor-respawn churn
dominating reproduction, selection is weak and slow.

**Modulator budget (awake point-samples).** |m_comfort| mean 0.003, p99 0.02;
|m_viability| mean 0.012–0.026, p99 0.18–0.34 (≈10× louder); M clip is 5.
Mean awake M is **negative** in every run (−0.020 / −0.006 / −0.016);
life_return_via −33 / −11 / −18 vs comfort's ~−1.

## Interpretation

Four stacked blockers, each measured, jointly sufficient to explain hunger-deafness:

1. **The teaching event is unreachable.** A hungry meal *would* spike
   m_comfort ≈ +0.9 (E 0.40→0.80 through the Keramati–Gutkin drive at gain 3) —
   perfectly audible. It happened 27–43 times per run. A
   reinforce-what-occurred rule cannot reinforce behavior that doesn't occur;
   the 2.3k-tick awake corridor makes the occurrence a lottery.
2. **The engram decays before hunger recurs.** W_fast decay 1e-3/act-step ⇒
   memory half-life ~3.5k ticks; the sated→hungry drain cycle is ~10k+. Even a
   perfectly credited meal is forgotten before it matters. tau 20 act-steps =
   100 ticks of eligibility credits only the final EAT grip, never the approach.
3. **Awake life is net-punished and the recovery is structurally unfeelable.**
   The viability reduction fires negative through the slow decline toward
   brownout (felt, anti-consolidating whatever the starving agent was doing —
   including food-seeking), but the recovery happens inside dormancy and
   `wake()` resets stream *and trace* (scheduler.py `_pending_wake` →
   `brain.wake()`), so the positive half multiplies a zero trace and can never
   consolidate anything. This confirms anima_001's P5 suspicion at scale, and
   it is beta_10's replay-subsidy asymmetry reborn in valence-gating form: the
   modulator budget reads "everything you do awake, do less of; dormancy is
   free."
4. **Nothing learned is heritable and selection is starved.** Darwinian
   inherit reinits W_fast; budding (the only selection channel) requires being
   awake and thriving, which the attractor forecloses.

On the round's registered question: finite felt time + earned reproduction did
**not** evolve a survival instinct at these settings — not because selection
was wrong in principle (anima_02's gene drift shows directional signal where
lifetimes were long enough) but because the phenotype selection needed
(eat-when-hungry) could not be expressed by any lineage: the body physics
(corridor), the plasticity constants (decay/tau), and the gate asymmetry made
it unlearnable within a life, and floor respawns diluted what little
differential reproduction existed.

Observation vs inference: the eat-rate gradient, the wake→re-hibernate rates,
the modulator magnitudes, and the inversion null are measured. The four-blocker
causal account is inference — consistent with all of it, but the package test
(anima_03) is what falsifies it.

## Caveats

- The inverted arm's world config differed (senescence 150k vs 90k): its
  population/lifespan numbers are not comparable to the other two arms; only
  its eating structure and life_return signs are used here, and those are
  robust to the confound.
- The inversion probe was off-books and its code is deleted (never committed);
  the save dir remains. Sign flip applied to m_comfort only.
- anima_02 ran ~3.7× longer than the other arms; per-window rates are used
  where it matters.
- `sated` in earlier gol-stats readouts uses E≥40 at eat; wake_energy is
  exactly 40, so post-wake eats land "sated" by construction. The
  exposure-normalized table above is the honest version.

## Next

- **anima_03 (staged): the audibility round.** Remove all four blockers as a
  package and ask whether hunger becomes audible at all: world wake_energy
  40→55 (corridor ~2.3k→4.6k ticks), brain appetite_gain 2.0 (drive scales
  restlessness innately; coupling strength is a new heritable gene),
  plasticity decay 1e-4 + tau 60, viability.rectified true (credit escapes
  only). thrive_energy is 65, NOT the thrive50 arm's 50: the thrive check is
  instantaneous, so any bar ≤ wake_energy makes waking itself eligibility —
  hibernators would bud ~3 free children each (integrity bleeds ~9.5/cycle,
  never repairable below E 60) and reproduction would subsidize the attractor,
  beta_10's replay-subsidy pattern in the budding channel. 65 requires one
  post-wake meal, matches repair_threshold 60, and 65 − 40 bud cost = 25 lands
  exactly at brownout. Frozen control arm (plasticity OFF, appetite
  ON) isolates what W_fast adds; the bar is plastic > frozen on
  eating-while-hungry. If the package fails with every measured blocker gone,
  that is a strong negative verdict on the three-factor Hebbian family as
  configured.
- Deferred: Lamarckian `inherit_mode: lineage` (built, flagged) once there is
  within-life learning worth inheriting; blocker attribution by ablation IF
  anima_03 works.
- Housekeeping: journal reorganized into per-track subdirs (`beta/`,
  `anima/`); anima_001 (anima_01 first sitting) entry is still unwritten —
  its findings live in the anima track memory and proposal 002.
