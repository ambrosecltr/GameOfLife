# beta_07 — the gratification round: do agents develop interests?

*Written 2026-07-06, ahead of the run. Code is complete and verified but
deliberately uncommitted until beta_07 launches and its early effects are
measured. Companion entry for results goes in `research_journal/` (round 007) afterward; the
mechanism documentation lives in `architecture.md` (DreamerBrain section).*

## Where the run line stands

**beta_05 is the baseline** (`local_hunger.yaml` + `dreamer_hungry.yaml`:
low_energy_threshold 0.4, penalty 0.25, w_homeostasis 2.0; OBS v2 vision;
paused resumable at 3.45M ticks). It solved most of what the earlier soaks
surfaced: the homeostatic reward held at −0.08 instead of flattening,
lifespans stabilized, and the motivational decay softened (~1.1 eats/day in
the window where the pre-hunger run had collapsed to 0.65). What it did NOT
solve: the decay only slowed, and actor entropy *rose* 6.0 → 6.37 — the
policy couldn't exploit the hunger gradient it was given. Conclusion at the
time: partly a capacity problem (nano/CPU), partly a motivation problem —
louder hunger is a crutch, not a cure. The deeper diagnosis stands: the
level-of-surprise curiosity signal is a bootstrapping drive, not a lifelong
one, and a world you simply exist in becomes boring (research_journal rounds 004–005).

**beta_06 (just completed, under review)** was intended as beta_05's
carry-on with OBS v3 RGB vision — but was accidentally launched with
`local_social.yaml`, so it ran the *quiet legacy* reward
(`configs/brain/dreamer.yaml` as of launch: disagreement curiosity,
events homeostasis, w_homeostasis 1.0) rather than the hunger settings.
That loses the clean "vision effect on the hunger line" measurement — but
it buys something better for *this* round: **beta_06 is a same-config,
same-vision control for beta_07.** If beta_07 launches with
`local_social.yaml` + the new `dreamer.yaml`, the two runs differ in exactly
one thing: the reward machinery. No replacement beta_06 is needed before
beta_07; a hunger+v3 control run is the follow-up disambiguator only if
beta_07's attribution comes out ambiguous.

## Why this round exists

The gratification stack replaces "how surprising is this?" with the
machinery of wanting: need-relative pleasure (drive reduction), interest
(learning progress), restlessness (boredom), and innate individual
difference (temperament). It addresses the motivation half of beta_05's
verdict head-on, rather than turning the hunger volume knob further. Note
that the HRRL drive reward *structurally supersedes* the hunger experiment's
knobs — the level penalty is a permanent hunger gradient and meals spike
~0.6 on the reward, so beta_07 starts back at `w_homeostasis: 1.0` with
escalation to 2.0 (the beta_05 lesson) held in reserve if the early
checkpoints show homeostasis drowned again.

The design constraint throughout: **nothing object-specific is wired in.**
No "bushes feel good," no "robots are interesting." Valence is defined over
abstract internal state; what any individual comes to care about must emerge
from its own history.

## What changed (all landed 2026-07-06, one working tree)

1. **HRRL drive-reduction homeostasis** (`reward.homeostasis: drive`) —
   reward = movement of (energy, integrity, restedness) toward setpoints,
   convex so the neediest drive dominates. Eating while starving ≈ +0.6;
   the same meal near satiety ≈ +0.17; past the setpoint, nothing. Satiation
   with no stop rule; damage and fatigue price themselves.
2. **Rest as a circadian affordance** (world: `rest_basal_mult`,
   `night_rest_bonus`) — resting discounts basal drain; fatigue recovery and
   repair run up to 2× in darkness. Sleep never mints energy, it slows the
   meter. "Sleep at night" is discoverable, not scripted.
3. **Learning-progress curiosity** (`reward.curiosity: lp`,
   `dreamer/interest.py`) — reward is the *rate the world model's error
   falls* per region; regions are online k-means over the agent's own latent
   space. Mastered things and unlearnable things (noise, other minds) both go
   stale; interest lives at the learnable frontier and is path-dependent —
   the individuality mechanism. `mix_disagreement: 0.1` keeps newborns
   moving; `partition: kind` and `curiosity: disagreement` are ablations.
4. **Boredom** (`reward.boredom`) — a penalty only when drives are met AND
   stimulation is flat. Never bored while needy, never bored while learning.
   This is the pressure that should produce play.
5. **Temperament** (`temperament:`) — heritable log-normal multipliers over
   seven abstract knobs (curiosity/homeostasis weights, three drive weights,
   boredom, entropy). Sampled at birth, exact across checkpoints, mutated
   only on `random_living` inheritance.
6. **Interest observability** — metrics now carry per-robot
   `resting`/`near_robots`/`near_bushes`; `gol-stats <save> --interests`
   computes per-agent activity profiles, within-agent stability, and
   between-agent divergence. The measurement side of the whole bet.

## What this round is looking for

- **H1 — motivation persists.** The legacy signature (curiosity collapsing
  as the world becomes predictable, reward landscape flattening, eats/day
  and awake fraction sagging in a stable world — softened but not cured in
  beta_05) should not reproduce. `stimulation` may decline but
  `drive_level` + boredom should keep the landscape from flattening.
  Success: late-generation dreamers whose eats/day and awake fraction hold
  where beta_05's still slipped and beta_06's (presumably, review pending)
  decayed faster on the quiet reward.
- **H2 — bodies find rhythm.** `drive_level` should oscillate (need → act →
  sate) rather than trend; resting fraction should develop *negative*
  correlation with `light` (sleeping at night) without any rule saying so.
  Watch actor entropy too: beta_05's rose while the policy failed to exploit
  its gradient — falling or stable entropy here means the drive gradients
  are actually being used.
- **H3 — interests, not noise.** In `--interests` (suggest `--window
  100000`, ~4 sim-days): dreamer within-agent stability should rise with age
  from the newborn ~0.5 toward (but plausibly below) the scripted-forager
  ~0.97 ceiling; between-agent divergence among dreamers should *grow*
  across windows. Divergence flat-at-zero = no individuality; stability at
  chance = restlessness without interests.
- **H4 — temperament shows through (weak test only).** With `lineage`
  inheritance there are just 3 dreamer temperament draws, fixed for the whole
  run. Look for coarse correlation between draw and profile (the high
  `temperament_drive_rest` lineage resting more, the high `w_curiosity` one
  ranging wider). n=3 is anecdote, not evidence — the real temperament
  experiment needs `random_living` or reproduction, later.

## Issues to look out for

- **Dithering.** LP + boredom both push "move on"; if `ema_fast/ema_slow`
  are too close to the behavior timescale, agents oscillate between niches
  mastering none. Symptom: `--interests` stability stuck near zero while
  `lp_reward` stays high. Knobs: lower both EMA rates, lower boredom weight.
- **Self-injury as stimulation.** The reward-hack where damaging yourself
  regenerates learnable signal / drive to reduce. The drive design nets this
  negative (creating a deficit costs what fixing it earns, plus the level
  penalty), but verify empirically: watch for `boredom` > 0 periods followed
  by fall/poison events; correlate damage ledgers with low `drive_level`.
- **Chronic boredom (couch-lock inverse).** If `stim_threshold: 0.5` is set
  too high relative to real normalized stimulation, boredom becomes a
  constant tax and just depresses everything. `boredom` metric should be ~0
  for newborns (it was, in the smoke run) and only occasionally positive
  later. If it's persistently > ~0.01, lower `stim_threshold`.
- **Hunger drowned again (the beta_05 lesson, top priority check).**
  Compare magnitudes of `reward_homeostasis` (drive spikes ~0.6 on meals,
  small negative drift between) against `stimulation` (sustained 0–5). If
  homeostasis is orders of magnitude quieter, raise `w_homeostasis` toward
  2.0 or `drive.scale` — but let the run speak first; boredom's drive-gate
  means satiety now has a voice the legacy reward lacked.
- **LP normalization warm-up.** `curiosity_scaled` pinned at the 5.0 clamp
  early (seen in smoke runs) is the RunningMeanStd warming up — fine for the
  first ~100k ticks, a red flag after.
- **Region pathologies.** `lp_regions` should reach 32 quickly and stay
  there. If LP reward variance goes to zero while model loss still falls,
  the partition may be too coarse/fine — `lp.regions` is the knob. (Gap: no
  per-region occupancy metric yet; add one if this needs diagnosing.)
- **Interests forgotten via replay turnover.** The 500k-step FIFO buffer
  erodes unpracticed interests (value estimates decay with their data).
  Life-like, but if divergence rises then collapses, this is a suspect —
  prioritized retention of high-LP sequences is the future fix.
- **Perf floor.** `test_perf_floor_16_agents` reads ~371 ticks/s with a beta
  run hogging two cores; the unchanged baseline reads 392 under the same
  load (stash A/B) — environmental, not a regression. Rerun once on the
  idle machine before launching (`uv run pytest -m slow -k perf`).

## Comparisons

Two baselines, each answering a different question:

- **beta_06 (same config, same v3 vision, legacy reward)** — the A/B for
  "what does the gratification stack do?" Every difference between beta_07
  and beta_06 is attributable to the reward machinery. Ask the beta_06
  reviewing agent to extract, per dreamer generation: eats/day, awake
  fraction, median lifespan, curiosity trajectory, world-model loss trend,
  actor entropy — those become the left column of this comparison.
- **beta_05 (hunger baseline, v2 vision, paused at 3.45M)** — the A/B for
  "did the motivational decay finally stop?" Vision is a confound here, so
  compare trajectory *shapes*, not levels: beta_05's decay was slowed-but-
  present (eats/day ~1.1 and slipping), homeostatic reward −0.08 flat,
  entropy rising. beta_07 should show oscillating (not flat) homeostasis,
  entropy not rising, and eats/day holding.
- **Foragers are the cross-run anchor.** Fixed policy, so their eats/day and
  lifespans should match beta_06 within noise — the world barely changed for
  them (only the small rest/night economy additions). If forager numbers
  shift materially, the *world* moved more than intended; investigate before
  crediting or blaming the reward stack. Note `--interests` works on beta_07
  only; older saves lack the fields (the tool degrades gracefully).

New signals that exist only in beta_07: `drive_level`, `lp_reward`,
`lp_regions`, `boredom`, `stimulation`, `temperament_*`, and the
`--interests` divergence/stability numbers.

If, after beta_07, the vision-vs-stack attribution is still murky (e.g.
beta_07 improves over beta_05 but beta_06's review shows v3 vision alone
already helped a lot), the tie-breaker is the originally-intended run:
`local_hunger.yaml` + v3 vision, i.e. the "replacement beta_06" — run it
then, not now.

## Starting the run

Preflight (all green as of writing, but re-verify on the idle machine):

    uv run pytest -m "not slow" -q && uv run ruff check . && uv run mypy packages
    uv run pytest -m slow -q        # perf floor + smoke, now without contention

Launch — deliberately `local_social.yaml` again, NOT `local_hunger.yaml`,
so beta_06 is a clean A/B (the hunger knobs live in events-mode config that
drive mode supersedes anyway; `dreamer_hungry.yaml` is legacy until further
notice). The new stack rides `configs/brain/dreamer.yaml`, already switched
on — no `--set` needed:

    PYTHONUNBUFFERED=1 caffeinate -is uv run gol-run --new saves/beta_07 \
        --config configs/run/local_social.yaml --headless

Monitoring cadence (a `gol-ctl checkpoint` then one SIGINT stops it cleanly —
never a second SIGINT, it aborts the final checkpoint write):

    uv run gol-stats saves/beta_07                       # population pulse
    uv run gol-stats saves/beta_07 --compare             # learning trend
    uv run gol-stats saves/beta_07 --interests --window 100000
    grep -o '"dreamer_[0-9]*":{[^}]*}' saves/beta_07/metrics.ndjson | tail -3
    # watch: drive_level, lp_reward, lp_regions, boredom, stimulation, temperament_*

Rules for this round:

- **Do not resume beta_05/beta_06 under the new configs.** Save manifests
  reference brain configs by *path*; resume re-reads the file, so an old save
  resumed now silently switches reward machinery mid-life. They stay frozen
  as baselines (beta_05 additionally is the hunger line's resumable head —
  resuming it later requires restoring its original config semantics first).
- Keep `inherit_weights: lineage` for beta_07 (clean A/B vs beta_06;
  temperament fixed per lineage). The temperament-selection experiment
  (`random_living`, mutation active) is its own future round — don't mix the
  two questions.
- Every new mechanism has an off switch for follow-up ablations:
  `reward.curiosity: disagreement`, `reward.homeostasis: events`,
  `boredom.weight: 0`, `temperament.enabled: false`, `lp.partition: kind`,
  world-side `night_rest_bonus: 0` / `rest_basal_mult: 1.0`.
- First checkpoint sanity check (~30 min in): confirm `lp_regions: 32`,
  `boredom` ≈ 0, three distinct `temperament_*` sets across the dreamers,
  `resting`/`near_*` fields in metrics, and — the beta_05 lesson —
  `reward_homeostasis` within an order of magnitude of `stimulation`. All
  mechanical checks were verified in the pre-launch scratchpad smoke (7.5k
  ticks, checkpoint + resume included); the magnitude check needs real run
  data.
