# Research Journal

One entry per **research round**: a deliberate change to the world/brains, the runs that
tested it, what the data showed, and what it implies for the next round. This folder
replaces the old `docs/journal.md` (migrated 2026-07-06).

## Why this format

The primary reader is the assistant analyzing the *next* round. Entries are optimized for
that: machine-scannable frontmatter (grep the `headline:` lines to recall every finding
without opening a file), self-contained statistics (save dirs get pruned after a round
closes — brains/checkpoints deleted, so the numbers in the entry are the durable record),
and explicit lineage (`baselines:` says which prior rounds a comparison is valid against,
and under what confounds).

## Rules

- One file per round, named `NNN-slug.md`. Lexicographic order = chronological order.
- Frontmatter follows [TEMPLATE.md](TEMPLATE.md). `headline:` is the one-sentence finding —
  write it last, make it the thing you'd want recalled a month later.
- Every quantitative claim in the body should carry its number. Assume the save dir may be
  gone; era tables and totals live in the entry, not just in `metrics.ndjson`.
- Record confounds and config mistakes honestly (see round 006 — a wrong config became the
  control arm). A confounded run is still data if the entry says exactly what it confounds.
- Standard stats come from `uv run gol-stats <save> [--compare|--events|--interests]`.
  Era-windowed tables (500k-tick windows) come from [tools/era_stats.py](tools/era_stats.py):
  `python3 docs/research_journal/tools/era_stats.py saves/<name> ...`
- When a round closes: fill `status: complete`, write the headline, add the index row here,
  and note the spawned follow-ups in the entry's **Next** section.

## Index (newest first)

| round | date | runs | headline |
|---|---|---|---|
| [011 — the reachability round](011-reachability.md) | 2026-07-07/08 | beta_10 | Reachability exonerated: the reward head learned the loud moments (spike err 0.25→0.06) and the dormancy crash was priced honestly into replay — and of 105 dreamer meals exactly 1 was eaten hungry. The binge fired earlier and stronger than 009's and collapsed the same way; 23/24 deaths stayed on the hibernation clock. Capacity (008), conditioning (009), reachability (011): three clean exonerations. Binding constraint moves to the actor/affordances fork — and the census discovery that replay had been *paying* the hibernation attractor (+1.2/wake, +3.9/rebirth) reframes all prior rounds. Bonus: first behaviorally-caused dreamer death (fall), longest life ever (a forager, 837k, dead of poison not the clock), and scripted-forager intake proved spawn-luck-dominated — affordances are patchy even for a perfect policy. |
| [009 — the conditioning round](009-conditioning.md) | 2026-07-07 | beta_09 | Conditioning worked and boredom is a thermostat, not a ratchet: anchored normalizers killed the treadmill (curiosity flat under a frozen anchor vs 008's 20× re-inflation), pressure charged to a self-limiting ~0.39 equilibrium, and it drove the first within-run rise in dreamer eating ever (5→14→9→5 per 200k) — a purposive binge that collapsed once its own success closed the gates and hunger had to carry. HRRL semantics are right but unreachable: the blackout is architecturally invisible (`reset_stream` on wake) and 55 meals/3M ticks starves the reward head. Binding constraint moves to reward reachability. Bonus: boredom also drove terraforming (still dreamer-only), and the whole population once slept through 200k ticks in sync. |
| [008 — the capacity round](008-capacity.md) | 2026-07-06/07 | beta_08 | Capacity was necessary but not sufficient: the model converged (loss 29→4.2), curiosity finally decayed, and boredom fired for the first time in the series — but it only flickered (≤1.6e-3), the running-std normalizer re-inflated curiosity 20× against the decay, and dreamer eating collapsed instead of rising (92 meals, 38% toxic, vs foragers' 5026 at 0.6%). Binding constraint moved from capacity to signal conditioning: normalization + boredom accumulator. Bonus: dreamers are the world's only terraformers (318 digs/287 places vs 0). |
| [007 — the gratification round](007-gratification.md) | 2026-07-06 | beta_07 | Individuality arrived before survival competence: the three lineages developed distinct, persistent behavioral profiles and actor entropy fell for the first time in the series — but relative LP + an unconverged model held curiosity at ~3 forever, so homeostasis stayed ~500× quieter, boredom never fired, and eating didn't improve. The stack's balance needs a converging world model, i.e. capacity (beta_08). |
| [006 — obs v3 and the capacity wall](006-obs-v3-capacity-wall.md) | 2026-07-06 | beta_06, beta_06h | Richer senses tripled scripted foraging but sank the learners: the hunger effect of round 005 did not replicate under obs v3 — the critic learned the future is hungry (value went negative) and the policy still couldn't act on it. Binding constraint confirmed as learning capacity, not reward design. |
| [005 — the hunger experiment](005-hunger-experiment.md) | 2026-07-05 | beta_05 | A louder body softened the motivational decay but couldn't stop it: hunger reward held steady and lifespans stabilized, but the policy never learned to cash the gradient in (~1 meal/day in replay is too sparse at nano/CPU capacity). |
| [004 — competence killed motivation](004-curiosity-collapse.md) | 2026-07-05 | beta_04 | Cross-lifetime learning works (model loss 84→19 across generations) — and that's the problem: curiosity collapsed 20× as the world became predictable, homeostasis was ~1000× too quiet to take over, and behavior decayed to aimless wandering in a food-rich world. |
| [003 — bush ecology and the toxic ratchet](003-bush-ecology.md) | 2026-07-05 | beta_03 | First emergent ecology: lifespans became behavior-dependent, but the regrow rule was a one-way toxic ratchet — the better the population avoided poison, the more poisoned the world became. Plants effectively evolved defenses under grazing pressure. |
| [002 — survival pressures, broken bodies](002-survival-pressures.md) | 2026-07-05 | beta_02 (beta_01 precursor) | The wake economy was a death ratchet (wake below brownout) and the calibration probe was wedged from spawn (eat resolved only at eye height) — when the probe fails you can't distinguish "economy too harsh" from "bodies broken." |
| [001 — build-out soaks](001-buildout-soaks.md) | 2026-07-05 | (pre-beta, saves not retained) | Economy calibrated (bush clumps + softened costs), first mind confirmed lifelong learning across a body death (lineage inheritance), devices benchmarked (nano→CPU on M1). |
