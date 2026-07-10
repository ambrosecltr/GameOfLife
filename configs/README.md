# Configs

Layering: dataclass defaults → YAML file → `--set a.b.c=value` CLI overrides.
`run/` configs reference a `world/` config and per-population-slot `brain/` configs by
path.

## Naming convention (adopted 2026-07-06, effective from round 008)

Two kinds of files:

**Bases** — maintained templates, edited freely *between* rounds, not launched directly
once a round adopts them:

- `run/local_m1.yaml` — dev default (`gol-run` uses it when `--config` is omitted)
- `run/local_social.yaml`, `run/cloud_gpu.yaml` — the local / cloud tier templates
- `run/exp_*.yaml` — protocol templates for future research-question rounds
- `brain/dreamer.yaml` — the current default dreamer stack; `brain/dreamer_masked.yaml`
  (curiosity-mask ablation); `brain/dreamer_hungry.yaml` (legacy, frozen — beta_05's
  resumable head references it)
- `world/default.yaml`

**Round configs** — the launch artifact for one specific save, copied from a base and
prefixed with the save name it produces:

- `run/beta_NN[_slug].yaml` → launched as `saves/beta_NN[...]`
- `brain/beta_NN_dreamer[_slug].yaml` — referenced from that round's run config
- `world/beta_NN_world.yaml` — only if the round changes world parameters

Track prefixes name a brain-family and its founding bet, not the whole project. Journal
numbers are chronological within each track; save prefixes keep their established run
sequence even where an earlier cross-track round made the numbers differ:

- `beta_` — the world-model/Dreamer track at the base-preset capacity bundle (cloud)
- `swift_` — same Dreamer family, the efficiency bet: nano + the swift speed core at
  real train_ratio 1.0 on local hardware (adopted round 010)
- `anima_` — the completed plastic-valence family: world-model-free, critic-free,
  neuromodulated plasticity (closed after anima 007)

Rules:

1. **Config prefix = save dir name.** One run config per save; multi-arm rounds get one
   file per arm (`beta_09a.yaml`, `beta_09b.yaml`).
2. **Round configs freeze at launch.** The save manifest records config *paths*,
   `--resume` re-reads `--config` from disk, and the running scheduler re-reads brain
   configs on every respawn — renaming or editing a round config silently mutates (or
   crashes) a live or resumable world. A deliberate mid-run intervention is an edit to
   the round config, recorded in that round's journal entry.
3. **Always pass `--config` on resume.** It defaults to `local_m1.yaml`; resuming
   without it silently switches the run config.
4. **Diff against the base before launch.** The diff between a round config and its base
   is the experiment — keep it minimal and note it in the journal entry's "What changed."

## Pre-convention rounds (grandfathered — do not rename these bases while their saves are live or resumable)

Older rounds launched directly on base names; what they actually ran is pinned by the
`commit:` field in their journal entries, not by the filename:

| save | run config | brain config |
|---|---|---|
| beta_04 | `local_social.yaml` | `dreamer.yaml` (legacy reward, at that commit) |
| beta_05 (paused, resumable) | `local_hunger.yaml` | `dreamer_hungry.yaml` |
| beta_06 | `local_social.yaml` | `dreamer.yaml` (legacy reward, at 2202117) |
| beta_06h | `local_hunger.yaml` | `dreamer_hungry.yaml` (at 2202117) |
| beta_07 (running) | `local_social.yaml` | `dreamer.yaml` (gratification stack) |
| beta_08 (staged) | `beta_08_capacity.yaml` | `beta_08_dreamer.yaml` |

beta_07 predates the convention by hours: it launched on the base names, its manifest
references them, and its dreamers re-read `brain/dreamer.yaml` on every respawn — so the
base names stay untouched until beta_07 (and beta_05) are closed and pruned. This
ambiguity — "which dreamer.yaml did beta_04 run?" — is exactly what the convention
removes from round 008 onward.
