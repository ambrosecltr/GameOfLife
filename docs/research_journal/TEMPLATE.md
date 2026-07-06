---
round: NNN
title: short noun phrase
date: YYYY-MM-DD            # when the round closed (or was written up)
status: planned | running | complete
question: one line — what this round was designed to find out
headline: one sentence — the finding you'd want recalled a month later; write it last
runs:
  - save: saves/<name>      # or "not retained"
    config: configs/run/<file>.yaml
    brain: configs/brain/<file>.yaml
    commit: <7-char sha>
    ticks: <last tick>      # ≈ ticks/24000 sim-days
    role: experiment | control | precursor
baselines: [NNN, ...]       # prior rounds this compares against; note confounds in body
tags: [ecology, motivation, capacity, vision, ...]
---

# NNN — title

## Why this round

What prior finding or hypothesis motivated it. Link prior rounds by number.

## What changed

Bulleted: each change, the config key or commit that carries it, and *why*. This is the
diff between this round and its baseline — anything not listed here is assumed identical.

## Results

The data. Era tables, totals, per-brain splits, lifespans, death ledgers. Include the
numbers inline — save dirs get pruned; the entry is the durable record. Standard source:
`gol-stats <save>` / `--compare`; era tables: `tools/era_stats.py`.

## Interpretation

What the data means. Separate observation from inference.

## Caveats

Confounds, config mistakes, variance warnings, anything that limits what the round can
claim. Be blunt — the next reader plans experiments off this section.

## Next

Follow-ups this round spawned: fixes committed, the experiment the findings call for,
open questions.
