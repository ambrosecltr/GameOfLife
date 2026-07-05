# Run Journal

Findings from long runs. One entry per notable run: save name, config, duration,
what happened, what it suggests. Newest first.

## 2026-07-05 — build-out soaks (local, M1 Pro)

**Economy calibration (100k ticks ≈ 4 sim-days, 5 foragers + 3 walkers).**
Foragers thrive indefinitely (energy 94–99 at age 100k); walkers starve within
a sim-day and churn through hibernate → death → scrap → respawn (9 walker
deaths). The two calibration changes that made this work: bushes must generate
in *clumps* (a single 1-block bush slips between the 9°-spaced sensor rays and
is invisible at range), and costs softened to basal 0.0015 / move 0.005 / eat 40.

**First mind (40k ticks, 1 nano dreamer + 7 scripted, lineage on).**
Model loss 115 → 83, ray-depth prediction error 0.46 → 0.16, ensemble
disagreement (curiosity) 0.118 → 0.079 — the model is learning its world and
the world is becoming less surprising, which is exactly the expected signature.
The original body (bot_000) starved during motor babbling and died; its mind
continued in bot_008 with the loss curve unbroken across the death. This run
motivated two permanent changes: warmup cut 2000 → 500 act-steps (newborns
were starving mid-babble), and `inherit_weights: lineage` as the default
(a lineage learns even though bodies die).

**Device benchmark (M1 Pro).** nano: cpu 474 ms/update vs mps 568 (cpu wins,
and act is 0.6 ms vs 9.1). small: mps 655 vs cpu 1009 (mps wins). Policy:
learning brains live on `devices.learning`; local nano default is cpu.

*(No cloud runs yet — first 4090 soak should target the social-curiosity
experiment: `configs/run/exp_social_curiosity.yaml` and its masked twin.)*
