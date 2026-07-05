# Run Journal

Findings from long runs. One entry per notable run: save name, config, duration,
what happened, what it suggests. Newest first.

## 2026-07-05 — beta soaks: the toxic ratchet, and a world you merely exist in becomes boring

**beta_02 (229k ticks, local_social: 3 dreamers + 3 foragers).** The wake economy
was a death ratchet: `wake_energy` (15) sat *below* the brownout threshold (25),
so robots woke pre-crippled, paying full commanded cost for reduced motion, with
~1000 ticks of budget before collapsing again. Worse, the scripted foragers — the
economy's calibration probe — were wedged from spawn: `GRIP_EAT` resolved targets
only at eye height, so a bush one block downhill (plainly visible to the −30° ray
row) could never be eaten. All three foragers stood in front of food, spamming
eat at exactly basal drain, until integrity death at ~209k. Lesson: when the
calibration probe fails, you cannot distinguish "economy too harsh" from "bodies
broken." Fixes: `_faced_edible` (gaze scan ±1 block vertically), a forager eat
stall-breaker, wake at 40 with faster solar, and the wear/repair/senescence
economy (repair funded by energy surplus, efficiency halving per
`senescence_halflife`; per-robot integrity ledger for death-cause attribution).

**beta_03 (10.5M ticks ≈ 437 sim-days, 189 deaths).** First emergent ecology
result. Lifespans became behavior-dependent (foragers ~16 days, dreamers ~10–12,
death ledgers cleanly attributing poison vs hibernation vs wear) — but the
commons quietly degraded: ripe bushes 290 → ~115 while toxic climbed 45 → ~210.
The mechanism is a one-way ratchet in the regrow rule: an eaten ripe bush regrows
toxic 15% of the time, but a toxic bush stays toxic until *someone eats it*. The
better the population avoids poison, the more poisoned the world becomes; the
only recycling agents were desperate or still-learning dreamers (374 poisonings).
Plants effectively evolved defenses under grazing pressure. Also observed:
dreamers dig up bushes and carry them (proto-provisioning) — and hand-eaten
bushes never scheduled a regrow, permanently deleting food sites. Fixes: bush
senescence (`bush_lifespan_ticks` ≈ 5 sim-days; withered bushes are replaced by
sprouts with toxicity *re-rolled*, biased toward existing patches by
`sprout_clump_bias`), a conserved bush budget (standing + held + queued sprouts
is invariant; hand-eaten, spoiled, and died-carrying bushes all return their
slot), and `held_spoil_ticks` so carried food perishes. Placement/caching/feeding
deliberately kept alive for the cultural-transmission questions. beta_03 is
preserved as the "before" dataset.

**beta_04 (2.2M+ ticks ≈ 93 sim-days, 33 deaths, still running at write-up).**
The ecology fixes hold: standing bush stock oscillates 374–461 with no drift
across 93 sim-days, withers ≈ sprouts (7.5k each), toxic share breathes between
12–23% around its 15% baseline instead of ratcheting. Foragers (fixed policy —
a true control) eat 7–47/day and live 16–22 days with heavy repair use (one
repaired 74 integrity through four poisonings). Then the real finding: **the
dreamers are learning to predict and forgetting to live.** Across generations
(lineage inheritance carrying weights and buffer), world-model loss fell 84 →
36 → 19 and prediction error 4×'d down — cross-lifetime learning, plainly. But
curiosity (Plan2Explore disagreement, the dominant reward) collapsed 20× as the
world became predictable, while the homeostatic term (ate − damage − 0.02·low)
was always ~1000× smaller than early curiosity. The reward landscape flattened;
actor loss fell 17 → ~0.5; behavior drifted to aimless wandering. Dreamer
eats/day: 1.9 → 0.5. Awake fraction: 16% → 11%. Median lifespan: 14.4 → 13.0
days, in a *food-rich, stable* world — the decline is motivational, not
ecological. Competence killed motivation, and hunger was never loud enough to
take over. A world you simply exist in becomes boring.

What it suggests: pure curiosity is a bootstrapping drive, not a lifelong one.
The next run (beta_05) rebalances the homeostat so survival pressure remains a
first-class gradient once the world is learned: `low_energy_threshold` 0.25 →
0.4 (hunger felt before brownout), `low_energy_penalty` 0.02 → 0.25 (dense,
learnable, rivals residual curiosity), `w_homeostasis` 1 → 2 (meals and damage
count double). Curiosity is left untouched — the question is whether a louder
body sustains purposeful behavior after the mind has mastered its world.

**beta_05 (3.45M ticks ≈ 144 sim-days, local_hunger: dreamer_hungry.yaml).**
The louder body softened the decay but did not stop it. The homeostatic reward
held rock-steady at −0.07..−0.08 for the whole run (~25% of the reward stream
after curiosity faded — the flat-landscape failure of beta_04 is fixed), and in
the window where beta_04 collapsed (1.7M+), hungry dreamers ate 1.1/day vs the
baseline's 0.6–0.7, with lifespans *stabilizing* (14.0 → 14.4 → 14.5 days)
instead of declining. But the within-run trend still pointed down (final era:
0.6 eats/day, awake 11%), and the tell is actor entropy: it *rose* all run
(6.0 → 6.37). The hunger gradient is present; the policy cannot cash it in.
With meals this rare in replay (~1/day per agent), the reward head barely
learns what eating is worth, so imagination can't find the payoff — and a nano
model on CPU at train_ratio 0.25 gives each lineage only a few hundred updates
per lifetime to break that chicken-and-egg. What it suggests: motivation was
necessary but not sufficient; the binding constraint has moved from reward
design to learning capacity and experience density. That is the case for the
cloud round: small/base preset, higher train_ratio, same paired configs
(local_social vs local_hunger) so the motivation ablation carries over.
Paused at ckpt 3450000 (verified resumable) for later continuation.

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
