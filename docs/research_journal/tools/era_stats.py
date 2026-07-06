"""Era-windowed run analysis for research-journal entries.

Complements `gol-stats` (whole-run summaries) with per-era trend tables:
eats/day per brain, awake fraction, learner telemetry, bush stocks, and
lifespans by death-order cohort. Windows default to 500k ticks (~20.8 sim-days).

Usage: python3 docs/research_journal/tools/era_stats.py saves/<name> [saves/<other> ...]
"""

import json
import sys
from collections import defaultdict

DAY = 24000
ERA = 500_000

BRAIN_FIELDS = [
    "curiosity", "curiosity_scaled", "reward_homeostasis", "entropy",
    "loss_actor", "loss_model", "pred_error_depth", "value",
]


def analyze(save: str) -> None:
    # events: eats/poisonings per era per brain, deaths per brain in order
    eats: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    poisons: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    deaths: dict[str, list[tuple[int, float, dict]]] = defaultdict(list)
    brain_of: dict[str, str] = {}
    arm: dict[str, int] = defaultdict(int)  # concurrent slots per arm = tick-0 spawns
    with open(f"{save}/events.ndjson") as f:
        for line in f:
            e = json.loads(line)
            kind = e["kind"]
            if kind == "spawn":
                brain_of[e["robot"]] = e["brain"]
                if e["tick"] == 0:
                    arm[e["brain"]] += 1
            elif kind == "eat":
                eats[e["tick"] // ERA][brain_of.get(e["robot"], "?")] += 1
            elif kind == "poisoned":
                poisons[e["tick"] // ERA][brain_of.get(e["robot"], "?")] += 1
            elif kind == "death":
                b = brain_of.get(e["robot"], "?")
                deaths[b].append((e["tick"], e["age_ticks"] / DAY, e.get("ledger", {})))

    # metrics: per-era learner stats, dreamer awake fraction, bush stocks
    acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    last_tick = 0
    with open(f"{save}/metrics.ndjson") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # runs killed mid-write leave a truncated last line
            last_tick = max(last_tick, d["tick"])
            era = d["tick"] // ERA
            for key in ("ripe_bushes", "toxic_bushes", "empty_bushes"):
                if key in d:
                    acc[era][key].append(d[key])
            for rv in d.get("robots", {}).values():
                if rv.get("brain") == "dreamer":
                    acc[era]["awake"].append(0.0 if rv.get("dormant") else 1.0)
            for bv in d.get("brains", {}).values():
                for fld in BRAIN_FIELDS:
                    if fld in bv:
                        acc[era][fld].append(bv[fld])

    print(f"\n===== {save} (last_tick {last_tick:,} ≈ {last_tick / DAY:.0f} sim-days) =====")
    days = ERA / DAY
    brains = sorted(arm)
    header = (
        ["era"] + [f"{b}_eats/day" for b in brains] + [f"{b}_poison" for b in brains]
        + ["awake"] + BRAIN_FIELDS + ["ripe", "toxic", "empty"]
    )
    print("\t".join(header))
    for era in range(last_tick // ERA + 1):
        row = [str(era)]
        row += [f"{eats[era].get(b, 0) / days / max(1, arm[b]):.2f}" for b in brains]
        row += [str(poisons[era].get(b, 0)) for b in brains]
        for fld in ["awake"] + BRAIN_FIELDS:
            v = acc[era][fld]
            row.append(f"{sum(v) / len(v):.4f}" if v else "")
        for fld in ("ripe_bushes", "toxic_bushes", "empty_bushes"):
            v = acc[era][fld]
            row.append(f"{sum(v) / len(v):.0f}" if v else "")
        print("\t".join(row))

    for b, dl in sorted(deaths.items()):
        dl.sort()
        ages = [a for _, a, _ in dl]
        third = max(1, len(ages) // 3)
        cohorts = [ages[:third], ages[third:2 * third], ages[2 * third:]]
        meds = [f"{sorted(c)[len(c) // 2]:.1f}" if c else "-" for c in cohorts]
        causes: dict[str, int] = defaultdict(int)
        for _, _, led in dl:
            if led:
                cause = max(((k, v) for k, v in led.items() if k != "repaired"),
                            key=lambda kv: kv[1])[0]
                causes[cause] += 1
        print(f"{b}: n={len(ages)} lifespan medians by death-order thirds (days): "
              f"{' -> '.join(meds)}  causes={dict(causes)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for save_dir in sys.argv[1:]:
        analyze(save_dir)
