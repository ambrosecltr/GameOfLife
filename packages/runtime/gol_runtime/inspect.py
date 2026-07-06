"""gol-stats: analyze a save dir's metrics and events.

gol-stats saves/alpha              # summary: population, lifespans, eats
gol-stats saves/alpha --events     # event counts by kind and robot
gol-stats saves/alpha --compare    # learning sanity probe: dreamers vs
                                   # scripted baselines (eat rates, prediction
                                   # error trend). Observational, not a benchmark.
gol-stats saves/alpha --interests  # per-agent activity profiles over time
                                   # windows: do agents differ (individuality)
                                   # and persist (interests, not noise)?
gol-stats saves/alpha --circadian  # does rest track the day/night cycle?
                                   # awake-resting vs light correlation, per
                                   # brain kind (dormancy reported separately
                                   # — hibernation is forced, not chosen)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(save_dir: Path) -> dict[str, Any]:
    events = _read_ndjson(save_dir / "events.ndjson")
    metrics = _read_ndjson(save_dir / "metrics.ndjson")

    by_kind = Counter(e["kind"] for e in events)
    eats_by_robot = Counter(e["robot"] for e in events if e["kind"] == "eat")
    deaths = [e for e in events if e["kind"] == "death"]
    spawns = {e["robot"]: e for e in events if e["kind"] == "spawn"}

    per_brain: dict[str, dict[str, float]] = defaultdict(lambda: {"eats": 0, "deaths": 0, "n": 0})
    for rid, spawn in spawns.items():
        brain = spawn.get("brain", "?")
        per_brain[brain]["n"] += 1
        per_brain[brain]["eats"] += eats_by_robot.get(rid, 0)
    for death in deaths:
        brain = spawns.get(death["robot"], {}).get("brain", "?")
        per_brain[brain]["deaths"] += 1

    last = metrics[-1] if metrics else {}
    return {
        "last_tick": last.get("tick", 0),
        "population": last.get("population", 0),
        "ripe_bushes": last.get("ripe_bushes", 0),
        "events": dict(by_kind),
        "per_brain": {k: dict(v) for k, v in per_brain.items()},
        "living": {
            rid: {"energy": r["energy"], "age": r["age"], "brain": r["brain"]}
            for rid, r in last.get("robots", {}).items()
        },
    }


def compare(save_dir: Path) -> dict[str, Any]:
    """Learning sanity probe: is the dreamer lineage pulling ahead of chance?

    Observational, from the world's own logs — not a benchmark and not a task.
    """
    events = _read_ndjson(save_dir / "events.ndjson")
    metrics = _read_ndjson(save_dir / "metrics.ndjson")
    spawns = {e["robot"]: e for e in events if e["kind"] == "spawn"}
    last_tick = metrics[-1]["tick"] if metrics else 0

    # Eat rate per 10k ticks of embodied lifetime, per brain kind.
    alive_ticks: dict[str, float] = defaultdict(float)
    eats: dict[str, int] = defaultdict(int)
    seen: dict[str, int] = {}
    for m in metrics:
        for rid, r in m.get("robots", {}).items():
            kind = r.get("brain", "?")
            prev = seen.get(rid, m["tick"])
            alive_ticks[kind] += m["tick"] - prev
            seen[rid] = m["tick"]
    for e in events:
        if e["kind"] == "eat":
            kind = spawns.get(e["robot"], {}).get("brain", "?")
            eats[kind] += 1
    eat_rate = {
        kind: round(eats[kind] / ticks * 10_000, 3)
        for kind, ticks in alive_ticks.items()
        if ticks > 0
    }

    # Dreamer prediction-error trend: mean over first vs last quartile of samples.
    per_metric: dict[str, list[float]] = defaultdict(list)
    for m in metrics:
        for _rid, bm in m.get("brains", {}).items():
            for key in ("pred_error_depth", "curiosity", "loss_model"):
                if key in bm:
                    per_metric[key].append(bm[key])
    trends = {}
    for key, series in per_metric.items():
        if len(series) >= 8:
            q = len(series) // 4
            trends[key] = {
                "first_quartile_mean": round(sum(series[:q]) / q, 4),
                "last_quartile_mean": round(sum(series[-q:]) / q, 4),
            }

    return {"last_tick": last_tick, "eat_rate_per_10k_ticks": eat_rate, "dreamer_trends": trends}


# The interest profile: how an agent's time is allocated within a window.
# Fractions of metrics samples (rest/social/forage/dormant) plus event counts
# per sample (eat/dig/place), so all coordinates share a comparable scale.
PROFILE_KEYS = ("rest", "social", "forage", "dormant", "eat", "dig", "place")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return dot / (na * nb)


def interests(save_dir: Path, window_ticks: int) -> dict[str, Any]:
    """Emergence observables for the gratification work, from the logs alone.

    Individuality = between-agent divergence of activity profiles (mean
    pairwise L1) growing over windows; an *interest* = within-agent stability
    (cosine of consecutive windows) beating noise. Without these two numbers,
    temperament tuning is blind.
    """
    metrics = _read_ndjson(save_dir / "metrics.ndjson")
    events = _read_ndjson(save_dir / "events.ndjson")

    counts: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for m in metrics:
        w = m["tick"] // window_ticks
        for rid, r in m.get("robots", {}).items():
            if "near_robots" not in r:  # save predates interest logging
                continue
            acc = counts[rid].setdefault(w, dict.fromkeys(("n", *PROFILE_KEYS), 0.0))
            acc["n"] += 1
            acc["rest"] += float(r.get("resting", False))
            acc["social"] += float(r["near_robots"] > 0)
            acc["forage"] += float(r["near_bushes"] > 0)
            acc["dormant"] += float(r.get("dormant", False))
    for e in events:
        if e["kind"] in ("eat", "dig", "place") and "robot" in e:
            w = e["tick"] // window_ticks
            if w in counts.get(e["robot"], {}):
                counts[e["robot"]][w][e["kind"]] += 1.0

    profiles: dict[str, dict[int, list[float]]] = {
        rid: {
            w: [acc[k] / acc["n"] for k in PROFILE_KEYS]
            for w, acc in windows.items()
            if acc["n"] >= 5  # too few samples = noise, not a profile
        }
        for rid, windows in counts.items()
    }
    profiles = {rid: wins for rid, wins in profiles.items() if wins}

    brains = {
        e["robot"]: e.get("brain", "?") for e in events if e["kind"] == "spawn" and "robot" in e
    }
    per_agent = {}
    for rid, wins in sorted(profiles.items()):
        ordered = [wins[w] for w in sorted(wins)]
        stability = None
        if len(ordered) >= 2:
            sims = [_cosine(a, b) for a, b in zip(ordered, ordered[1:], strict=False)]
            stability = round(sum(sims) / len(sims), 4)
        per_agent[rid] = {
            "brain": brains.get(rid, "?"),
            "windows": len(ordered),
            "profile": {k: round(v, 4) for k, v in zip(PROFILE_KEYS, ordered[-1], strict=True)},
            "stability": stability,
        }

    divergence = []
    for w in sorted({w for wins in profiles.values() for w in wins}):
        cohort = [wins[w] for wins in profiles.values() if w in wins]
        if len(cohort) < 2:
            continue
        dists = [
            sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(PROFILE_KEYS)
            for i, a in enumerate(cohort)
            for b in cohort[i + 1 :]
        ]
        divergence.append(
            {
                "from_tick": w * window_ticks,
                "agents": len(cohort),
                "divergence": round(sum(dists) / len(dists), 4),
            }
        )

    return {"window_ticks": window_ticks, "per_agent": per_agent, "divergence": divergence}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return None
    return cov / math.sqrt(vx * vy)


def circadian(save_dir: Path, since_tick: int = 0) -> dict[str, Any]:
    """Rest-vs-light structure, per brain kind.

    The H2 question ("do bodies find a rhythm?") needs *chosen* rest split
    from forced dormancy: awake_resting is the low-motor state while awake,
    dormancy is hibernation. A negative rest-light correlation = sleeps at
    night; beta_07 measured the inverse (+0.12: rests by day, likely riding
    the solar trickle), which is why this is a standard readout now.
    """
    metrics = _read_ndjson(save_dir / "metrics.ndjson")
    # kind -> (light, awake_resting) / (light, dormant) sample pairs
    rest: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    dorm: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for m in metrics:
        if m["tick"] < since_tick:
            continue
        light = float(m.get("light", 0.0))
        for r in m.get("robots", {}).values():
            if "resting" not in r:  # save predates rest logging
                continue
            kind = r.get("brain", "?")
            dorm[kind][0].append(light)
            dorm[kind][1].append(float(r.get("dormant", False)))
            if not r.get("dormant", False):
                rest[kind][0].append(light)
                rest[kind][1].append(float(r["resting"]))

    def _split(lights: list[float], flags: list[float]) -> dict[str, float | int | None]:
        day = [f for li, f in zip(lights, flags, strict=True) if li > 0.5]
        night = [f for li, f in zip(lights, flags, strict=True) if li <= 0.5]
        corr = _pearson(lights, flags)
        return {
            "samples": len(flags),
            "corr_with_light": None if corr is None else round(corr, 4),
            "day_frac": round(sum(day) / len(day), 4) if day else None,
            "night_frac": round(sum(night) / len(night), 4) if night else None,
        }

    return {
        "since_tick": since_tick,
        "awake_resting": {kind: _split(li, fl) for kind, (li, fl) in sorted(rest.items())},
        "dormant": {kind: _split(li, fl) for kind, (li, fl) in sorted(dorm.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gol-stats", description=__doc__)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--events", action="store_true", help="dump event kind counts only")
    parser.add_argument("--compare", action="store_true", help="learning sanity probe")
    parser.add_argument("--interests", action="store_true", help="per-agent activity profiles")
    parser.add_argument(
        "--window", type=int, default=50_000, help="interest-profile window (ticks)"
    )
    parser.add_argument("--circadian", action="store_true", help="rest-vs-light structure")
    parser.add_argument(
        "--since", type=int, default=0, help="circadian: ignore samples before this tick"
    )
    args = parser.parse_args(argv)

    if not (args.save_dir / "manifest.json").exists():
        print(f"error: {args.save_dir} is not a save dir", file=sys.stderr)
        return 1
    if args.compare:
        print(json.dumps(compare(args.save_dir), indent=2))
        return 0
    if args.interests:
        print(json.dumps(interests(args.save_dir, args.window), indent=2))
        return 0
    if args.circadian:
        print(json.dumps(circadian(args.save_dir, args.since), indent=2))
        return 0
    summary = summarize(args.save_dir)
    if args.events:
        print(json.dumps(summary["events"], indent=2))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
