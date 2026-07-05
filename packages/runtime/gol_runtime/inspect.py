"""gol-stats: analyze a save dir's metrics and events.

gol-stats saves/alpha              # summary: population, lifespans, eats
gol-stats saves/alpha --events     # event counts by kind and robot
gol-stats saves/alpha --compare    # learning sanity probe: dreamers vs
                                   # scripted baselines (eat rates, prediction
                                   # error trend). Observational, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gol-stats", description=__doc__)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--events", action="store_true", help="dump event kind counts only")
    parser.add_argument("--compare", action="store_true", help="learning sanity probe")
    args = parser.parse_args(argv)

    if not (args.save_dir / "manifest.json").exists():
        print(f"error: {args.save_dir} is not a save dir", file=sys.stderr)
        return 1
    if args.compare:
        print(json.dumps(compare(args.save_dir), indent=2))
        return 0
    summary = summarize(args.save_dir)
    if args.events:
        print(json.dumps(summary["events"], indent=2))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
