#!/usr/bin/env python
"""Human-readable progress report for a running or completed Aion save.

Summarizes runtime health, learning, survival, lineage continuity, temperament,
behavior, ecology, deaths, and checkpoint state from the append-only logs.

    uv run python scripts/aion_stats.py saves/aion_01_2gpu
    uv run python scripts/aion_stats.py saves/aion_01_2gpu --window 100000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

JsonObject = dict[str, Any]
TEMPERAMENT_KEYS = (
    "w_curiosity",
    "w_homeostasis",
    "drive_energy",
    "drive_integrity",
    "drive_rest",
    "boredom",
    "entropy",
)
ACTIVITY_EVENTS = ("eat", "dig", "place", "hibernate", "wake", "poisoned")


def _read_ndjson(path: Path) -> list[JsonObject]:
    if not path.exists():
        return []
    records = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error.msg}") from error
    return records


def _format_duration(ticks: int, tick_rate: int) -> str:
    seconds = ticks // max(1, tick_rate)
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.extend((f"{minutes}m", f"{seconds}s"))
    return " ".join(parts)


def _number(value: object, decimals: int = 2) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):,.{decimals}f}"


def _mean_metric(samples: list[JsonObject], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if key in sample]
    return mean(values) if values else None


def _trend(samples: list[JsonObject], key: str) -> tuple[float, float] | None:
    values = [float(sample[key]) for sample in samples if key in sample]
    if len(values) < 8:
        return None
    width = max(1, len(values) // 4)
    return mean(values[:width]), mean(values[-width:])


def _checkpoint_tick(save: Path) -> int | None:
    marker = save / "checkpoints" / "LATEST"
    if not marker.exists():
        return None
    name = marker.read_text().strip()
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _spawn_kinds(events: list[JsonObject]) -> dict[str, str]:
    return {
        str(event["robot"]): str(event.get("brain", "?"))
        for event in events
        if event.get("kind") == "spawn" and "robot" in event
    }


def build_report(save: Path, window_ticks: int) -> str:
    if window_ticks < 1:
        raise ValueError("window must be positive")
    manifest_path = save / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{save} is not a save directory")

    metrics = _read_ndjson(save / "metrics.ndjson")
    events = _read_ndjson(save / "events.ndjson")
    if not metrics:
        return f"{save}: no metrics have been recorded yet"

    manifest: JsonObject = json.loads(manifest_path.read_text())
    latest = metrics[-1]
    tick = int(latest["tick"])
    window_start = max(0, tick - window_ticks)
    recent_metrics = [row for row in metrics if int(row["tick"]) >= window_start]
    recent_events = [event for event in events if int(event["tick"]) >= window_start]
    run_config = manifest.get("run_config", {})
    tick_rate = int(run_config.get("tick_rate", 20))
    checkpoint_interval = int(run_config.get("checkpoint_interval_ticks", 0))
    max_debt = float(run_config.get("pacing", {}).get("max_debt_updates", math.inf))
    inheritance = str(run_config.get("population", {}).get("inherit_weights", "none"))
    reproduction = str(run_config.get("reproduction", {}).get("mode", "respawn"))
    spawn_kinds = _spawn_kinds(events)
    robots: JsonObject = latest.get("robots", {})
    brains: JsonObject = latest.get("brains", {})
    runtime: JsonObject = latest.get("runtime", {})
    aion_ids = sorted(rid for rid, robot in robots.items() if robot.get("brain") == "aion")
    aion_metric_samples = [
        values
        for row in metrics
        for rid, values in row.get("brains", {}).items()
        if spawn_kinds.get(rid, "aion" if rid.startswith("aion_") else "?") == "aion"
    ]

    event_counts = Counter(str(event.get("kind", "?")) for event in events)
    recent_event_counts = Counter(str(event.get("kind", "?")) for event in recent_events)
    aion_events = [
        event for event in events if spawn_kinds.get(str(event.get("robot", ""))) == "aion"
    ]
    aion_deaths = [event for event in aion_events if event.get("kind") == "death"]
    aion_spawns = [
        event
        for event in events
        if event.get("kind") == "spawn" and event.get("brain") == "aion"
    ]
    latest_checkpoint = _checkpoint_tick(save)

    warnings = []
    dropped_credit = float(runtime.get("dropped_update_credit", 0.0))
    learner_debt = float(runtime.get("max_learner_debt", 0.0))
    if dropped_credit > 0.0:
        warnings.append(f"ALERT: {dropped_credit:.2f} units of learning credit were dropped")
    if learner_debt >= max_debt:
        warnings.append(f"ALERT: learner debt {learner_debt:.2f} reached its {max_debt:.2f} limit")
    if latest_checkpoint is None:
        warnings.append("ALERT: no complete checkpoint exists")
    elif checkpoint_interval and tick - latest_checkpoint > checkpoint_interval:
        warnings.append(
            f"WATCH: latest checkpoint trails the log by {tick - latest_checkpoint:,} ticks"
        )
    if any(float(brains.get(rid, {}).get("act_latched_frac", 0.0)) > 0.0 for rid in aion_ids):
        warnings.append("WATCH: at least one Aion has latched actions")
    if tick >= 100_000 and not any(event.get("kind") == "eat" for event in aion_events):
        warnings.append("WATCH: no Aion has eaten yet")
    low_vitality = [
        rid
        for rid in aion_ids
        if float(robots[rid].get("energy", 0.0)) < 15.0
        or float(robots[rid].get("integrity", 0.0)) < 25.0
    ]
    if low_vitality:
        warnings.append(f"WATCH: low current vitality for {', '.join(low_vitality)}")

    lines = [
        "=" * 78,
        f"  AION CHECK-IN — {save.name}",
        f"  tick {tick:,}  |  simulated {_format_duration(tick, tick_rate)}  "
        f"|  recent window {window_ticks:,} ticks",
        "=" * 78,
        "",
        "HEALTH",
    ]
    if warnings:
        lines.extend(f"  {warning}" for warning in warnings)
    else:
        lines.append("  OK: no report-level warnings")
    recent_event_summary = "  ".join(
        f"{kind}={count}"
        for kind, count in recent_event_counts.most_common()
        if kind not in ("sprout", "wither")
    )
    lines.extend(
        [
            f"  runtime: {_number(runtime.get('actual_virtual_ticks_per_second'))} actual "
            f"/ {_number(runtime.get('safe_ticks_per_second'))} safe ticks/s",
            f"  learners: {_number(runtime.get('total_learner_updates_per_second'))} updates/s  "
            f"debt={_number(learner_debt)}  dropped={_number(dropped_credit)}  "
            f"precision={runtime.get('precision', 'n/a')}",
            f"  pacing: limiter={runtime.get('limiting_subsystem', 'n/a')}  "
            f"deadline_misses={int(float(runtime.get('inference_deadline_misses', 0))):,}",
            f"  checkpoint: {latest_checkpoint if latest_checkpoint is not None else 'none'}  "
            f"commit={manifest.get('git_commit', 'unknown')}",
            "",
            "LEARNING",
            "  lifetime trend (first quartile -> latest quartile)",
        ]
    )

    for key, label in (
        ("loss_model", "model loss"),
        ("pred_error_depth", "depth error"),
        ("pred_error_kind", "kind error"),
        ("curiosity", "curiosity"),
    ):
        change = _trend(aion_metric_samples, key)
        if change is not None:
            first, last = change
            direction = "down" if last < first else "up"
            lines.append(f"  {label:12s}: {first:,.4f} -> {last:,.4f} ({direction})")
    if not aion_ids:
        lines.append("  no embodied Aion is currently alive")
    for rid in aion_ids:
        brain = brains.get(rid, {})
        lines.append(
            f"  {rid}: updates={int(float(brain.get('updates', 0))):,}  "
            f"buffer={int(float(brain.get('buffer', 0))):,}  "
            f"loss={_number(brain.get('loss_model'), 3)}  "
            f"depth_err={_number(brain.get('pred_error_depth'), 4)}"
        )
        lines.append(
            f"    learn={_number(float(brain.get('learn_seconds', 0)) * 1000, 1)}ms  "
            f"action={_number(float(brain.get('action_seconds', 0)) * 1000, 1)}ms  "
            f"pending={_number(brain.get('pending_update_credit'))}  "
            f"inference_lag={_number(brain.get('inference_lag_updates'), 0)} updates"
        )

    lines.extend(["", "SURVIVAL & BODIES"])
    events_by_robot: dict[str, Counter[str]] = defaultdict(Counter)
    for event in aion_events:
        events_by_robot[str(event.get("robot", "?"))][str(event.get("kind", "?"))] += 1
    for rid in aion_ids:
        robot = robots[rid]
        state = "dormant" if robot.get("dormant") else "awake"
        counts = events_by_robot[rid]
        lines.append(
            f"  {rid}: {state:7s}  energy={_number(robot.get('energy'), 1)}  "
            f"integrity={_number(robot.get('integrity'), 1)}  age={int(robot.get('age', 0)):,}"
        )
        lines.append(
            f"    eats={counts['eat']}  hibernations={counts['hibernate']}  "
            f"wakes={counts['wake']}  poisonings={counts['poisoned']}"
        )
    lines.append(
        f"  Aion bodies: spawns={len(aion_spawns)}  deaths={len(aion_deaths)}  "
        f"inheritance={inheritance}  reproduction={reproduction}"
    )
    for death in aion_deaths[-3:]:
        ledger = death.get("ledger", {})
        harmful = {key: float(value) for key, value in ledger.items() if key != "repaired"}
        cause = max(harmful, key=lambda name: harmful[name]) if harmful else "unknown"
        lines.append(
            f"    death tick {int(death['tick']):,}: {death.get('robot', '?')}  "
            f"age={int(death.get('age_ticks', 0)):,}  dominant_damage={cause}"
        )

    lines.extend(["", "LINEAGE & TEMPERAMENT"])
    if inheritance == "lineage" and reproduction == "respawn":
        lines.append(
            "  The same Aion minds survive bodily death; this arm has fixed founder"
        )
        lines.append(
            "  temperaments, not genetic selection or generational gene drift."
        )
    elif reproduction == "budding":
        lines.append("  Budding is active; temperament mutation can create heritable drift.")
    else:
        lines.append(f"  inheritance={inheritance}; reproduction={reproduction}")
    for rid in aion_ids:
        brain = brains.get(rid, {})
        traits = [
            f"{key}={_number(brain.get(f'temperament_{key}'), 3)}" for key in TEMPERAMENT_KEYS
        ]
        lines.append(f"  {rid}: " + "  ".join(traits[:4]))
        lines.append(" " * (len(rid) + 4) + "  ".join(traits[4:]))

    lines.extend(["", f"BEHAVIOR — last {window_ticks:,} ticks"])
    recent_robot_samples: dict[str, list[JsonObject]] = defaultdict(list)
    for row in recent_metrics:
        for rid, robot in row.get("robots", {}).items():
            if spawn_kinds.get(rid, "aion" if rid.startswith("aion_") else "?") == "aion":
                recent_robot_samples[rid].append(robot)
    recent_aion_events = [
        event
        for event in recent_events
        if spawn_kinds.get(str(event.get("robot", ""))) == "aion"
    ]
    recent_by_robot: dict[str, Counter[str]] = defaultdict(Counter)
    for event in recent_aion_events:
        recent_by_robot[str(event.get("robot", "?"))][str(event.get("kind", "?"))] += 1
    for rid in aion_ids:
        samples = recent_robot_samples.get(rid, [])
        if not samples:
            continue
        dormant = mean(float(sample.get("dormant", False)) for sample in samples)
        awake_samples = [sample for sample in samples if not sample.get("dormant", False)]
        awake_rest = (
            mean(float(sample.get("resting", False)) for sample in awake_samples)
            if awake_samples
            else 0.0
        )
        social = mean(float(sample.get("near_robots", 0) > 0) for sample in samples)
        forage = mean(float(sample.get("near_bushes", 0) > 0) for sample in samples)
        counts = recent_by_robot[rid]
        lines.append(
            f"  {rid}: dormant={dormant:.1%}  awake-rest={awake_rest:.1%}  "
            f"near-peers={social:.1%}  near-food={forage:.1%}"
        )
        lines.append(
            "    " + "  ".join(f"{kind}={counts[kind]}" for kind in ACTIVITY_EVENTS)
        )

    lines.extend(
        [
            "",
            f"ECOLOGY & EVENTS — last {window_ticks:,} ticks",
            f"  bushes now: ripe={int(latest.get('ripe_bushes', 0)):,}  "
            f"toxic={int(latest.get('toxic_bushes', 0)):,}  "
            f"empty={int(latest.get('empty_bushes', 0)):,}",
            "  recent events: " + (recent_event_summary or "none"),
            f"  lifetime: eats={event_counts['eat']:,}  deaths={event_counts['death']:,}  "
            f"hibernations={event_counts['hibernate']:,}  wakes={event_counts['wake']:,}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("save", type=Path, help="save directory")
    parser.add_argument(
        "--window", type=int, default=50_000, help="recent behavior/event window in ticks"
    )
    args = parser.parse_args(argv)
    try:
        print(build_report(args.save, args.window))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
