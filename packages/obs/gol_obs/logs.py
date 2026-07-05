"""Append-only analysis logs: the tool-agnostic source of truth.

events.ndjson  — discrete happenings (spawn, eat, dig, place, death, ...)
metrics.ndjson — per-agent + world time series, sampled every N ticks

Both survive crashes (line-buffered appends) and are what gol-stats and any
future analysis notebooks read. Rerun charts are a *view*; these are the data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from gol_world.blocks import Block
from gol_world.world import World

from gol_obs.heatmap import VisitHeatmap

IntrospectionFn = Callable[[], dict[str, dict[str, float]]]
EventSink = Callable[[list[dict[str, Any]]], None]


class NdjsonWriter:
    def __init__(self, path: Path) -> None:
        self._fh: IO[str] = open(path, "a", buffering=1)  # noqa: SIM115 - long-lived handle

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._fh.close()


class RunLogs:
    """Drains world events every tick and samples metrics every N ticks."""

    def __init__(
        self,
        save_dir: Path,
        metrics_every_ticks: int,
        introspection: IntrospectionFn | None = None,
        heatmap: VisitHeatmap | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.events = NdjsonWriter(save_dir / "events.ndjson")
        self.metrics = NdjsonWriter(save_dir / "metrics.ndjson")
        self.metrics_every = metrics_every_ticks
        self.introspection = introspection
        self.heatmap = heatmap
        self.event_sink = event_sink

    def on_tick(self, world: World) -> None:
        events = world.consume_events()
        for event in events:
            self.events.write(event)
        if self.event_sink is not None and events:
            self.event_sink(events)
        if self.heatmap is not None:
            self.heatmap.on_tick(world)
        if world.tick % self.metrics_every == 0:
            self._sample(world)

    def _sample(self, world: World) -> None:
        record: dict[str, Any] = {
            "tick": world.tick,
            "light": round(world.light_level, 3),
            "population": len(world.robots),
            "ripe_bushes": int((world.grid.blocks == Block.BUSH_RIPE).sum()),
            "toxic_bushes": int((world.grid.blocks == Block.BUSH_TOXIC).sum()),
            "empty_bushes": int((world.grid.blocks == Block.BUSH_EMPTY).sum()),
            "robots": {
                r.id: {
                    "pos": [round(float(p), 1) for p in r.pos],
                    "energy": round(r.energy, 2),
                    "integrity": round(r.integrity, 2),
                    "fatigue": round(r.fatigue, 4),
                    "dormant": r.dormant,
                    "age": r.age_ticks,
                    "brain": r.brain_name,
                    "ledger": {k: round(v, 2) for k, v in r.ledger.items()},
                }
                for r in world.robots.values()
            },
        }
        if self.introspection is not None:
            brains = {
                rid: {k: round(v, 5) for k, v in m.items()}
                for rid, m in self.introspection().items()
                if m
            }
            if brains:
                record["brains"] = brains
        self.metrics.write(record)

    def close(self) -> None:
        self.events.close()
        self.metrics.close()
