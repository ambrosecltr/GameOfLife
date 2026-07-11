"""RunLogs metrics sampling: the interest-profile fields ride every record."""

import json
from pathlib import Path

from gol_obs.heatmap import VisitHeatmap
from gol_obs.logs import RunLogs
from gol_world.config import WorldConfig
from gol_world.world import World


def test_metrics_carry_interest_fields(tmp_path: Path) -> None:
    world = World.new(WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000))
    a = world.spawn_robot("bot_000", "test")
    b = world.spawn_robot("bot_001", "test")
    b.pos = a.pos.copy()  # right next to each other: near_robots must see it
    logs = RunLogs(
        tmp_path,
        metrics_every_ticks=1,
        runtime_metrics=lambda: {"precision": "amp_bf16", "safe_ticks_per_second": 42.0},
    )
    world.step()
    logs.on_tick(world)
    logs.close()

    record = json.loads((tmp_path / "metrics.ndjson").read_text().splitlines()[-1])
    bot = record["robots"]["bot_000"]
    assert bot["near_robots"] == 1
    assert bot["near_bushes"] >= 0
    assert bot["resting"] is True  # no drive commanded
    assert bot["signal"] == [0.0, 0.0]
    assert bot["signal_magnitude"] == 0.0
    assert record["runtime"] == {
        "precision": "amp_bf16",
        "safe_ticks_per_second": 42.0,
    }


def test_bulk_stationary_heatmap_matches_per_tick_accounting() -> None:
    world = World.new(WorldConfig(seed=6, size=(32, 32, 40), day_length_ticks=1000))
    world.spawn_robot("bot_000", "test")
    ordinary = VisitHeatmap(world.cfg.size)
    accelerated = VisitHeatmap(world.cfg.size)
    start = world.tick
    for tick in range(start + 1, start + 251):
        world.tick = tick
        ordinary.on_tick(world)
    accelerated.advance_stationary(world, start)
    assert (accelerated.grid == ordinary.grid).all()


def test_backpressure_hold_writes_an_explicit_runtime_sample(tmp_path: Path) -> None:
    world = World.new(WorldConfig(seed=7, size=(32, 32, 40), day_length_ticks=1000))
    logs = RunLogs(
        tmp_path,
        metrics_every_ticks=100,
        runtime_metrics=lambda: {"backpressure_reason": "causal_lag"},
    )

    logs.on_backpressure(world)
    logs.close()

    record = json.loads((tmp_path / "metrics.ndjson").read_text().splitlines()[-1])
    assert record["tick"] == 0
    assert record["backpressure_hold"] is True
    assert record["runtime"]["backpressure_reason"] == "causal_lag"
