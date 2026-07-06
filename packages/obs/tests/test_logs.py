"""RunLogs metrics sampling: the interest-profile fields ride every record."""

import json
from pathlib import Path

from gol_obs.logs import RunLogs
from gol_world.config import WorldConfig
from gol_world.world import World


def test_metrics_carry_interest_fields(tmp_path: Path) -> None:
    world = World.new(WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000))
    a = world.spawn_robot("bot_000", "test")
    b = world.spawn_robot("bot_001", "test")
    b.pos = a.pos.copy()  # right next to each other: near_robots must see it
    logs = RunLogs(tmp_path, metrics_every_ticks=1)
    world.step()
    logs.on_tick(world)
    logs.close()

    record = json.loads((tmp_path / "metrics.ndjson").read_text().splitlines()[-1])
    bot = record["robots"]["bot_000"]
    assert bot["near_robots"] == 1
    assert bot["near_bushes"] >= 0
    assert bot["resting"] is True  # no drive commanded
