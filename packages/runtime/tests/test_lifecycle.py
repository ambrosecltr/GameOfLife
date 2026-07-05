"""Population lifecycle: deaths lead to delayed respawns with fresh brains."""

import dataclasses
from pathlib import Path

from gol_runtime.config import PopulationConfig, RunConfig
from gol_runtime.loop import SimLoop
from gol_runtime.scheduler import Population
from gol_world import persistence
from gol_world.config import EconomyConfig, WorldConfig
from gol_world.world import World

CFG = WorldConfig(
    seed=21,
    size=(64, 64, 40),
    day_length_ticks=1000,
    economy=EconomyConfig(hibernate_integrity_drain=2.0),  # dormancy kills fast
)
RUN = RunConfig(
    checkpoint_interval_ticks=100_000,
    population=PopulationConfig(
        target=3,
        respawn_delay_ticks=50,
        mix=({"brain": {"kind": "random_walker"}, "count": 3},),
    ),
)


def test_death_triggers_respawn(tmp_path: Path) -> None:
    save = tmp_path / "save"
    persistence.create_save(save, CFG, run_config=dataclasses.asdict(RUN))
    world = World.new(CFG)
    population = Population(world, RUN)
    assert set(world.robots) == {"bot_000", "bot_001", "bot_002"}

    # Starve one robot to death.
    world.robots["bot_001"].energy = 0.0
    world.robots["bot_001"].integrity = 0.5

    loop = SimLoop(world, save, RUN, act_step=population.act_step)
    loop.run(max_ticks=300, paced=False)

    assert "bot_001" not in world.robots
    assert "bot_003" in world.robots, "a replacement should have spawned"
    assert len(world.robots) == 3
    assert "bot_003" in population.brains
    assert "bot_001" not in population.brains
