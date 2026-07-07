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
    assert set(world.robots) == {"walker_000", "walker_001", "walker_002"}

    # Starve one robot to death.
    world.robots["walker_001"].energy = 0.0
    world.robots["walker_001"].integrity = 0.5

    loop = SimLoop(world, save, RUN, act_step=population.act_step)
    loop.run(max_ticks=300, paced=False)

    assert "walker_001" not in world.robots
    assert "walker_003" in world.robots, "a replacement should have spawned"
    assert len(world.robots) == 3
    assert "walker_003" in population.brains
    assert "walker_001" not in population.brains


def test_wake_from_dormancy_calls_wake_hook() -> None:
    world = World.new(WorldConfig(seed=21, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, RUN)
    rid = "walker_000"
    calls: list[str] = []
    population.brains[rid].wake = lambda: calls.append(rid)  # type: ignore[method-assign]

    population.act_step(world)  # awake baseline: no wake
    assert calls == []
    robot = world.robots[rid]
    robot.dormant = True
    population.act_step(world)  # dormant: unobserved, still no wake
    assert calls == []
    robot.dormant = False
    robot.energy = 50.0
    population.act_step(world)  # first awake cycle after the gap
    assert calls == [rid]
    population.act_step(world)  # wake fires once, not every cycle
    assert calls == [rid]


def test_default_wake_is_a_stream_cut() -> None:
    """Brains that don't override wake() keep the legacy reset semantics."""
    world = World.new(WorldConfig(seed=21, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, RUN)
    rid = "walker_000"
    resets: list[str] = []
    population.brains[rid].reset_stream = lambda: resets.append(rid)  # type: ignore[method-assign]
    robot = world.robots[rid]
    robot.dormant = True
    population.act_step(world)
    robot.dormant = False
    robot.energy = 50.0
    population.act_step(world)
    assert resets == [rid]
