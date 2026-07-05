"""World-hash determinism: same seed + same brains => identical evolution."""

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
from gol_runtime.config import PopulationConfig, RunConfig
from gol_runtime.loop import SimLoop
from gol_runtime.scheduler import Population
from gol_world import persistence
from gol_world.config import WorldConfig
from gol_world.world import World

CFG = WorldConfig(seed=11, size=(64, 64, 40), day_length_ticks=800)
RUN = RunConfig(
    checkpoint_interval_ticks=10_000,
    population=PopulationConfig(mix=({"brain": {"kind": "random_walker"}, "count": 3},)),
)


def world_hash(world: World) -> str:
    h = hashlib.sha256()
    h.update(world.grid.blocks.tobytes())
    h.update(str(world.tick).encode())
    for robot in world.robots.values():
        h.update(robot.id.encode())
        h.update(np.round(robot.pos, 9).tobytes())
        h.update(np.float64(round(robot.yaw, 9)).tobytes())
        h.update(np.float64(robot.energy).tobytes())
    return h.hexdigest()


def run_fresh(tmp_path: Path, name: str, ticks: int) -> str:
    save = tmp_path / name
    persistence.create_save(save, CFG, run_config=dataclasses.asdict(RUN))
    world = World.new(CFG)
    population = Population(world, RUN)
    loop = SimLoop(world, save, RUN, act_step=population.act_step)
    loop.run(max_ticks=ticks, paced=False)
    return world_hash(world)


def test_same_seed_same_world(tmp_path: Path) -> None:
    assert run_fresh(tmp_path, "a", 2000) == run_fresh(tmp_path, "b", 2000)


def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    # One continuous 1200-tick run...
    continuous = run_fresh(tmp_path, "cont", 1200)

    # ...must equal 700 ticks, checkpoint, resume in a new process-life, 500 more.
    save = tmp_path / "split"
    persistence.create_save(save, CFG, run_config=dataclasses.asdict(RUN))
    world = World.new(CFG)
    population = Population(world, RUN)
    loop = SimLoop(
        world, save, RUN, brain_states=population.brain_states, act_step=population.act_step
    )
    loop.run(max_ticks=700, paced=False)

    resumed = persistence.load_world(save)
    population2 = Population(resumed, RUN)
    ckpt = persistence.latest_checkpoint(save)
    assert ckpt is not None
    population2.restore_brain_states(persistence.load_brain_states(ckpt))
    loop2 = SimLoop(
        resumed, save, RUN, brain_states=population2.brain_states, act_step=population2.act_step
    )
    loop2.run(max_ticks=500, paced=False)

    assert world_hash(resumed) == continuous
