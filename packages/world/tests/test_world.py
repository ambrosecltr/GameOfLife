import numpy as np
from gol_world.blocks import Block
from gol_world.config import EcologyConfig, WorldConfig
from gol_world.world import World

CFG = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=1000)


def test_day_night_cycle() -> None:
    world = World.new(CFG)
    assert world.tick == 0
    lights = []
    for _ in range(CFG.day_length_ticks):
        world.step()
        lights.append(world.light_level)
    arr = np.array(lights)
    assert arr.max() == 1.0 and arr.min() == 0.0
    # Roughly half the day is fully lit, half fully dark.
    assert 0.3 < (arr == 1.0).mean() < 0.6
    assert 0.3 < (arr == 0.0).mean() < 0.6


def test_regrowth_flips_bushes_back() -> None:
    world = World.new(CFG)
    empties = np.argwhere(world.grid.blocks == Block.BUSH_EMPTY)
    assert len(empties) > 0, "generation should leave some depleted bushes"
    # Initial regrowth is due within the first day, but bushes landing at
    # night are deferred to just after the next dawn (+ up to 2000 jitter).
    for _ in range(CFG.day_length_ticks + 2001):
        world.step()
    assert (world.grid.blocks == Block.BUSH_EMPTY).sum() == 0


def test_regrowth_waits_for_daytime() -> None:
    cfg = WorldConfig(
        seed=5,
        size=(48, 48, 40),
        day_length_ticks=1000,
        ecology=EcologyConfig(regrow_ticks=10, regrow_jitter=0, regrow_daytime_only=True),
    )
    world = World.new(cfg)
    # Jump to night (sun_height < 0 in the second half of the day).
    world.tick = 600
    x, y, z = map(int, np.argwhere(world.grid.blocks == Block.BUSH_EMPTY)[0])
    world.schedule_regrow(x, y, z)
    for _ in range(50):
        world.step()
    assert world.grid.get_block(x, y, z) == Block.BUSH_EMPTY, "must not regrow at night"
    # By some point after the next dawn (tick 1000 + jitter < 2000), it regrew.
    while world.tick < 3000 and world.grid.get_block(x, y, z) == Block.BUSH_EMPTY:
        world.step()
    assert world.grid.get_block(x, y, z) == Block.BUSH_RIPE


def test_step_never_resets() -> None:
    world = World.new(CFG)
    ticks = [world.tick]
    for _ in range(100):
        world.step()
        ticks.append(world.tick)
    assert ticks == list(range(101))
