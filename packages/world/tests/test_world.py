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


def _bush_count(world: World) -> int:
    from gol_world.world import BUSH_BLOCKS

    return int(np.isin(world.grid.blocks, BUSH_BLOCKS).sum())


def _bush_sites(world: World) -> set[tuple[int, int, int]]:
    from gol_world.world import BUSH_BLOCKS

    return {
        (int(p[0]), int(p[1]), int(p[2]))
        for p in np.argwhere(np.isin(world.grid.blocks, BUSH_BLOCKS))
    }


def test_wither_conserves_stock_and_drifts_the_food_map() -> None:
    cfg = WorldConfig(
        seed=5,
        size=(48, 48, 40),
        day_length_ticks=1000,
        ecology=EcologyConfig(
            regrow_ticks=10,
            regrow_jitter=0,
            regrow_daytime_only=False,
            bush_lifespan_ticks=50,
            bush_lifespan_jitter=0,
        ),
    )
    world = World.new(cfg)
    start = _bush_count(world)
    original_sites = _bush_sites(world)
    assert start > 0
    # standing + queued replacements is invariant, tick by tick
    for _ in range(400):
        world.step()
        assert _bush_count(world) + len(world.sprout_heap) == start
    kinds = {e["kind"] for e in world.consume_events()}
    assert "wither" in kinds and "sprout" in kinds
    # ~8 lifespans in: the bushes live somewhere else now
    assert _bush_sites(world) != original_sites


def test_wither_unwinds_a_toxic_ratchet() -> None:
    cfg = WorldConfig(
        seed=5,
        size=(48, 48, 40),
        day_length_ticks=1000,
        ecology=EcologyConfig(
            regrow_ticks=10,
            regrow_jitter=0,
            regrow_daytime_only=False,
            toxic_fraction=0.0,
            bush_lifespan_ticks=50,
            bush_lifespan_jitter=0,
        ),
    )
    world = World.new(cfg)
    # Force a poisoned world: every standing bush turns toxic.
    for x, y, z in _bush_sites(world):
        world.grid.set_block(x, y, z, Block.BUSH_TOXIC)
    assert (world.grid.blocks == Block.BUSH_TOXIC).sum() > 0
    for _ in range(400):  # several lifespans; replacements re-roll toxicity (0%)
        world.step()
    assert (world.grid.blocks == Block.BUSH_TOXIC).sum() == 0
    assert (world.grid.blocks == Block.BUSH_RIPE).sum() > 0


def test_bush_lifespan_zero_disables_withering() -> None:
    cfg = WorldConfig(
        seed=5,
        size=(48, 48, 40),
        day_length_ticks=1000,
        ecology=EcologyConfig(bush_lifespan_ticks=0),
    )
    world = World.new(cfg)
    assert world.wither_heap == []
    sites = _bush_sites(world)
    for _ in range(500):
        world.step()
    assert sites <= _bush_sites(world)  # nothing withered (regrowth may add)
    assert not any(e["kind"] == "wither" for e in world.consume_events())


def test_step_never_resets() -> None:
    world = World.new(CFG)
    ticks = [world.tick]
    for _ in range(100):
        world.step()
        ticks.append(world.tick)
    assert ticks == list(range(101))
