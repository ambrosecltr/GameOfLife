import numpy as np
from gol_world.blocks import Block
from gol_world.config import TerrainConfig, WorldConfig
from gol_world.terrain import generate

SMALL = WorldConfig(seed=42, size=(64, 64, 48))


def test_same_seed_is_identical() -> None:
    a = generate(SMALL)
    b = generate(SMALL)
    assert np.array_equal(a.blocks, b.blocks)


def test_different_seed_differs() -> None:
    a = generate(SMALL)
    b = generate(WorldConfig(seed=43, size=(64, 64, 48)))
    assert not np.array_equal(a.blocks, b.blocks)


def test_structure() -> None:
    grid = generate(SMALL)
    blocks = grid.blocks
    # Bedrock floor everywhere.
    assert (blocks[:, :, 0] == Block.BEDROCK).all()
    # Sky above every column.
    assert (blocks[:, :, -1] == Block.AIR).all()
    # The essential block types all occur.
    for kind in (Block.ROCK, Block.SOIL, Block.GRASS, Block.WATER, Block.BUSH_RIPE, Block.ORE):
        assert (blocks == kind).any(), f"no {kind.name} generated"


def test_bushes_sit_on_grass() -> None:
    grid = generate(SMALL)
    for x, y, z in np.argwhere(
        (grid.blocks == Block.BUSH_RIPE) | (grid.blocks == Block.BUSH_EMPTY)
    ):
        assert grid.blocks[x, y, z - 1] == Block.GRASS


def test_no_floating_water() -> None:
    grid = generate(SMALL)
    water_level = TerrainConfig().water_level
    xs, ys, zs = np.nonzero(grid.blocks == Block.WATER)
    assert (zs <= water_level).all()
