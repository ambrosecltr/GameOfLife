"""Deterministic terrain generation.

Value-noise fBm heightmap in pure numpy: layered rock/soil/grass, sand shores,
water-filled basins, ore pockets inside rock, and food bushes on grass. Same
seed => byte-identical world (tested).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gol_world.blocks import Block
from gol_world.config import WorldConfig
from gol_world.grid import VoxelGrid

FloatMap = npt.NDArray[np.float64]


def _value_noise(rng: np.random.Generator, shape: tuple[int, int], cells: int) -> FloatMap:
    """Smooth noise in [0, 1]: random lattice values, bilinearly interpolated."""
    lattice = rng.random((cells + 1, cells + 1))
    xs = np.linspace(0, cells, shape[0], endpoint=False)
    ys = np.linspace(0, cells, shape[1], endpoint=False)
    x0 = xs.astype(np.int64)
    y0 = ys.astype(np.int64)
    fx = (xs - x0)[:, None]
    fy = (ys - y0)[None, :]
    # Smoothstep the interpolants to avoid lattice-aligned creases.
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    v00 = lattice[np.ix_(x0, y0)]
    v10 = lattice[np.ix_(x0 + 1, y0)]
    v01 = lattice[np.ix_(x0, y0 + 1)]
    v11 = lattice[np.ix_(x0 + 1, y0 + 1)]
    top = v00 * (1 - fx) + v10 * fx
    bottom = v01 * (1 - fx) + v11 * fx
    result: FloatMap = top * (1 - fy) + bottom * fy
    return result


def _fbm(rng: np.random.Generator, shape: tuple[int, int], octaves: int) -> FloatMap:
    """Fractal sum of value noise, normalized to [0, 1]."""
    total = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    cells = 4
    norm = 0.0
    for _ in range(octaves):
        total += amplitude * _value_noise(rng, shape, cells)
        norm += amplitude
        amplitude *= 0.5
        cells *= 2
    result: FloatMap = total / norm
    return result


def generate(cfg: WorldConfig) -> VoxelGrid:
    rng = np.random.default_rng(cfg.seed)
    sx, sy, sz = cfg.size
    t = cfg.terrain

    noise = _fbm(rng, (sx, sy), t.octaves)
    heights = np.clip(
        np.round(t.height_base + (noise - 0.5) * 2 * t.height_amp).astype(np.int64),
        2,
        sz - 8,
    )

    blocks = np.zeros((sx, sy, sz), dtype=np.uint8)
    zs = np.arange(sz)[None, None, :]
    surface = heights[:, :, None]

    blocks[zs <= surface] = Block.ROCK
    blocks[(zs > surface - 4) & (zs <= surface)] = Block.SOIL
    blocks[zs == surface] = Block.GRASS
    blocks[:, :, 0] = Block.BEDROCK

    # Water fills columns whose surface sits below the water line; shores are sand.
    water = (zs > surface) & (zs <= t.water_level)
    blocks[water] = Block.WATER
    underwater = heights < t.water_level
    shore = ~underwater & (heights <= t.water_level + 1)
    surf_idx = (np.arange(sx)[:, None], np.arange(sy)[None, :], heights)
    blocks[surf_idx] = np.where(
        underwater, Block.SAND, np.where(shore, Block.SAND, blocks[surf_idx])
    )

    # Ore pockets: small random blobs strictly inside rock.
    for _ in range(t.ore_pockets):
        cx = int(rng.integers(2, sx - 2))
        cy = int(rng.integers(2, sy - 2))
        top = int(heights[cx, cy]) - 5
        if top <= 2:
            continue
        cz = int(rng.integers(2, top))
        for _ in range(t.ore_pocket_size):
            ox, oy, oz = (
                cx + int(rng.integers(-1, 2)),
                cy + int(rng.integers(-1, 2)),
                cz + int(rng.integers(-1, 2)),
            )
            if 0 <= ox < sx and 0 <= oy < sy and 1 <= oz < sz and blocks[ox, oy, oz] == Block.ROCK:
                blocks[ox, oy, oz] = Block.ORE

    # Bushes on grass. A fraction start depleted so regrowth is visible from tick 0.
    grass_x, grass_y = np.nonzero(blocks[surf_idx] == Block.GRASS)
    count = int(len(grass_x) * t.bush_density)
    if count and len(grass_x):
        picks = rng.choice(len(grass_x), size=min(count, len(grass_x)), replace=False)
        for i in picks:
            bx, by = int(grass_x[i]), int(grass_y[i])
            bz = int(heights[bx, by]) + 1
            if bz < sz:
                ripe = rng.random() > 0.3
                blocks[bx, by, bz] = Block.BUSH_RIPE if ripe else Block.BUSH_EMPTY

    return VoxelGrid(blocks)
