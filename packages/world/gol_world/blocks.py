"""Block palette and per-block property tables.

This module is the single source of truth for what blocks exist and how they
behave/appear. Property tables are numpy arrays indexed by block id so the sim
hot paths (physics, raycasting) and the renderer share one definition.
"""

from enum import IntEnum
from typing import cast

import numpy as np
import numpy.typing as npt


class Block(IntEnum):
    AIR = 0
    BEDROCK = 1
    ROCK = 2
    SOIL = 3
    GRASS = 4
    SAND = 5
    WATER = 6
    BUSH_EMPTY = 7
    BUSH_RIPE = 8
    ORE = 9
    SCRAP = 10
    BUSH_TOXIC = 11


NUM_BLOCKS = len(Block)

# Blocks a robot body cannot occupy (bushes and water are passable).
SOLID: npt.NDArray[np.bool_] = np.zeros(NUM_BLOCKS, dtype=np.bool_)
SOLID[[Block.BEDROCK, Block.ROCK, Block.SOIL, Block.GRASS, Block.SAND, Block.ORE, Block.SCRAP]] = (
    True
)

# Blocks the gripper can dig out of the world.
DIGGABLE: npt.NDArray[np.bool_] = np.zeros(NUM_BLOCKS, dtype=np.bool_)
DIGGABLE[
    [
        Block.ROCK,
        Block.SOIL,
        Block.GRASS,
        Block.SAND,
        Block.ORE,
        Block.SCRAP,
        Block.BUSH_EMPTY,
        Block.BUSH_RIPE,
        Block.BUSH_TOXIC,
    ]
] = True

# RGB, shared by every rendering surface.
COLOR: npt.NDArray[np.uint8] = np.array(
    [
        [0, 0, 0],  # AIR (never rendered)
        [40, 40, 46],  # BEDROCK
        [125, 125, 130],  # ROCK
        [121, 85, 58],  # SOIL
        [92, 158, 70],  # GRASS
        [216, 200, 152],  # SAND
        [58, 120, 200],  # WATER
        [60, 96, 52],  # BUSH_EMPTY
        [196, 74, 60],  # BUSH_RIPE (red berries)
        [186, 150, 62],  # ORE
        [210, 120, 34],  # SCRAP (robot-orange debris)
        [148, 70, 168],  # BUSH_TOXIC (purple berries)
    ],
    dtype=np.uint8,
)

assert COLOR.shape == (NUM_BLOCKS, 3)

# Fake directional light, shared by the mesher and ray sensing so the viewer
# and the agents see the same shaded world. Indexed [axis, sign>0].
FACE_SHADE: npt.NDArray[np.float32] = np.array(
    [[0.80, 0.80], [0.70, 0.70], [0.45, 1.00]], dtype=np.float32
)

# Nighttime shade floor: color dims toward this fraction as light_level -> 0.
AMBIENT_LIGHT = 0.08

# Sky, as seen by rays that hit nothing: interpolated by light_level, so dawn
# is visible as a brightening horizon. Float RGB in [0, 1].
SKY_DAY: npt.NDArray[np.float32] = np.array([128, 178, 230], dtype=np.float32) / 255.0
SKY_NIGHT: npt.NDArray[np.float32] = np.array([9, 11, 20], dtype=np.float32) / 255.0

# Per-voxel luminance grain: +/- this fraction around 1.0.
TINT_STRENGTH = 0.12


def light_factor(light_level: float) -> float:
    """Global shade multiplier for lit surfaces at a given light level."""
    return AMBIENT_LIGHT + (1.0 - AMBIENT_LIGHT) * light_level


def sky_color(light_level: float) -> npt.NDArray[np.float32]:
    """What a miss looks like: the sky between SKY_NIGHT and SKY_DAY."""
    blend = SKY_NIGHT + (SKY_DAY - SKY_NIGHT) * light_level
    return cast("npt.NDArray[np.float32]", np.asarray(blend, dtype=np.float32))


def tint_factors(cells: npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
    """Deterministic per-voxel luminance grain in [1-TINT_STRENGTH, 1+TINT_STRENGTH].

    A pure function of position (splitmix-style integer hash): the world's
    texture, not noise — identical for every observer, every tick, every
    restart. This is what makes places look like themselves.

    cells: (..., 3) integer voxel coordinates.
    """
    x = cells[..., 0].astype(np.uint64)
    y = cells[..., 1].astype(np.uint64)
    z = cells[..., 2].astype(np.uint64)
    h = (
        x * np.uint64(0x9E3779B97F4A7C15)
        ^ y * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ z * np.uint64(0x165667B19E3779F9)
    )
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    u = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    grain = 1.0 - TINT_STRENGTH + 2.0 * TINT_STRENGTH * u
    return cast("npt.NDArray[np.float32]", np.asarray(grain, dtype=np.float32))
