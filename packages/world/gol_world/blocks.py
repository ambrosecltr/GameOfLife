"""Block palette and per-block property tables.

This module is the single source of truth for what blocks exist and how they
behave/appear. Property tables are numpy arrays indexed by block id so the sim
hot paths (physics, raycasting) and the renderer share one definition.
"""

from enum import IntEnum

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
