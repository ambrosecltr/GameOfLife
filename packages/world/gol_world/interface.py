"""The sensing/action contract between world and brains.

This is the stable wall: brains see Observations and emit Actions, nothing
else. It changes only by deliberate versioned decision — bump OBS_VERSION and
brain checkpoints refuse to load across the change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from gol_world.blocks import NUM_BLOCKS

OBS_VERSION = 1

# Ray hit classes: one per block id (AIR's slot is unused — a miss is NOTHING),
# then entities, then "no hit within range".
RAY_CLASS_ROBOT = NUM_BLOCKS  # 11
RAY_CLASS_DORMANT = NUM_BLOCKS + 1  # 12
RAY_CLASS_ITEM = NUM_BLOCKS + 2  # 13
RAY_CLASS_NOTHING = NUM_BLOCKS + 3  # 14
NUM_RAY_CLASSES = NUM_BLOCKS + 4  # 15

PROPRIO_DIM = 14
SOUND_DIM = 4
EVENTS_DIM = 4
SIGNAL_DIM = 2

# Gripper action modes.
GRIP_NOOP = 0
GRIP_DIG = 1
GRIP_PLACE = 2
GRIP_EAT = 3
NUM_GRIP_MODES = 4


@dataclass(frozen=True)
class BodySpec:
    """Tunable body parameters. Variant bodies are a config change, not code."""

    width: float = 0.8  # AABB footprint (blocks)
    height: float = 0.9
    max_speed: float = 4.0  # blocks/sec at full forward drive
    max_turn: float = 2.5  # rad/sec at full turn drive
    ray_pitches_deg: tuple[float, ...] = (0.0, -30.0)
    rays_per_row: int = 16
    fov_deg: float = 144.0
    ray_range: float = 24.0
    eye_height: float = 0.75  # ray origin above feet
    hear_radius: float = 12.0
    reach: float = 1.6  # gripper distance, from eye

    @property
    def num_rays(self) -> int:
        return len(self.ray_pitches_deg) * self.rays_per_row


class Observation(TypedDict):
    rays: npt.NDArray[np.float32]  # (num_rays, 1 + NUM_RAY_CLASSES): depth + class one-hot
    proprio: npt.NDArray[np.float32]  # (PROPRIO_DIM,)
    sound: npt.NDArray[np.float32]  # (SOUND_DIM,)
    events: npt.NDArray[np.float32]  # (EVENTS_DIM,): ate, took_damage, dig_success, bumped_robot


@dataclass(frozen=True)
class Action:
    drive: npt.NDArray[np.float32]  # (2,): forward in [-1, 1], turn in [-1, 1]
    gripper: int = GRIP_NOOP
    signal: npt.NDArray[np.float32] | None = None  # (SIGNAL_DIM,) in [-1, 1]

    @classmethod
    def idle(cls) -> Action:
        return cls(drive=np.zeros(2, dtype=np.float32))
