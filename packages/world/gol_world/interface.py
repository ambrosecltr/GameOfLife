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

# v3: color vision. Rays carry depth + shaded RGB + a small hit-kind one-hot
# instead of a block-class oracle: what a block *is* must now be read from how
# it looks. Gaze control adds 2 action dims and 2 proprio dims; the ray fan
# grows to 6 pitch rows (including upward — hills and skylines are visible).
# v4 (proposal 004, finitude): one proprio channel for the body's SENESCENCE —
# how worn/old it is (0 young → 1 aged, = 1 − repair efficiency). The literal
# substrate for "time awareness": the agent can feel its finite life running
# down the way it feels hunger. We never say what it means; behaviour discovers it.
OBS_VERSION = 4

# Ray hit kinds: an animacy channel, not a semantic oracle. Blocks are told
# apart only by color; other robots stay perceptually salient as "alive".
RAY_KIND_BLOCK = 0
RAY_KIND_ROBOT = 1
RAY_KIND_DORMANT = 2
RAY_KIND_NOTHING = 3  # no hit within range (sky)
NUM_RAY_KINDS = 4

# Per-ray features: depth, r, g, b, kind one-hot.
RAY_DIM = 1 + 3 + NUM_RAY_KINDS

PROPRIO_DIM = 18  # v4: +1 for the senescence channel (index 17)
SOUND_DIM = 4
EVENTS_DIM = 4
SIGNAL_DIM = 2
GAZE_DIM = 2  # (pitch, yaw) targets in [-1, 1], scaled by the body's gaze range

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
    # An elliptical visual field: wide horizontally, narrower vertically,
    # denser near the horizon where most of life happens.
    ray_pitches_deg: tuple[float, ...] = (30.0, 12.0, 0.0, -12.0, -30.0, -50.0)
    rays_per_row: int = 24
    fov_deg: float = 160.0
    ray_range: float = 32.0
    # Gaze: the head aims the eyes beyond the fixed fan, without turning the
    # body. The gripper still works along the body heading — eyes look, arms
    # reach from the chest.
    gaze_pitch_max_deg: float = 45.0
    gaze_yaw_max_deg: float = 90.0
    eye_height: float = 0.75  # ray origin above feet
    hear_radius: float = 12.0
    reach: float = 1.6  # gripper distance, from eye

    @property
    def num_rays(self) -> int:
        return len(self.ray_pitches_deg) * self.rays_per_row


class Observation(TypedDict):
    rays: npt.NDArray[np.float32]  # (num_rays, RAY_DIM): depth + rgb + kind one-hot
    proprio: npt.NDArray[np.float32]  # (PROPRIO_DIM,)
    sound: npt.NDArray[np.float32]  # (SOUND_DIM,)
    events: npt.NDArray[np.float32]  # (EVENTS_DIM,): ate, took_damage, dig_success, bumped_robot


@dataclass(frozen=True)
class Action:
    drive: npt.NDArray[np.float32]  # (2,): forward in [-1, 1], turn in [-1, 1]
    gripper: int = GRIP_NOOP
    signal: npt.NDArray[np.float32] | None = None  # (SIGNAL_DIM,) in [-1, 1]
    gaze: npt.NDArray[np.float32] | None = None  # (GAZE_DIM,) in [-1, 1]; None = look straight

    @classmethod
    def idle(cls) -> Action:
        return cls(drive=np.zeros(2, dtype=np.float32))
