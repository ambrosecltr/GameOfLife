"""Scripted baseline brains: the in-world control group.

RandomWalkerBrain wanders with a persistent heading and occasional turns —
the null hypothesis for any behavior claim. ScriptedForagerBrain (M2) is the
economy calibration probe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gol_world.blocks import Block
from gol_world.interface import GRIP_EAT, Action, BodySpec, Observation

from gol_brains.base import Brain


class RandomWalkerBrain(Brain):
    """Drives forward, occasionally picking a new turn rate; backs off walls."""

    def __init__(self, seed: int = 0, turn_every: int = 12) -> None:
        self.rng = np.random.default_rng(seed)
        self.turn_every = turn_every
        self._steps = 0
        self._turn = 0.0
        self._backing_off = 0

    def act(self, obs: Observation) -> Action:
        self._steps += 1
        touch_front = obs["proprio"][9] > 0.5

        if self._backing_off > 0:
            self._backing_off -= 1
            return Action(drive=np.array([-0.6, 0.9], dtype=np.float32))
        if touch_front:
            self._backing_off = int(self.rng.integers(2, 5))
            return Action(drive=np.array([-0.6, 0.9], dtype=np.float32))

        if self._steps % self.turn_every == 0 or self._steps == 1:
            self._turn = float(self.rng.uniform(-0.6, 0.6))
        forward = float(self.rng.uniform(0.5, 1.0))
        return Action(drive=np.array([forward, self._turn], dtype=np.float32))

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "steps": self._steps,
            "turn": self._turn,
            "backing_off": self._backing_off,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng_state"]
        self._steps = state["steps"]
        self._turn = state["turn"]
        self._backing_off = state["backing_off"]


class ScriptedForagerBrain(Brain):
    """Seeks ripe bushes, eats, avoids water, rests at night when fed.

    The economy calibration probe: world costs/yields are tuned so this brain
    thrives while RandomWalkerBrain starves.
    """

    def __init__(self, seed: int = 0, body: BodySpec | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.body = body or BodySpec()
        self._wander_turn = 0.0
        self._steps = 0
        self._backing_off = 0
        n = self.body.rays_per_row
        half_fov = np.radians(self.body.fov_deg) / 2
        # Azimuth (relative to heading) of each ray, tiled across pitch rows.
        self._azimuths = np.tile(
            np.linspace(-half_fov, half_fov, n), len(self.body.ray_pitches_deg)
        )

    def act(self, obs: Observation) -> Action:
        self._steps += 1
        rays = obs["rays"]
        depths = rays[:, 0] * self.body.ray_range
        classes = rays[:, 1:].argmax(axis=1)
        proprio = obs["proprio"]
        energy = float(proprio[5])
        light = float(proprio[13])
        touch_front = proprio[9] > 0.5

        if self._backing_off > 0:
            self._backing_off -= 1
            return Action(drive=np.array([-0.6, 1.0], dtype=np.float32))
        if touch_front:
            self._backing_off = int(self.rng.integers(2, 5))
            return Action(drive=np.array([-0.6, 1.0], dtype=np.float32))

        # Rest through the night unless hungry.
        if light < 0.15 and energy > 0.5:
            return Action(drive=np.zeros(2, dtype=np.float32))

        # Steer to the nearest visible ripe bush; eat when within reach.
        bush = classes == Block.BUSH_RIPE
        if bush.any():
            idx = int(np.flatnonzero(bush)[np.argmin(depths[bush])])
            azimuth = float(self._azimuths[idx])
            if depths[idx] < self.body.reach and abs(azimuth) < 0.45:
                return Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT)
            turn = float(np.clip(azimuth * 2.0, -1.0, 1.0))
            forward = 0.9 if abs(azimuth) < 0.8 else 0.3
            return Action(drive=np.array([forward, turn], dtype=np.float32))

        # Avoid driving into water dead ahead.
        center = np.abs(self._azimuths) < 0.35
        water_ahead = (classes == Block.WATER) & center & (depths < 3.0)
        if water_ahead.any():
            return Action(drive=np.array([0.3, 1.0], dtype=np.float32))

        # Wander.
        if self._steps % 10 == 0 or self._steps == 1:
            self._wander_turn = float(self.rng.uniform(-0.5, 0.5))
        return Action(drive=np.array([0.8, self._wander_turn], dtype=np.float32))

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "steps": self._steps,
            "wander_turn": self._wander_turn,
            "backing_off": self._backing_off,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng_state"]
        self._steps = state["steps"]
        self._wander_turn = state["wander_turn"]
        self._backing_off = state["backing_off"]
