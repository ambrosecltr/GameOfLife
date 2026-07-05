"""Scripted baseline brains: the in-world control group.

RandomWalkerBrain wanders with a persistent heading and occasional turns —
the null hypothesis for any behavior claim. ScriptedForagerBrain (M2) is the
economy calibration probe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gol_world.interface import Action, Observation

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
