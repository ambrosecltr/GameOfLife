"""Entities that live in the world: robots (and, from M2, dropped items)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from gol_world.interface import EVENTS_DIM, SIGNAL_DIM, BodySpec

# Indices into Robot.events (mirrors Observation["events"]).
EV_ATE = 0
EV_TOOK_DAMAGE = 1
EV_DIG_SUCCESS = 2
EV_BUMPED_ROBOT = 3

# Indices into Robot.touch (mirrors the proprio touch block).
TOUCH_FRONT = 0
TOUCH_LEFT = 1
TOUCH_RIGHT = 2
TOUCH_GROUND = 3

# Lifetime integrity ledger: cumulative damage by cause, plus repair. Answers
# "what killed this robot" (death events) and "what is wearing them down"
# (metrics) — the observability half of the wear/repair economy.
LEDGER_KEYS = ("wear", "hibernation", "exhaustion", "fall", "poison", "repaired")


def new_ledger() -> dict[str, float]:
    return dict.fromkeys(LEDGER_KEYS, 0.0)


@dataclass
class Robot:
    id: str
    pos: npt.NDArray[np.float64]  # (3,) feet-center position (z = bottom of AABB)
    yaw: float
    brain_name: str
    body: BodySpec = field(default_factory=BodySpec)
    vel: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    energy: float = 100.0
    integrity: float = 100.0
    held: int | None = None  # block id being carried
    dormant: bool = False
    fatigue: float = 0.0  # 0..1; builds with activity, clears with rest
    age_ticks: int = 0
    ledger: dict[str, float] = field(default_factory=new_ledger)
    # Commanded controls; persist between act-steps (grip is one-shot).
    drive: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    signal: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(SIGNAL_DIM))
    pending_grip: int = 0
    # Set by physics each tick.
    touch: npt.NDArray[np.bool_] = field(default_factory=lambda: np.zeros(4, dtype=np.bool_))
    in_water: bool = False
    fall_peak_z: float = 0.0
    # Accumulated since the robot's last act-step; drained into observations.
    events: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(EVENTS_DIM))

    @property
    def aabb(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        half = self.body.width / 2
        lo = np.array([self.pos[0] - half, self.pos[1] - half, self.pos[2]])
        hi = np.array([self.pos[0] + half, self.pos[1] + half, self.pos[2] + self.body.height])
        return lo, hi

    @property
    def eye(self) -> npt.NDArray[np.float64]:
        return np.array([self.pos[0], self.pos[1], self.pos[2] + self.body.eye_height])

    def drain_events(self) -> npt.NDArray[np.float64]:
        out, self.events = self.events, np.zeros(EVENTS_DIM)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pos": self.pos.tolist(),
            "yaw": self.yaw,
            "brain_name": self.brain_name,
            "vel": self.vel.tolist(),
            "energy": self.energy,
            "integrity": self.integrity,
            "held": self.held,
            "dormant": self.dormant,
            "fatigue": self.fatigue,
            "age_ticks": self.age_ticks,
            "ledger": self.ledger,
            "drive": self.drive.tolist(),
            "signal": self.signal.tolist(),
            "pending_grip": self.pending_grip,
            "fall_peak_z": self.fall_peak_z,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], body: BodySpec | None = None) -> Robot:
        return cls(
            id=data["id"],
            pos=np.array(data["pos"], dtype=np.float64),
            yaw=float(data["yaw"]),
            brain_name=data["brain_name"],
            body=body or BodySpec(),
            vel=np.array(data["vel"], dtype=np.float64),
            energy=float(data["energy"]),
            integrity=float(data["integrity"]),
            held=data["held"],
            dormant=bool(data["dormant"]),
            fatigue=float(data.get("fatigue", 0.0)),
            age_ticks=int(data["age_ticks"]),
            ledger={**new_ledger(), **data.get("ledger", {})},
            drive=np.array(data["drive"], dtype=np.float64),
            signal=np.array(data["signal"], dtype=np.float64),
            pending_grip=int(data.get("pending_grip", 0)),
            fall_peak_z=float(data.get("fall_peak_z", data["pos"][2])),
        )
