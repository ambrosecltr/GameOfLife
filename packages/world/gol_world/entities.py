"""Entities that live in the world: robots (and, from M2, dropped items)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from gol_world.interface import EVENTS_DIM, GAZE_DIM, SIGNAL_DIM, BodySpec

# Stable per-robot identity colors: how a body *looks*, to rays and viewer
# alike — individuals are visually recognizable. Dormant bodies dim.
ROBOT_PALETTE: npt.NDArray[np.uint8] = np.array(
    [
        [230, 80, 60],  # red
        [70, 160, 235],  # blue
        [110, 205, 90],  # green
        [240, 190, 60],  # gold
        [180, 110, 235],  # violet
        [250, 140, 190],  # pink
        [90, 220, 210],  # teal
        [250, 160, 60],  # orange
        [165, 165, 175],  # gray
        [200, 220, 120],  # lime
    ],
    dtype=np.uint8,
)
DORMANT_DIM = 0.45


def robot_color(robot_id: str) -> npt.NDArray[np.uint8]:
    try:
        idx = int(robot_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        idx = abs(hash(robot_id))
    color: npt.NDArray[np.uint8] = ROBOT_PALETTE[idx % len(ROBOT_PALETTE)]
    return color


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


# Lifetime energy ledger: cumulative spend by cause, plus income. Answers
# "where does the energy actually go" — anima_03 sized the wake corridor
# against an assumed drain (basal+move ≈ 0.0065/tick) and the measured total
# was 2-3× that; every affordance calibration after that round uses this
# ledger's measured breakdown instead of arithmetic on config constants.
# Spend keys are the charge sites; `exhaustion`/`water` hold only the
# multiplier SURCHARGE over the base drain. `eaten`/`solar` record the energy
# actually banked (a meal at 97/100 banks 3, not 40 — overflow is visible as
# eat_events × eat_energy − eaten).
ENERGY_LEDGER_KEYS = (
    "basal",
    "move",
    "turn",
    "climb",
    "signal",
    "exhaustion",
    "water",
    "dig",
    "place",
    "repair",
    "bud",
    "eaten",
    "solar",
)


def new_energy_ledger() -> dict[str, float]:
    return dict.fromkeys(ENERGY_LEDGER_KEYS, 0.0)


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
    held_age_ticks: int = 0  # ticks the current bush has been carried (spoilage clock)
    dormant: bool = False
    fatigue: float = 0.0  # 0..1; builds with activity, clears with rest
    age_ticks: int = 0
    ledger: dict[str, float] = field(default_factory=new_ledger)
    energy_ledger: dict[str, float] = field(default_factory=new_energy_ledger)
    # Commanded controls; persist between act-steps (grip is one-shot).
    drive: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    signal: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(SIGNAL_DIM))
    gaze: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(GAZE_DIM))
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
            "held_age_ticks": self.held_age_ticks,
            "dormant": self.dormant,
            "fatigue": self.fatigue,
            "age_ticks": self.age_ticks,
            "ledger": self.ledger,
            "energy_ledger": self.energy_ledger,
            "drive": self.drive.tolist(),
            "signal": self.signal.tolist(),
            "gaze": self.gaze.tolist(),
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
            held_age_ticks=int(data.get("held_age_ticks", 0)),
            dormant=bool(data["dormant"]),
            fatigue=float(data.get("fatigue", 0.0)),
            age_ticks=int(data["age_ticks"]),
            ledger={**new_ledger(), **data.get("ledger", {})},
            energy_ledger={**new_energy_ledger(), **data.get("energy_ledger", {})},
            drive=np.array(data["drive"], dtype=np.float64),
            signal=np.array(data["signal"], dtype=np.float64),
            gaze=np.array(data.get("gaze", [0.0] * GAZE_DIM), dtype=np.float64),
            pending_grip=int(data.get("pending_grip", 0)),
            fall_peak_z=float(data.get("fall_peak_z", data["pos"][2])),
        )
