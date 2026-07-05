"""World configuration: terrain, ecology, and the energy economy.

Frozen dataclasses populated from YAML (configs/world/*.yaml). Defaults here
mirror configs/world/default.yaml so a config file only needs to state what it
changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_origin, get_type_hints

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class TerrainConfig:
    height_base: int = 24
    height_amp: int = 14
    octaves: int = 4
    water_level: int = 22
    ore_pockets: int = 60
    ore_pocket_size: int = 4
    bush_density: float = 0.012


@dataclass(frozen=True)
class EcologyConfig:
    regrow_ticks: int = 12000
    regrow_jitter: int = 3000
    regrow_daytime_only: bool = True
    toxic_fraction: float = 0.15  # chance a bush (generated or regrown) is toxic
    toxic_mimic: bool = False  # ablation: toxic bushes look identical to ripe ones
    # Bush lifespan: plants senesce too. A standing bush withers after roughly
    # a lifespan and a replacement sprouts elsewhere with toxicity re-rolled —
    # so the toxic share can't ratchet up (grazed ripe bushes regrow 15% toxic
    # but avoided toxic bushes never recycled) and the food map slowly drifts.
    # The bush stock is conserved: hand-eaten, spoiled, and died-carrying
    # bushes all schedule replacement sprouts. 0 disables withering (ablation).
    bush_lifespan_ticks: int = 120000  # ~5 sim-days standing, whatever its state
    bush_lifespan_jitter: int = 24000  # +/- uniform, so patches don't die in waves
    sprout_clump_bias: float = 0.7  # chance a sprout roots near an existing bush
    held_spoil_ticks: int = 24000  # a carried bush perishes after ~a sim-day (0 disables)


@dataclass(frozen=True)
class EconomyConfig:
    energy_max: float = 100.0
    integrity_max: float = 100.0
    basal_drain: float = 0.0015
    move_cost: float = 0.005
    turn_cost: float = 0.002
    climb_cost: float = 0.15
    dig_cost: float = 0.5
    place_cost: float = 0.1
    signal_cost: float = 0.01
    eat_energy: float = 40.0
    water_speed_mult: float = 0.5
    water_drain_mult: float = 3.0
    fall_damage_per_block: float = 8.0
    hibernate_integrity_drain: float = 0.0003  # per dormant tick; a coma is survivable, not free
    solar_trickle: float = 0.006  # energy/tick at full light, dormant robots only
    wake_energy: float = 40.0  # dormant robots wake above this (clears the brownout threshold)
    toxic_energy: float = 10.0  # poison berries still hold some charge
    toxic_integrity_damage: float = 12.0
    # Fatigue: a 0..1 homeostat. Builds while active, clears while still;
    # past the exhaustion threshold, energy costs multiply and integrity bleeds.
    fatigue_rise_base: float = 0.000015  # per awake tick above the rest threshold
    fatigue_rise_active: float = 0.00004  # additional per tick at full drive
    fatigue_recover: float = 0.000125  # per tick while resting (or dormant)
    rest_drive_threshold: float = 0.1  # |drive| below this counts as resting
    exhaustion_threshold: float = 0.9
    exhaustion_drain_mult: float = 1.5
    exhaustion_integrity_drain: float = 0.002
    # Brownout: a starving body sags. Below the threshold, actuation (speed and
    # turn rate) fades linearly to the floor at zero energy, so depletion is
    # felt in the body's own dynamics before stasis. 0 disables (ablation).
    brownout_threshold: float = 25.0  # below wake_energy: robots wake at full actuation
    brownout_floor: float = 0.35  # actuation fraction remaining at zero energy
    # Wear and repair: integrity is condition, not a countdown. Awake bodies
    # wear slowly; energy surplus above repair_threshold funds repair, faster
    # at rest, at an efficiency that halves every senescence_halflife ticks of
    # age. Lifespan becomes the integral of how well the robot lived.
    # repair_rate 0 disables repair; senescence_halflife 0 disables aging (ablations).
    awake_wear: float = 0.0002  # integrity/tick while awake
    repair_threshold: float = 60.0  # energy above this funds repair (never drains below it)
    repair_rate: float = 0.002  # integrity/tick at full youth, before the rest multiplier
    rest_repair_mult: float = 3.0  # repair speed multiplier while resting (sleep heals)
    repair_energy_per_point: float = 1.0  # energy cost per integrity point repaired
    senescence_halflife: float = 150000.0  # age ticks per halving of repair efficiency


@dataclass(frozen=True)
class SoundConfig:
    """Involuntary world sounds — physics, not messaging (0 ticks disables)."""

    death_cry_ticks: int = 40
    hurt_cry_ticks: int = 10


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 7
    size: tuple[int, int, int] = (256, 256, 64)
    day_length_ticks: int = 24000
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    ecology: EcologyConfig = field(default_factory=EcologyConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    sounds: SoundConfig = field(default_factory=SoundConfig)


def dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Unknown keys are rejected so config typos fail loudly.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    # get_type_hints resolves the string annotations that `from __future__
    # import annotations` leaves on dataclass fields.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints[name]
        if isinstance(ftype, type) and dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = dataclass_from_dict(ftype, value)
        elif get_origin(ftype) is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_world_config(path: str | Path) -> WorldConfig:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return dataclass_from_dict(WorldConfig, data)
