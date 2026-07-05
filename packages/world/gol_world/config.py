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
    hibernate_integrity_drain: float = 0.005


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 7
    size: tuple[int, int, int] = (256, 256, 64)
    day_length_ticks: int = 24000
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    ecology: EcologyConfig = field(default_factory=EcologyConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)


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
