"""Run configuration: population, tick rates, devices, observability.

Layering: dataclass defaults -> YAML file -> `--set a.b.c=value` CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from gol_world.config import WorldConfig, dataclass_from_dict, load_world_config


@dataclass(frozen=True)
class DevicesConfig:
    inference: str = "cpu"
    learning: str = "cpu"


@dataclass(frozen=True)
class PopulationConfig:
    target: int = 8
    respawn_delay_ticks: int = 1200
    inherit_weights: str = "none"  # none | random_living | lineage
    mix: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ObservabilityConfig:
    rerun: bool = True
    rerun_fps: int = 10
    metrics_every_ticks: int = 100
    rrd_rotate_sim_hours: int = 6


@dataclass(frozen=True)
class RunConfig:
    world_config: str = "configs/world/default.yaml"
    tick_rate: int = 20
    act_every: int = 5
    checkpoint_interval_ticks: int = 30000
    control_port: int = 7301
    devices: DevicesConfig = field(default_factory=DevicesConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)


def apply_overrides(data: dict[str, Any], sets: list[str]) -> dict[str, Any]:
    """Apply `--set a.b.c=value` overrides (values parsed as YAML)."""
    for item in sets:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {item!r}")
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {key}: {part} is not a mapping")
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load_run_config(
    path: str | Path, sets: list[str] | None = None
) -> tuple[RunConfig, WorldConfig]:
    """Load a run config and the world config it references."""
    with open(path) as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    world_sets = [s.removeprefix("world.") for s in (sets or []) if s.startswith("world.")]
    run_sets = [s for s in (sets or []) if not s.startswith("world.")]
    data = apply_overrides(data, run_sets)
    run_cfg = dataclass_from_dict(RunConfig, data)

    with open(run_cfg.world_config) as fh:
        world_data: dict[str, Any] = yaml.safe_load(fh) or {}
    world_data = apply_overrides(world_data, world_sets)
    world_cfg = dataclass_from_dict(WorldConfig, world_data)
    return run_cfg, world_cfg


__all__ = [
    "DevicesConfig",
    "PopulationConfig",
    "ObservabilityConfig",
    "RunConfig",
    "apply_overrides",
    "load_run_config",
    "load_world_config",
]
