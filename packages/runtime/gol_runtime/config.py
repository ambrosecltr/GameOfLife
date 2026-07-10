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
    learning: str | tuple[str, ...] = "cpu"

    def learning_devices(self) -> tuple[str, ...]:
        devices = (self.learning,) if isinstance(self.learning, str) else self.learning
        if not devices or any(not device for device in devices):
            raise ValueError("devices.learning must contain at least one device")
        return devices


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
    # Live-viewer memory ceiling: the spawned viewer garbage-collects the OLDEST
    # timepoints once it exceeds this, so watching a long run keeps a sliding
    # window of recent past instead of growing without bound. Accepts an
    # absolute size ("2GB") or a fraction of system RAM ("75%"). Only affects the
    # spawned viewer; file recording (--rrd) is bounded by rotation instead.
    rerun_memory_limit: str = "2GB"

    def __post_init__(self) -> None:
        if self.rerun_fps < 1 or self.metrics_every_ticks < 1:
            raise ValueError("observability rates must be positive")
        if self.rrd_rotate_sim_hours < 1:
            raise ValueError("observability.rrd_rotate_sim_hours must be positive")


@dataclass(frozen=True)
class PacingConfig:
    """Wall-clock execution policy; virtual-time semantics stay fixed."""

    mode: str = "fixed"  # fixed | adaptive
    debt_policy: str = "backpressure"  # backpressure | drop
    headroom: float = 0.85
    min_tick_rate: float = 1.0
    max_tick_rate: float = 1000.0
    max_debt_updates: float = 4.0
    resume_debt_updates: float = 2.0
    hysteresis: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError("pacing.mode must be 'fixed' or 'adaptive'")
        if self.debt_policy not in ("backpressure", "drop"):
            raise ValueError("pacing.debt_policy must be 'backpressure' or 'drop'")
        if not 0.0 < self.headroom <= 1.0:
            raise ValueError("pacing.headroom must be in (0, 1]")
        if not 0.0 < self.min_tick_rate <= self.max_tick_rate:
            raise ValueError("pacing tick-rate bounds must satisfy 0 < min <= max")
        if self.max_debt_updates < 1.0:
            raise ValueError("pacing.max_debt_updates must be at least one")
        if not 1.0 <= self.resume_debt_updates < self.max_debt_updates:
            raise ValueError("pacing.resume_debt_updates must be in [1, max_debt_updates)")
        if not 0.0 <= self.hysteresis < 1.0:
            raise ValueError("pacing.hysteresis must be in [0, 1)")


@dataclass(frozen=True)
class DormancyAccelerationConfig:
    exact_unpaced: bool = False
    event_fast_forward: bool = False
    max_jump_ticks: int = 100_000

    def __post_init__(self) -> None:
        if self.max_jump_ticks < 1:
            raise ValueError("dormancy_acceleration.max_jump_ticks must be positive")
        if self.event_fast_forward and not self.exact_unpaced:
            raise ValueError("event_fast_forward requires exact_unpaced")


@dataclass(frozen=True)
class ReproductionConfig:
    """How the population replaces the dead (proposal 004).

    "respawn" (default): the legacy timer — the dead are replaced after a delay,
    regardless of how anyone lived. "budding": earned, endogenous reproduction —
    an evolving lineage continues only by a THRIVING body spending its own
    surplus to bud a child, so lineages that sleep their lives away leave fewer
    descendants and a survival instinct can be selected for rather than taught.
    A low respawn floor guards against extinction for evolving kinds; scripted
    anchor kinds (foragers) are always kept at their mix count by respawn.
    """

    mode: str = "respawn"  # respawn | budding
    # NOTE: the thrive check is INSTANTANEOUS (at the bud tick, not sustained) —
    # keep thrive_energy strictly above the world's wake_energy, or waking from
    # hibernation is itself eligibility and budding subsidizes the attractor.
    thrive_energy: float = 75.0  # energy at/above this makes a body eligible to bud
    thrive_integrity: float = 70.0  # an intact body — reproduction needs a well-lived one
    min_bud_age: int = 20000  # no budding before this age (a juvenile can't reproduce)
    bud_cooldown: int = 15000  # ticks a parent must wait between buds
    bud_cost_energy: float = 40.0  # energy the parent spends to bud (reproduction is costly)
    bud_cost_integrity: float = 5.0  # integrity the parent spends to bud
    floor: int = 4  # emergency respawn if an evolving kind falls below this (extinction guard)


@dataclass(frozen=True)
class RunConfig:
    world_config: str = "configs/world/default.yaml"
    tick_rate: int = 20
    act_every: int = 5
    checkpoint_interval_ticks: int = 30000
    control_port: int = 7301
    devices: DevicesConfig = field(default_factory=DevicesConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    pacing: PacingConfig = field(default_factory=PacingConfig)
    dormancy_acceleration: DormancyAccelerationConfig = field(
        default_factory=DormancyAccelerationConfig
    )

    def __post_init__(self) -> None:
        if self.tick_rate < 1 or self.act_every < 1:
            raise ValueError("tick_rate and act_every must be positive")
        if self.checkpoint_interval_ticks < 1:
            raise ValueError("checkpoint_interval_ticks must be positive")


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
    "DormancyAccelerationConfig",
    "PopulationConfig",
    "PacingConfig",
    "ReproductionConfig",
    "ObservabilityConfig",
    "RunConfig",
    "apply_overrides",
    "load_run_config",
    "load_world_config",
]
