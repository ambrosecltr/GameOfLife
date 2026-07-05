"""Brain construction from config: `kind:` string -> Brain instance.

A population is a YAML list of {brain: <path or inline dict>, count: N};
the runtime resolves each entry through build_brain. A brain config may carry
a `body:` section (BodySpec overrides) — the runtime gives the robot the same
body its brain was sized for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from gol_world.interface import BodySpec

from gol_brains.base import Brain
from gol_brains.scripted import RandomWalkerBrain, ScriptedForagerBrain


def resolve_brain_config(spec: str | dict[str, Any]) -> dict[str, Any]:
    """A brain spec is either an inline dict or a path to a YAML file."""
    if isinstance(spec, str):
        with open(Path(spec)) as fh:
            loaded: dict[str, Any] = yaml.safe_load(fh)
            return loaded
    return spec


def body_from_config(cfg: dict[str, Any]) -> BodySpec:
    """BodySpec from a brain config's optional `body:` section."""
    body_cfg = dict(cfg.get("body") or {})
    if "ray_pitches_deg" in body_cfg:
        body_cfg["ray_pitches_deg"] = tuple(float(p) for p in body_cfg["ray_pitches_deg"])
    return BodySpec(**body_cfg)


def build_brain(spec: str | dict[str, Any], seed: int, device: str = "cpu") -> Brain:
    cfg = resolve_brain_config(spec)
    kind = cfg.get("kind")
    body = body_from_config(cfg)
    if kind == "random_walker":
        return RandomWalkerBrain(seed=seed)
    if kind == "scripted_forager":
        return ScriptedForagerBrain(seed=seed, body=body)
    if kind == "dreamer":
        from gol_brains.dreamer import DreamerBrain

        return DreamerBrain(cfg, seed=seed, device=device, body=body)
    raise ValueError(f"unknown brain kind: {kind!r}")


def is_learning_kind(spec: str | dict[str, Any]) -> bool:
    return resolve_brain_config(spec).get("kind") == "dreamer"
