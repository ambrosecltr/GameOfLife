#!/usr/bin/env python
"""Screen an Aion felt economy against an archived chronological replay.

The screen validates reward arithmetic and blackout timing before a live run. It
does not claim that an off-policy replay predicts the behavior of a newly trained
actor.

    uv run python scripts/aion_economy_screen.py \
      saves/archive/aion_01_2gpu/best_brain_aion_114.pt.zst \
      --brain configs/brain/aion_02_economy.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from gol_brains import feeling


@contextmanager
def _restored_checkpoint(path: Path) -> Iterator[Path]:
    if path.suffix != ".zst":
        yield path
        return
    with tempfile.NamedTemporaryFile(suffix=".pt") as restored:
        subprocess.run(
            ["zstd", "-d", "-f", str(path), "-o", restored.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        yield Path(restored.name)


def _load_pickle_on_cpu(path: Path) -> dict[str, Any]:
    original = torch.storage._load_from_bytes

    def load_storage(data: bytes) -> Any:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

    torch.storage._load_from_bytes = load_storage
    try:
        with path.open("rb") as checkpoint:
            state = pickle.load(checkpoint)
    finally:
        torch.storage._load_from_bytes = original
    if not isinstance(state, dict) or not isinstance(state.get("buffer"), dict):
        raise ValueError(f"{path} is not a brain checkpoint with replay")
    return state


def _quantiles(values: torch.Tensor) -> list[float]:
    return [
        round(float(value), 6)
        for value in torch.quantile(values.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    ]


def _economy_tensors(
    replay: dict[str, Any], config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    reward = dict(config["reward"])
    drive_config = dict(reward["drive"])
    viability_config = dict(reward["viability"])
    wellbeing_config = dict(reward["wellbeing"])
    pain_config = dict(reward["pain"])
    proprio = torch.as_tensor(replay["proprio"].astype(np.float32))
    events = torch.as_tensor(replay["events"].astype(np.float32))
    first = torch.as_tensor(replay["first"].astype(np.float32))
    wake = torch.as_tensor(replay["wake"].astype(np.float32))
    step_scale = torch.as_tensor(replay["step_scale"].astype(np.float32))
    discontinuity = torch.maximum(first, wake)
    setpoints = torch.tensor(
        [
            drive_config["energy_setpoint"],
            drive_config["integrity_setpoint"],
            drive_config["rested_setpoint"],
        ]
    )
    drive_weights = torch.tensor(
        [
            drive_config["energy_weight"],
            drive_config["integrity_weight"],
            drive_config["rest_weight"],
        ]
    )
    drive = feeling.drive_level(
        proprio,
        setpoints,
        drive_weights,
        float(drive_config["pow_m"]),
        float(drive_config["pow_n"]),
    )
    viability = feeling.viability(
        proprio,
        barrier_cap=float(viability_config["barrier_cap"]),
        total_cap=float(viability_config["total_cap"]),
        energy_lethal=float(viability_config["energy_lethal"]),
        energy_safe=float(viability_config["energy_safe"]),
        integrity_lethal=float(viability_config["integrity_lethal"]),
        integrity_safe=float(viability_config["integrity_safe"]),
        energy_weight=float(viability_config["energy_weight"]),
        integrity_weight=float(viability_config["integrity_weight"]),
    )
    comfort = float(drive_config["scale"]) * feeling.reduction(drive, discontinuity)
    comfort = comfort - float(drive_config["level_penalty"]) * drive
    wellbeing = feeling.wellbeing(
        viability,
        drive,
        weight=float(wellbeing_config["weight"]),
        barrier_cap=float(viability_config["barrier_cap"]),
        comfort_decay=float(wellbeing_config["comfort_decay"]),
    )
    pain = -float(pain_config["weight"]) * feeling.acute_integrity_loss(
        proprio, events[..., 1], discontinuity
    )
    return {
        "proprio": proprio,
        "events": events,
        "first": first,
        "wake": wake,
        "step_scale": step_scale,
        "drive": drive,
        "viability": viability,
        "comfort": comfort,
        "wellbeing": wellbeing,
        "pain": pain,
        "body_total": comfort + wellbeing + pain,
    }


def _state_band_summary(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    proprio = tensors["proprio"]
    lived = tensors["first"] < 0.5
    bands = {
        "healthy": lived
        & (proprio[..., feeling.ENERGY_IDX] >= 0.6)
        & (proprio[..., feeling.INTEGRITY_IDX] >= 0.7)
        & (proprio[..., feeling.FATIGUE_IDX] <= 0.5),
        "worn": lived
        & (proprio[..., feeling.ENERGY_IDX] >= 0.25)
        & (proprio[..., feeling.ENERGY_IDX] < 0.6)
        & (proprio[..., feeling.INTEGRITY_IDX] >= 0.3),
        "dying": lived
        & (
            (proprio[..., feeling.ENERGY_IDX] < 0.1)
            | (proprio[..., feeling.INTEGRITY_IDX] < 0.15)
        ),
    }
    result: dict[str, dict[str, float | int]] = {}
    for name, mask in bands.items():
        result[name] = {
            "samples": int(mask.sum()),
            "wellbeing_mean": round(float(tensors["wellbeing"][mask].mean()), 6),
            "body_total_mean": round(float(tensors["body_total"][mask].mean()), 6),
        }
    return result


def _meal_screen(config: dict[str, Any]) -> list[dict[str, float]]:
    states = ((0.15, 1.0), (0.3, 1.0), (0.5, 0.8), (0.3, 0.6), (0.85, 1.0))
    results = []
    for energy, integrity in states:
        row: dict[str, float] = {"energy": energy, "integrity": integrity}
        for name, energy_gain, integrity_change, damaged in (
            ("ripe", 0.4, 0.0, False),
            ("toxic", 0.1, -0.12, True),
        ):
            proprio = np.zeros((2, 19), dtype=np.float32)
            proprio[:, feeling.INTEGRITY_IDX] = integrity
            proprio[0, feeling.ENERGY_IDX] = energy
            proprio[1, feeling.ENERGY_IDX] = min(1.0, energy + energy_gain)
            proprio[1, feeling.INTEGRITY_IDX] = max(0.0, integrity + integrity_change)
            events = np.zeros((2, 4), dtype=np.uint8)
            events[1, 1] = damaged
            synthetic = {
                "proprio": proprio,
                "events": events,
                "first": np.array([1, 0], dtype=np.uint8),
                "wake": np.zeros(2, dtype=np.uint8),
                "step_scale": np.ones(2, dtype=np.float32),
            }
            row[name] = round(float(_economy_tensors(synthetic, config)["body_total"][1]), 6)
        results.append(row)
    return results


def build_report(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    replay = state["buffer"]
    tensors = _economy_tensors(replay, config)
    lived = tensors["first"] < 0.5
    wake = tensors["wake"] > 0.5
    gamma = float(config["actor_critic"]["gamma"])
    wake_discounts = torch.pow(
        torch.full_like(tensors["step_scale"][wake], gamma),
        tensors["step_scale"][wake] - 1.0,
    )
    channels = {}
    for name in ("drive", "viability", "comfort", "wellbeing", "pain", "body_total"):
        values = tensors[name][lived]
        channels[name] = {
            "mean": round(float(values.mean()), 6),
            "quantiles": _quantiles(values),
        }
    bands = _state_band_summary(tensors)
    meals = _meal_screen(config)
    return {
        "brain_family": state.get("brain_family"),
        "updates": int(state.get("updates", 0)),
        "replay_samples": int(len(replay["proprio"])),
        "body_streams": int(tensors["first"].sum()),
        "wake_transitions": int(tensors["wake"].sum()),
        "acute_damage_events": int(tensors["events"][..., 1].sum()),
        "channels": channels,
        "state_bands": bands,
        "ordering_pass": (
            bands["healthy"]["body_total_mean"]
            > bands["worn"]["body_total_mean"]
            > bands["dying"]["body_total_mean"]
            > 0.0
        ),
        "meal_screen": meals,
        "meal_screen_pass": all(row["ripe"] > 0.0 and row["toxic"] < 0.0 for row in meals),
        "wake_step_scale_quantiles": _quantiles(tensors["step_scale"][wake]),
        "wake_discount_quantiles": _quantiles(wake_discounts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.brain.read_text())
    with _restored_checkpoint(args.checkpoint) as checkpoint:
        state = _load_pickle_on_cpu(checkpoint)
    report = build_report(state, config)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
