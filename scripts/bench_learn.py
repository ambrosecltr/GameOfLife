#!/usr/bin/env python
"""Synchronized single-brain precision and throughput benchmark.

The benchmark preserves replay shape and train ratio while varying only the
requested precision policy. Target-GPU measurements, never device marketing
specifications, gate the sustainable virtual tick rate.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import yaml
from gol_brains.precision import PrecisionPolicy, configure_process_precision
from gol_world.interface import (
    EVENTS_DIM,
    NUM_RAY_KINDS,
    PROPRIO_DIM,
    RAY_DIM,
    SOUND_DIM,
    BodySpec,
    Observation,
)

T = TypeVar("T")
PRECISION_CHOICES = ("config", "ieee_fp32", "tf32", "amp_bf16")


def synthetic_obs(rng: np.random.Generator, body: BodySpec) -> Observation:
    rays = np.zeros((body.num_rays, RAY_DIM), dtype=np.float32)
    rays[:, 0] = rng.random(body.num_rays)
    rays[:, 1:4] = rng.random((body.num_rays, 3)).astype(np.float32)
    kinds = rng.integers(0, NUM_RAY_KINDS, body.num_rays)
    rays[np.arange(body.num_rays), 4 + kinds] = 1.0
    return Observation(
        rays=rays,
        proprio=rng.random(PROPRIO_DIM).astype(np.float32),
        sound=rng.random(SOUND_DIM).astype(np.float32),
        events=np.zeros(EVENTS_DIM, dtype=np.float32),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def synchronized_call(device: torch.device, call: Callable[[], T]) -> tuple[T, float]:
    synchronize(device)
    began = time.perf_counter()
    result = call()
    synchronize(device)
    return result, time.perf_counter() - began


def peak_host_memory_mb() -> float:
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return maximum / divisor


def build_cfg(
    args: argparse.Namespace,
    compile_on: bool,
    precision: str,
) -> dict[str, Any]:
    if args.brain:
        with open(args.brain) as file:
            cfg: dict[str, Any] = yaml.safe_load(file)
    else:
        cfg = {"kind": "dreamer", "preset": args.preset}
    replay = cfg.setdefault("replay", {})
    training = cfg.setdefault("training", {})
    if args.batch:
        replay["batch_size"] = args.batch
    if args.seq:
        replay["seq_len"] = args.seq
    if args.burn_in is not None:
        replay["burn_in"] = args.burn_in
    if args.recent is not None:
        replay["recent"] = args.recent
    if args.optimizer:
        training["optimizer"] = args.optimizer
    if args.ratio is not None:
        training["train_ratio"] = args.ratio
    if precision != "config":
        training["precision"] = precision
    replay["warmup_steps"] = min(args.fill, int(replay.get("warmup_steps", 100)))
    training["compile"] = compile_on
    if compile_on:
        training["async_inference"] = False
    return cfg


def _profile_update(brain: Any, device: torch.device, output: Path) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        result, _ = synchronized_call(device, brain.learn)
        if result is None:
            raise RuntimeError("profile update found no learnable replay")
    profile.export_chrome_trace(str(output))
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(profile.key_averages().table(sort_by=sort_key, row_limit=30))


def bench_one(
    args: argparse.Namespace,
    device_name: str,
    compile_on: bool,
    precision: str,
) -> dict[str, float | str]:
    from gol_brains.registry import body_from_config, build_brain

    cfg = build_cfg(args, compile_on, precision)
    device = torch.device(device_name)
    policy = PrecisionPolicy.from_config(dict(cfg.get("training", {})), device)
    configure_process_precision([policy])
    brain = build_brain(cfg, seed=0, device=device_name)
    body = body_from_config(cfg)
    rng = np.random.default_rng(0)
    replay = cfg["replay"]
    fill = max(
        args.fill,
        int(replay.get("warmup_steps", 0)),
        int(replay.get("seq_len", 64)) + int(replay.get("burn_in", 0)) + 2,
    )

    act_times: list[float] = []
    for index in range(fill):
        observation = synthetic_obs(rng, body)
        _, elapsed = synchronized_call(device, partial(brain.act, observation))
        if index >= fill // 2:
            act_times.append(elapsed)

    warmup_updates = max(2, args.updates // 5)
    for _ in range(warmup_updates):
        update, _ = synchronized_call(device, brain.learn)
        if update is None:
            raise RuntimeError("buffer must be past warmup")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    learn_times: list[float] = []
    for _ in range(args.updates):
        metrics, elapsed = synchronized_call(device, brain.learn)
        if metrics is None:
            raise RuntimeError("buffer must be past warmup")
        learn_times.append(elapsed)

    host_peak_mb = peak_host_memory_mb()
    peak_vram_allocated_mb = None
    peak_vram_reserved_mb = None
    if device.type == "cuda":
        peak_vram_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_vram_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2

    if args.profile is not None:
        suffix = f"{device_name.replace(':', '_')}-{policy.mode.value}.json"
        _profile_update(brain, device, args.profile / suffix)

    sequence_timepoints = int(replay.get("batch_size", 16)) * int(replay.get("seq_len", 64))
    mean = statistics.mean(learn_times)
    p50 = statistics.median(learn_times)
    train_ratio = float(
        args.ratio if args.ratio is not None else cfg["training"].get("train_ratio", 0.25)
    )
    if train_ratio <= 0.0:
        raise ValueError("throughput benchmark requires a positive train ratio")
    sustainable_tick_rate = (1.0 / p50) * args.act_every / train_ratio
    benchmark: dict[str, float | str] = {
        "device": device_name,
        "precision": policy.mode.value,
        "compile": "on" if compile_on else "off",
        "learn_mean_seconds": mean,
        "learn_p50_seconds": p50,
        "learn_min_seconds": min(learn_times),
        "act_p50_seconds": statistics.median(act_times),
        "act_p95_seconds": float(np.quantile(act_times, 0.95)),
        "graded_timepoints_per_second": sequence_timepoints / p50,
        "sustainable_ticks_per_second_one_brain": sustainable_tick_rate,
        "sustainable_ticks_per_second_with_headroom": (
            sustainable_tick_rate * args.headroom
        ),
        "headroom": args.headroom,
        "host_peak_mb": host_peak_mb,
        "batch_size": float(replay.get("batch_size", 16)),
        "sequence_length": float(replay.get("seq_len", 64)),
        "burn_in": float(replay.get("burn_in", 0)),
        "train_ratio": train_ratio,
    }
    if peak_vram_allocated_mb is not None and peak_vram_reserved_mb is not None:
        benchmark["peak_vram_allocated_mb"] = peak_vram_allocated_mb
        benchmark["peak_vram_reserved_mb"] = peak_vram_reserved_mb
    if args.gpu_hourly_cost is not None:
        seconds_per_million = 1_000_000.0 / (sequence_timepoints / p50)
        benchmark["cost_per_million_timepoints"] = (
            seconds_per_million / 3600.0 * args.gpu_hourly_cost
        )
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", help="brain YAML (default: inline preset config)")
    parser.add_argument("--preset", default="nano", choices=["nano", "small", "base"])
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--precision", nargs="+", default=["config"], choices=PRECISION_CHOICES)
    parser.add_argument("--compile", nargs="+", default=["off"], choices=["off", "on"])
    parser.add_argument("--updates", type=int, default=20, help="timed learn() calls")
    parser.add_argument("--fill", type=int, default=1400, help="buffer act-steps before timing")
    parser.add_argument("--batch", type=int, help="replay.batch_size override")
    parser.add_argument("--seq", type=int, help="replay.seq_len override")
    parser.add_argument("--burn-in", type=int, help="replay.burn_in override")
    parser.add_argument("--recent", type=int, help="replay.recent override")
    parser.add_argument("--optimizer", choices=["adam", "muon"])
    parser.add_argument("--act-every", type=int, default=5)
    parser.add_argument("--headroom", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, help="target train ratio override")
    parser.add_argument("--gpu-hourly-cost", type=float)
    parser.add_argument("--profile", type=Path, help="directory for Chrome profiler traces")
    parser.add_argument("--json", action="store_true", help="emit one JSON record per case")
    args = parser.parse_args()

    if args.updates < 1 or args.fill < 1:
        parser.error("--updates and --fill must be positive")
    if not 0.0 < args.headroom <= 1.0:
        parser.error("--headroom must be in (0, 1]")
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    rows = []
    for device in args.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            print("mps unavailable, skipping")
            continue
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"{device} unavailable, skipping")
            continue
        for precision in args.precision:
            for compile_value in args.compile:
                row = bench_one(args, device, compile_value == "on", precision)
                rows.append(row)
                if args.json:
                    print(json.dumps(row, sort_keys=True))
                else:
                    print(
                        f"{row['device']}/{row['precision']}/compile={row['compile']}: "
                        f"learn {float(row['learn_mean_seconds']) * 1e3:.1f}ms mean, "
                        f"{float(row['learn_p50_seconds']) * 1e3:.1f}ms p50, "
                        f"act {float(row['act_p50_seconds']) * 1e3:.2f}ms p50, "
                        f"{float(row['graded_timepoints_per_second']):.0f} graded timepoints/s"
                    )
    if not rows:
        raise SystemExit("no benchmark case ran")


if __name__ == "__main__":
    main()
