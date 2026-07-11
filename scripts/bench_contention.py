#!/usr/bin/env python
"""Synchronized K-brain contention and inference-deadline benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from typing import Any

import numpy as np
import torch
import yaml
from bench_learn import peak_host_memory_mb, synchronize, synthetic_obs
from gol_brains.precision import PrecisionPolicy, configure_process_precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", required=True)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--device", help="single device (default: cuda)")
    device_group.add_argument(
        "--devices",
        nargs="+",
        help="devices assigned round-robin, for example cuda:0 cuda:1",
    )
    parser.add_argument(
        "--precision",
        default="config",
        choices=["config", "ieee_fp32", "tf32", "amp_bf16"],
    )
    parser.add_argument("--brains", type=int, default=3)
    parser.add_argument("--updates", type=int, default=20, help="timed updates per brain")
    parser.add_argument("--fill", type=int, default=1400)
    parser.add_argument("--act-every", type=int, default=5)
    parser.add_argument("--headroom", type=float, default=1.0)
    parser.add_argument("--action-deadline-ms", type=float, default=250.0)
    parser.add_argument("--gpu-hourly-cost", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.brains < 1 or args.updates < 1 or args.fill < 1:
        parser.error("--brains, --updates, and --fill must be positive")
    if not 0.0 < args.headroom <= 1.0:
        parser.error("--headroom must be in (0, 1]")

    from gol_brains.base import Brain
    from gol_brains.registry import body_from_config, build_brain

    with open(args.brain) as file:
        cfg: dict[str, Any] = yaml.safe_load(file)
    replay = cfg.setdefault("replay", {})
    training = cfg.setdefault("training", {})
    if args.precision != "config":
        training["precision"] = args.precision
    replay["warmup_steps"] = min(args.fill, int(replay.get("warmup_steps", 100)))
    device_names = args.devices or [args.device or "cuda"]
    devices = [torch.device(name) for name in device_names]
    policies = [PrecisionPolicy.from_config(training, device) for device in devices]
    configure_process_precision(policies)
    fill = max(
        args.fill,
        int(replay.get("warmup_steps", 0)),
        int(replay.get("seq_len", 64)) + int(replay.get("burn_in", 0)) + 2,
    )
    body = body_from_config(cfg)

    brains: list[Brain] = []
    brain_devices: list[torch.device] = []
    observations = []
    for index in range(args.brains):
        device = devices[index % len(devices)]
        brain = build_brain(cfg, seed=index, device=str(device))
        rng = np.random.default_rng(index)
        for _ in range(fill):
            brain.act(synthetic_obs(rng, body))
        observations.append(synthetic_obs(rng, body))
        brains.append(brain)
        brain_devices.append(device)
        print(f"brain {index} on {device} filled ({fill} steps)", flush=True)

    warmup_updates = max(2, args.updates // 5)
    for brain, device in zip(brains, brain_devices, strict=True):
        for _ in range(warmup_updates):
            synchronize(device)
            update = brain.learn()
            synchronize(device)
            if update is None:
                raise RuntimeError("buffer must be past warmup")
    for device in set(brain_devices):
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    start = threading.Barrier(args.brains + 1)
    learn_times: list[list[float]] = [[] for _ in brains]
    worker_errors: list[BaseException] = []
    error_lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            start.wait()
            for _ in range(args.updates):
                device = brain_devices[index]
                synchronize(device)
                began = time.perf_counter()
                update = brains[index].learn()
                synchronize(device)
                if update is None:
                    raise RuntimeError("buffer must remain learnable")
                learn_times[index].append(time.perf_counter() - began)
        except BaseException as error:  # propagate worker failures on the main thread
            with error_lock:
                worker_errors.append(error)

    threads = [
        threading.Thread(target=worker, args=(index,), name=f"bench-learner-{index}")
        for index in range(args.brains)
    ]
    for thread in threads:
        thread.start()
    start.wait()
    began = time.perf_counter()
    action_times: list[float] = []
    probe = 0
    if all(brain.allows_concurrent_learning() for brain in brains):
        while any(thread.is_alive() for thread in threads):
            device = brain_devices[probe]
            synchronize(device)
            action_began = time.perf_counter()
            brains[probe].act(observations[probe])
            synchronize(device)
            action_times.append(time.perf_counter() - action_began)
            probe = (probe + 1) % len(brains)
    for thread in threads:
        thread.join()
    if worker_errors:
        raise RuntimeError("contention benchmark worker failed") from worker_errors[0]
    for device in set(brain_devices):
        synchronize(device)
    wall = time.perf_counter() - began

    aggregate_updates = args.brains * args.updates
    aggregate_rate = aggregate_updates / wall
    timepoints_per_update = int(replay.get("batch_size", 16)) * int(
        replay.get("seq_len", 64)
    )
    train_ratio = float(training.get("train_ratio", 0.25))
    if train_ratio <= 0.0:
        raise ValueError("throughput benchmark requires a positive train ratio")
    sustainable_tick_rate = aggregate_rate * args.act_every / (args.brains * train_ratio)
    result: dict[str, float | str] = {
        "devices": ",".join(str(device) for device in brain_devices),
        "precision": policies[0].mode.value,
        "brains": float(args.brains),
        "aggregate_updates_per_second": aggregate_rate,
        "graded_timepoints_per_second": aggregate_rate * timepoints_per_update,
        "sustainable_ticks_per_second": sustainable_tick_rate,
        "sustainable_ticks_per_second_with_headroom": (
            sustainable_tick_rate * args.headroom
        ),
        "headroom": args.headroom,
        "learn_mean_seconds": statistics.mean(
            elapsed for per_brain in learn_times for elapsed in per_brain
        ),
        "learn_p50_seconds": statistics.median(
            elapsed for per_brain in learn_times for elapsed in per_brain
        ),
        "learn_min_seconds": min(elapsed for per_brain in learn_times for elapsed in per_brain),
        "host_peak_mb": peak_host_memory_mb(),
        "batch_size": float(replay.get("batch_size", 16)),
        "sequence_length": float(replay.get("seq_len", 64)),
        "burn_in": float(replay.get("burn_in", 0)),
        "train_ratio": train_ratio,
    }
    policy_metric_names = (
        "policy_cont_std_mean",
        "policy_cont_std_max",
        "policy_action_abs_mean",
        "policy_action_saturation_frac",
        "policy_rest_sample_frac",
        "affect_viability",
    )
    for index, per_brain in enumerate(learn_times):
        result[f"brain_{index}_updates_per_second"] = 1.0 / statistics.mean(per_brain)
        metrics = brains[index].introspect()
        for name in policy_metric_names:
            if name in metrics:
                result[f"brain_{index}_{name}"] = float(metrics[name])
    slowest_brain_rate = min(
        float(result[f"brain_{index}_updates_per_second"]) for index in range(len(brains))
    )
    result["slowest_brain_sustainable_ticks_per_second_with_headroom"] = (
        slowest_brain_rate * args.act_every / train_ratio * args.headroom
    )
    if action_times:
        deadline = args.action_deadline_ms / 1000.0
        result.update(
            {
                "action_p50_seconds": statistics.median(action_times),
                "action_p95_seconds": float(np.quantile(action_times, 0.95)),
                "action_max_seconds": max(action_times),
                "action_deadline_misses": float(sum(value > deadline for value in action_times)),
                "action_samples": float(len(action_times)),
            }
        )
    else:
        result["action_probe"] = "unsupported_without_async_inference_snapshot"
    cuda_devices = sorted(
        {device for device in brain_devices if device.type == "cuda"}, key=str
    )
    for device in cuda_devices:
        suffix = str(device).replace(":", "_")
        result[f"total_vram_mb_{suffix}"] = (
            torch.cuda.get_device_properties(device).total_memory / 1024**2
        )
        result[f"peak_vram_allocated_mb_{suffix}"] = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        result[f"peak_vram_reserved_mb_{suffix}"] = (
            torch.cuda.max_memory_reserved(device) / 1024**2
        )
    if len(cuda_devices) == 1:
        device = cuda_devices[0]
        result["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        result["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2
    if args.gpu_hourly_cost is not None:
        seconds_per_million = 1_000_000.0 / (aggregate_rate * timepoints_per_update)
        result["cost_per_million_timepoints"] = (
            seconds_per_million / 3600.0 * args.gpu_hourly_cost
        )

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
