#!/usr/bin/env python
"""Benchmark DreamerBrain learn()/act() across device x compile x config.

The number that gates everything local is learn_seconds: achieved train_ratio
= updates/s vs lived act-steps/s, and round 008 proved update density is the
binding variable. This measures it directly and prints the pacing math for a
paced local run (what tick_rate holds ratio 1.0 with K brains).

Usage:
  uv run python scripts/bench_learn.py                        # default grid
  uv run python scripts/bench_learn.py --devices cpu mps --compile off on
  uv run python scripts/bench_learn.py --brain configs/brain/swift_01_dreamer.yaml
  uv run python scripts/bench_learn.py --preset small --updates 30
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import numpy as np
import torch
import yaml


def synthetic_obs(rng: np.random.Generator, body: Any) -> Any:
    from gol_world.interface import (
        EVENTS_DIM,
        NUM_RAY_KINDS,
        PROPRIO_DIM,
        RAY_DIM,
        SOUND_DIM,
        Observation,
    )

    rays = np.zeros((body.num_rays, RAY_DIM), dtype=np.float32)
    rays[:, 0] = rng.random(body.num_rays)
    rays[:, 1:4] = rng.random((body.num_rays, 3)).astype(np.float32)
    rays[np.arange(body.num_rays), 4 + rng.integers(0, NUM_RAY_KINDS, body.num_rays)] = 1.0
    return Observation(
        rays=rays,
        proprio=rng.random(PROPRIO_DIM).astype(np.float32),
        sound=rng.random(SOUND_DIM).astype(np.float32),
        events=np.zeros(EVENTS_DIM, dtype=np.float32),
    )


def build_cfg(args: argparse.Namespace, compile_on: bool) -> dict[str, Any]:
    if args.brain:
        with open(args.brain) as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh)
    else:
        cfg = {"kind": "dreamer", "preset": args.preset}
    cfg.setdefault("replay", {})
    cfg.setdefault("training", {})
    if args.batch:
        cfg["replay"]["batch_size"] = args.batch
    if args.seq:
        cfg["replay"]["seq_len"] = args.seq
    if args.burn_in is not None:
        cfg["replay"]["burn_in"] = args.burn_in
    if args.recent is not None:
        cfg["replay"]["recent"] = args.recent
    if args.optimizer:
        cfg["training"]["optimizer"] = args.optimizer
    cfg["replay"]["warmup_steps"] = 100
    cfg["training"]["compile"] = compile_on
    return cfg


def bench_one(args: argparse.Namespace, device: str, compile_on: bool) -> dict[str, float]:
    from gol_brains.dreamer.brain import DreamerBrain

    cfg = build_cfg(args, compile_on)
    brain = DreamerBrain(cfg, seed=0, device=device)
    rng = np.random.default_rng(0)

    act_times = []
    for i in range(args.fill):
        obs = synthetic_obs(rng, brain.body)
        began = time.perf_counter()
        brain.act(obs)
        if i >= args.fill // 2:  # settle past warmup's random-action branch
            act_times.append(time.perf_counter() - began)

    warm = max(2, args.updates // 5)  # discard compile/allocator warmup
    learn_times = []
    for i in range(args.updates + warm):
        began = time.perf_counter()
        metrics = brain.learn()
        assert metrics is not None, "buffer must be past warmup"
        if i >= warm:
            learn_times.append(time.perf_counter() - began)

    return {
        "learn_mean": statistics.mean(learn_times),
        "learn_p50": statistics.median(learn_times),
        "learn_min": min(learn_times),
        "act_p50": statistics.median(act_times),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brain", help="brain YAML (default: inline preset config)")
    ap.add_argument("--preset", default="nano", choices=["nano", "small", "base"])
    ap.add_argument("--devices", nargs="+", default=["cpu"], choices=["cpu", "mps", "cuda"])
    ap.add_argument("--compile", nargs="+", default=["off"], choices=["off", "on"])
    ap.add_argument("--updates", type=int, default=20, help="timed learn() calls")
    ap.add_argument("--fill", type=int, default=600, help="buffer act-steps before timing")
    ap.add_argument("--batch", type=int, help="replay.batch_size override")
    ap.add_argument("--seq", type=int, help="replay.seq_len override")
    ap.add_argument("--burn-in", type=int, help="replay.burn_in override")
    ap.add_argument("--recent", type=int, help="replay.recent override")
    ap.add_argument("--optimizer", choices=["adam", "muon"])
    ap.add_argument("--brains", type=int, default=3, help="dreamers for the pacing math")
    ap.add_argument("--act-every", type=int, default=5, help="ticks per act (pacing math)")
    ap.add_argument("--ratio", type=float, default=1.0, help="target train ratio (pacing math)")
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())  # respect env tuning if any
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    rows = []
    for device in args.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            print("mps unavailable, skipping")
            continue
        for comp in args.compile:
            label = f"{device}/compile={comp}"
            print(f"--- {label} (building + filling {args.fill} steps...)")
            r = bench_one(args, device, comp == "on")
            rows.append((label, r))
            print(
                f"{label}: learn {r['learn_mean'] * 1e3:.0f}ms mean / "
                f"{r['learn_p50'] * 1e3:.0f}ms p50 / {r['learn_min'] * 1e3:.0f}ms min, "
                f"act {r['act_p50'] * 1e3:.1f}ms p50"
            )

    if rows:
        best_label, best = min(rows, key=lambda kv: kv[1]["learn_p50"])
        ups = 1.0 / best["learn_p50"]
        # One brain's worker is serial; siblings share cores, so scale the
        # aggregate by an assumed 2x (measured on M1: ~1.7-2.2x for 3 workers).
        agg = ups * min(2.0, args.brains)
        tick_rate = agg / args.brains / args.ratio * args.act_every
        print(f"\nbest: {best_label} at {ups:.1f} updates/s/brain")
        print(
            f"pacing: {args.brains} brains at ratio {args.ratio:g}, act_every {args.act_every} "
            f"-> sustainable tick_rate ~= {tick_rate:.0f} t/s "
            f"(assumes ~2x aggregate scaling across workers; verify train_ratio_eff live)"
        )


if __name__ == "__main__":
    main()
