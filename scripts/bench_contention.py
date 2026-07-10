#!/usr/bin/env python
"""Sibling-worker contention probe: K brains learning concurrently, one device.

The learner runs one worker thread per brain; solo learn_seconds lies when
siblings share a device (measured on M1: 3 cpu workers aggregate 7.6 upd/s
where mps manages 4.9 because one GPU queue serializes). This times the real
thing: K threads, each brain paying updates as fast as the device lets it,
reporting per-brain and aggregate updates/s for the pacing identity.

  uv run python scripts/bench_contention.py --brain configs/brain/<round>.yaml \
      --device cuda --brains 3 --updates 20

For Aion 01:
  uv run python scripts/bench_contention.py \
      --brain configs/brain/aion_01_s5.yaml --device cuda --brains 3 --updates 10
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import yaml
from bench_learn import synthetic_obs  # noqa: E402 - sibling script, same dir on sys.path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brain", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--brains", type=int, default=3)
    ap.add_argument("--updates", type=int, default=20, help="timed learn() calls per brain")
    ap.add_argument("--fill", type=int, default=600)
    args = ap.parse_args()

    from gol_brains.base import Brain
    from gol_brains.registry import body_from_config, build_brain

    with open(args.brain) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("replay", {})["warmup_steps"] = 100
    replay = cfg["replay"]
    fill = max(
        args.fill,
        int(replay.get("warmup_steps", 0)),
        int(replay.get("seq_len", 64)) + int(replay.get("burn_in", 0)) + 2,
    )
    body = body_from_config(cfg)

    brains: list[Brain] = []
    for k in range(args.brains):
        brain = build_brain(cfg, seed=k, device=args.device)
        rng = np.random.default_rng(k)
        for _ in range(fill):
            brain.act(synthetic_obs(rng, body))
        brains.append(brain)
        print(f"brain {k} filled ({fill} steps)", flush=True)

    warm = max(2, args.updates // 5)
    per_brain: list[float] = [0.0] * args.brains

    def worker(k: int) -> None:
        for _ in range(warm):
            brains[k].learn()
        began = time.perf_counter()
        for _ in range(args.updates):
            brains[k].learn()
        per_brain[k] = args.updates / (time.perf_counter() - began)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(args.brains)]
    began = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - began
    agg = args.brains * args.updates / wall
    timepoints = int(replay.get("batch_size", 16)) * int(replay.get("seq_len", 64))
    for k, rate in enumerate(per_brain):
        print(f"brain {k}: {rate:.2f} upd/s under contention ({1 / rate:.3f}s/update)")
    print(
        f"aggregate: {agg:.2f} upd/s across {args.brains} workers "
        f"= {agg * timepoints:.0f} replay timepoints/s "
        f"(includes warm={warm} skew; per-brain numbers are the honest ones)"
    )


if __name__ == "__main__":
    main()
