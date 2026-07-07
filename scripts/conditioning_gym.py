#!/usr/bin/env python
"""Offline conditioning gym: replay a recorded life through a candidate brain.

The signal-conditioning machinery (normalizer anchoring, LP decay, trickle
anneal, boredom pressure) is a function of learn()-side statistics over
recorded experience — none of it needs a running world. This loads a
checkpointed brain blob (a save's checkpoints/ckpt_*/brains/<id>.pt), replays
learn() N times under a candidate config, and writes per-update metrics
ndjson. beta_08's normalizer re-inflation (curiosity_scaled 0.09→1.86 while
raw LP fell) is exactly the class of failure this screens for in minutes
instead of a 12-hour pod round.

Two modes:
  default        continue the stored mind under its (or an edited) config —
                 weights, normalizers, LP ledgers all resume. Config must
                 match the stored preset.
  --fresh-model  keep only the stored replay buffer and grow a fresh model
                 over the recorded life. This is the screening mode: watch a
                 candidate conditioning stack live through real data from
                 update 0 — and it lets a nano candidate train on a base
                 brain's cloud life (the obs contract is shared).

Honest limits: open-loop. The actor never changes what was lived, so
closed-loop effects (policy chasing its own curiosity into new data) are
invisible here. The gym earns knobs a live round; it doesn't replace one.

Usage:
  uv run python scripts/conditioning_gym.py saves/beta_09 --robot dreamer_000 \
      --brain configs/brain/swift_01_dreamer.yaml --fresh-model \
      --updates 2000 --out /tmp/gym_swift.ndjson \
      --set reward.norm_anchor_samples=500000
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import yaml


def resolve_blob(target: Path, robot: str | None) -> Path:
    """A .pt path passes through; a save dir resolves LATEST + robot id."""
    if target.suffix == ".pt":
        return target
    ckpt_root = target / "checkpoints"
    latest = (ckpt_root / "LATEST").read_text().strip()
    brains_dir = ckpt_root / latest / "brains"
    if not brains_dir.exists():
        raise SystemExit(
            f"{brains_dir} does not exist — this checkpoint was synced without brain "
            "blobs; point at a .pt directly or sync one back"
        )
    blobs = sorted(brains_dir.glob("*.pt"))
    if robot:
        blobs = [b for b in blobs if robot in b.stem]
    if not blobs:
        raise SystemExit(f"no brain blob matching {robot!r} under {brains_dir}")
    return blobs[0]


def apply_sets(cfg: dict[str, Any], sets: list[str]) -> None:
    for item in sets:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", type=Path, help="brain blob .pt, or a save dir (uses LATEST)")
    ap.add_argument("--robot", help="robot id substring when target is a save dir")
    ap.add_argument("--brain", required=True, help="candidate brain config YAML")
    ap.add_argument("--set", action="append", default=[], help="dotted override, a.b.c=value")
    ap.add_argument("--fresh-model", action="store_true", help="keep only the stored buffer")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act-steps", type=int, help="override lived act-steps (drives the "
                    "trickle anneal; default: buffer length, i.e. an adult)")
    ap.add_argument("--out", type=Path, help="metrics ndjson (default: <blob>.gym.ndjson)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    from gol_brains.dreamer.brain import DreamerBrain

    blob_path = resolve_blob(args.target, args.robot)
    with open(args.brain) as fh:
        cfg = yaml.safe_load(fh)
    apply_sets(cfg, args.set)
    print(f"blob: {blob_path}")
    # Cloud blobs pickle CUDA storages, whose unpickling calls torch.load
    # without map_location; pin it to the gym's device so a pod-grown life
    # loads on a CPU-only laptop.
    import functools

    import torch

    orig_load = torch.load
    torch.load = functools.partial(orig_load, map_location=args.device)  # type: ignore[assignment]
    try:
        state = pickle.loads(blob_path.read_bytes())
    finally:
        torch.load = orig_load

    brain = DreamerBrain(cfg, seed=args.seed, device=args.device)
    if args.fresh_model:
        brain.buffer.load_state_dict(state["buffer"])
        if "salience" not in state["buffer"]:
            brain._recompute_salience()  # pre-salience blob: backfill for prioritized replay
        brain._act_steps = args.act_steps if args.act_steps is not None else len(brain.buffer)
    else:
        brain.load_state_dict(state)
        if args.act_steps is not None:
            brain._act_steps = args.act_steps
    print(
        f"mode: {'fresh-model' if args.fresh_model else 'continue'}, "
        f"buffer {len(brain.buffer)} steps, act_steps {brain._act_steps}, "
        f"device {args.device}"
    )

    out_path = args.out or blob_path.with_suffix(".gym.ndjson")
    watch = ("curiosity_scaled", "lp_reward", "lp_stale_frac", "boredom_pressure",
             "boredom", "stimulation", "lp_mix_eff", "loss_model",
             "loss_reward", "reward_head_spike_err", "spike_row_frac")
    began = time.monotonic()
    with open(out_path, "w") as out:
        for i in range(args.updates):
            metrics = brain.learn()
            if metrics is None:
                raise SystemExit("nothing to learn: buffer below warmup_steps")
            out.write(json.dumps({"update": i, **metrics}) + "\n")
            if i % args.log_every == 0 or i == args.updates - 1:
                line = " ".join(
                    f"{k}={metrics[k]:.4g}" for k in watch if k in metrics
                )
                rate = (i + 1) / (time.monotonic() - began)
                print(f"[{i:>6}/{args.updates}] {rate:5.1f} upd/s  {line}", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
