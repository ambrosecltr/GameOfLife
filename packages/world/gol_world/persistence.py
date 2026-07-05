"""Save-dir persistence: manifests, atomic checkpoints, resume.

A save dir is the unit of a world's life:

    saves/alpha/
    ├── manifest.json                 # config snapshot, seed, code version
    ├── checkpoints/ckpt_000030000/   # written atomically (tmp dir + rename)
    │   ├── world.npz                 # blocks, tick, regrowth heap, rng state
    │   └── brains/<agent_id>.pt      # (from M3) brain state, saved at the same tick
    ├── checkpoints/LATEST            # name of the newest complete checkpoint
    ├── events.ndjson                 # append-only, survives crashes
    └── metrics.ndjson

Crash recovery is simply resuming LATEST; at most one checkpoint interval of
sim time is replayed.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from gol_world.config import WorldConfig, dataclass_from_dict
from gol_world.entities import Robot
from gol_world.grid import VoxelGrid
from gol_world.world import World

KEEP_CHECKPOINTS = 3


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def create_save(save_dir: Path, cfg: WorldConfig, run_config: dict[str, Any]) -> None:
    """Initialize a save dir with its manifest. Fails if one already exists."""
    if (save_dir / "manifest.json").exists():
        raise FileExistsError(f"{save_dir} already contains a world")
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    manifest = {
        "world_config": dataclasses.asdict(cfg),
        "run_config": run_config,
        "git_commit": _git_commit(),
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_manifest(save_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((save_dir / "manifest.json").read_text())
    return data


def load_world_config_from_save(save_dir: Path) -> WorldConfig:
    return dataclass_from_dict(WorldConfig, load_manifest(save_dir)["world_config"])


def checkpoint_dir(save_dir: Path, tick: int) -> Path:
    return save_dir / "checkpoints" / f"ckpt_{tick:012d}"


def save_checkpoint(save_dir: Path, world: World, brain_states: dict[str, bytes]) -> Path:
    """Write a checkpoint atomically: tmp dir, fsync-free rename, LATEST update."""
    final = checkpoint_dir(save_dir, world.tick)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    heap = np.array(sorted(world.regrow_heap), dtype=np.int64).reshape(-1, 4)
    np.savez_compressed(
        tmp / "world.npz",
        blocks=world.grid.blocks,
        tick=np.int64(world.tick),
        regrow_heap=heap,
        rng_state=np.frombuffer(json.dumps(world.rng.bit_generator.state).encode(), dtype=np.uint8),
    )
    entities = {"robots": [robot.to_dict() for robot in world.robots.values()]}
    (tmp / "entities.json").write_text(json.dumps(entities, indent=2))
    if brain_states:
        brains_dir = tmp / "brains"
        brains_dir.mkdir()
        for agent_id, blob in brain_states.items():
            (brains_dir / f"{agent_id}.pt").write_bytes(blob)

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    (save_dir / "checkpoints" / "LATEST").write_text(final.name)
    _prune(save_dir)
    return final


def _prune(save_dir: Path) -> None:
    ckpts = sorted((save_dir / "checkpoints").glob("ckpt_*"))
    ckpts = [c for c in ckpts if c.is_dir() and not c.name.endswith(".tmp")]
    for old in ckpts[:-KEEP_CHECKPOINTS]:
        shutil.rmtree(old)


def latest_checkpoint(save_dir: Path) -> Path | None:
    marker = save_dir / "checkpoints" / "LATEST"
    if not marker.exists():
        return None
    path = save_dir / "checkpoints" / marker.read_text().strip()
    return path if path.exists() else None


def load_world(save_dir: Path, ckpt: Path | None = None) -> World:
    """Load the world from a checkpoint (default: LATEST; else a fresh world)."""
    cfg = load_world_config_from_save(save_dir)
    if ckpt is None:
        ckpt = latest_checkpoint(save_dir)
    if ckpt is None:
        return World.new(cfg)
    with np.load(ckpt / "world.npz") as data:
        grid = VoxelGrid(np.ascontiguousarray(data["blocks"]))
        tick = int(data["tick"])
        heap = [
            (int(a), int(b), int(c), int(d)) for a, b, c, d in data["regrow_heap"].reshape(-1, 4)
        ]
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(data["rng_state"].tobytes().decode())
    world = World(cfg, grid, tick=tick, regrow_heap=heap, rng=rng)
    entities_file = ckpt / "entities.json"
    if entities_file.exists():
        entities = json.loads(entities_file.read_text())
        for robot_data in entities.get("robots", []):
            robot = Robot.from_dict(robot_data)
            world.robots[robot.id] = robot
    return world


def load_brain_states(ckpt: Path) -> dict[str, bytes]:
    brains_dir = ckpt / "brains"
    if not brains_dir.exists():
        return {}
    return {p.stem: p.read_bytes() for p in sorted(brains_dir.glob("*.pt"))}
