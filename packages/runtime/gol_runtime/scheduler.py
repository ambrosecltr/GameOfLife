"""Population management: robots and their brains.

Builds the population from the run config's mix on a fresh world, or rebuilds
brains for the robots already living in a resumed one. Runs the act-step
(observe -> act -> apply) every k ticks, and serializes brain state for the
joint world+brain checkpoint.
"""

from __future__ import annotations

import pickle
import zlib
from typing import Any

from gol_brains.base import Brain
from gol_brains.registry import build_brain, resolve_brain_config
from gol_world.interface import Observation
from gol_world.sensing import observe
from gol_world.world import World

from gol_runtime.config import RunConfig


class Population:
    def __init__(self, world: World, run_cfg: RunConfig) -> None:
        self.world = world
        self.cfg = run_cfg
        self.brains: dict[str, Brain] = {}
        self.last_obs: dict[str, Observation] = {}
        self._specs_by_kind = {
            str(resolve_brain_config(entry["brain"]).get("kind")): entry["brain"]
            for entry in run_cfg.population.mix
        }
        if world.robots:
            self._rebuild_brains()
        else:
            self._spawn_initial()

    def _seed_for(self, robot_id: str) -> int:
        # crc32, not hash(): stable across processes, so resumed brains that
        # reseed anything derive the same stream.
        return (self.world.cfg.seed * 100003 + zlib.crc32(robot_id.encode())) % (2**31)

    def _spawn_initial(self) -> None:
        idx = 0
        for entry in self.cfg.population.mix:
            spec = entry["brain"]
            kind = str(resolve_brain_config(spec).get("kind"))
            for _ in range(int(entry.get("count", 1))):
                robot_id = f"bot_{idx:03d}"
                idx += 1
                self.world.spawn_robot(robot_id, brain_name=kind)
                self.brains[robot_id] = build_brain(spec, seed=self._seed_for(robot_id))

    def _rebuild_brains(self) -> None:
        """On resume: robots already exist; rebuild each brain from its kind."""
        for robot in self.world.robots.values():
            spec = self._specs_by_kind.get(robot.brain_name, {"kind": robot.brain_name})
            self.brains[robot.id] = build_brain(spec, seed=self._seed_for(robot.id))

    def restore_brain_states(self, blobs: dict[str, bytes]) -> None:
        for robot_id, blob in blobs.items():
            if robot_id in self.brains:
                self.brains[robot_id].load_state_dict(pickle.loads(blob))

    def brain_states(self) -> dict[str, bytes]:
        return {rid: pickle.dumps(brain.state_dict()) for rid, brain in self.brains.items()}

    def act_step(self, world: World) -> None:
        """One perception-action cycle for every awake robot."""
        obs = observe(world.grid.blocks, list(world.robots.values()), world.light_level)
        self.last_obs = obs
        for robot_id, o in obs.items():
            action = self.brains[robot_id].act(o)
            world.apply_action(robot_id, action)

    def introspection(self) -> dict[str, dict[str, float]]:
        return {rid: brain.introspect() for rid, brain in self.brains.items()}

    def stats(self) -> dict[str, Any]:
        robots = self.world.robots.values()
        return {
            "population": len(self.world.robots),
            "awake": sum(1 for r in robots if not r.dormant),
        }
