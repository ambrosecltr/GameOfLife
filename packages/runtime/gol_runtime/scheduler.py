"""Population management: robots and their brains.

Builds the population from the run config's mix on a fresh world, or rebuilds
brains for the robots already living in a resumed one. Runs the act-step
(observe -> act -> apply) every k ticks, and serializes brain state for the
joint world+brain checkpoint.
"""

from __future__ import annotations

import pickle
import threading
import time
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
        self.device = run_cfg.devices.inference
        self.brains: dict[str, Brain] = {}
        self.kinds: dict[str, str] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.last_obs: dict[str, Observation] = {}
        self._specs_by_kind = {
            str(resolve_brain_config(entry["brain"]).get("kind")): entry["brain"]
            for entry in run_cfg.population.mix
        }
        self._next_idx = 0
        self._respawn_queue: list[tuple[int, str]] = []  # (due_tick, brain kind)
        # inherit_weights == "lineage": dead learning brains wait here for a
        # new body — weights and replay memory persist across deaths.
        self._lineage_stash: dict[str, list[Brain]] = {}
        if world.robots:
            self._rebuild_brains()
            self._next_idx = (
                max(int(rid.split("_")[1]) for rid in world.robots) + 1 if world.robots else 0
            )
        else:
            self._spawn_initial()

    def _seed_for(self, robot_id: str) -> int:
        # crc32, not hash(): stable across processes, so resumed brains that
        # reseed anything derive the same stream.
        return (self.world.cfg.seed * 100003 + zlib.crc32(robot_id.encode())) % (2**31)

    def _spawn_initial(self) -> None:
        for entry in self.cfg.population.mix:
            kind = str(resolve_brain_config(entry["brain"]).get("kind"))
            for _ in range(int(entry.get("count", 1))):
                self._spawn(kind)

    def _rebuild_brains(self) -> None:
        """On resume: robots already exist; rebuild each brain from its kind."""
        for robot in self.world.robots.values():
            spec = self._specs_by_kind.get(robot.brain_name, {"kind": robot.brain_name})
            self.brains[robot.id] = build_brain(
                spec, seed=self._seed_for(robot.id), device=self.device
            )
            self.kinds[robot.id] = robot.brain_name
            self.locks[robot.id] = threading.Lock()

    def restore_brain_states(self, blobs: dict[str, bytes]) -> None:
        for robot_id, blob in blobs.items():
            if robot_id == "__scheduler__":
                state = pickle.loads(blob)
                self._next_idx = state["next_idx"]
                self._respawn_queue = state["respawn_queue"]
            elif robot_id.startswith("__lineage__"):
                kind = robot_id.split("__")[2]
                spec = self._specs_by_kind.get(kind, {"kind": kind})
                brain = build_brain(spec, seed=0, device=self.device)
                brain.load_state_dict(pickle.loads(blob))
                self._lineage_stash.setdefault(kind, []).append(brain)
            elif robot_id in self.brains:
                self.brains[robot_id].load_state_dict(pickle.loads(blob))

    def brain_states(self) -> dict[str, bytes]:
        blobs = {}
        for rid, brain in self.brains.items():
            # Lock out the learner so the snapshot is a coherent post-update state.
            with self.locks[rid]:
                blobs[rid] = pickle.dumps(brain.state_dict())
        for kind, stash in self._lineage_stash.items():
            for i, brain in enumerate(stash):
                blobs[f"__lineage__{kind}__{i}"] = pickle.dumps(brain.state_dict())
        blobs["__scheduler__"] = pickle.dumps(
            {"next_idx": self._next_idx, "respawn_queue": self._respawn_queue}
        )
        return blobs

    def _process_lifecycle(self, world: World) -> None:
        """Queue respawns for the dead; spawn queued newborns when due."""
        died = [rid for rid in self.brains if rid not in world.robots]
        for rid in died:
            kind = self.kinds.pop(rid, "random_walker")
            brain = self.brains.pop(rid)
            if self.cfg.population.inherit_weights == "lineage":
                self._lineage_stash.setdefault(kind, []).append(brain)
            self.locks.pop(rid, None)
            self.last_obs.pop(rid, None)
            self._respawn_queue.append((world.tick + self.cfg.population.respawn_delay_ticks, kind))
        if len(world.robots) < self.cfg.population.target:
            due = [entry for entry in self._respawn_queue if entry[0] <= world.tick]
            for entry in due:
                self._respawn_queue.remove(entry)
                _, kind = entry
                self._spawn(kind)

    def _spawn(self, kind: str) -> None:
        robot_id = f"bot_{self._next_idx:03d}"
        self._next_idx += 1
        self.world.spawn_robot(robot_id, brain_name=kind)
        mode = self.cfg.population.inherit_weights
        stash = self._lineage_stash.get(kind, [])
        if mode == "lineage" and stash:
            # Reincarnation: same mind, new body.
            brain = stash.pop(0)
            brain.reset_stream()
        else:
            spec = self._specs_by_kind.get(kind, {"kind": kind})
            brain = build_brain(spec, seed=self._seed_for(robot_id), device=self.device)
            if mode == "random_living":
                donors = [
                    b
                    for rid2, b in self.brains.items()
                    if self.kinds.get(rid2) == kind and rid2 in self.world.robots
                ]
                if donors:
                    donor = donors[int(self.world.rng.integers(0, len(donors)))]
                    brain.load_state_dict(donor.state_dict())
                    brain.reset_stream()
        self.brains[robot_id] = brain
        self.kinds[robot_id] = kind
        self.locks[robot_id] = threading.Lock()

    def act_step(self, world: World) -> None:
        """One perception-action cycle for every awake robot.

        The sim never waits: if a brain's lock is held by the learner right
        now, the robot simply keeps its previously latched command this cycle.
        """
        self._process_lifecycle(world)
        obs = observe(world.grid.blocks, list(world.robots.values()), world.light_level)
        self.last_obs = obs
        for robot_id, o in obs.items():
            lock = self.locks[robot_id]
            if not lock.acquire(blocking=False):
                continue
            try:
                action = self.brains[robot_id].act(o)
            finally:
                lock.release()
            world.apply_action(robot_id, action)

    def sync_learn(self) -> None:
        """Inline learning (--sync): deterministic, single-threaded."""
        for brain in self.brains.values():
            brain.learn()

    def learning_ids(self) -> list[str]:
        return [rid for rid, kind in self.kinds.items() if kind == "dreamer"]

    def introspection(self) -> dict[str, dict[str, float]]:
        return {rid: brain.introspect() for rid, brain in self.brains.items()}

    def stats(self) -> dict[str, Any]:
        robots = self.world.robots.values()
        return {
            "population": len(self.world.robots),
            "awake": sum(1 for r in robots if not r.dormant),
        }


class LearnerThread:
    """Background learning: round-robins learn() across learning brains.

    Backpressure rule: the learner skips, the sim never waits. If learning is
    slow, updates simply get sparser; the world does not stall.
    """

    def __init__(self, population: Population, min_round_seconds: float = 1.0) -> None:
        self.population = population
        self.min_round_seconds = min_round_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rounds = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="learner")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.monotonic()
            did_anything = False
            for rid in self.population.learning_ids():
                if self._stop.is_set():
                    return
                brain = self.population.brains.get(rid)
                lock = self.population.locks.get(rid)
                if brain is None or lock is None:
                    continue
                with lock:
                    result = brain.learn()
                did_anything = did_anything or result is not None
            elapsed = time.monotonic() - began
            sleep_for = max(self.min_round_seconds - elapsed, 0.05 if did_anything else 0.5)
            self._stop.wait(sleep_for)
