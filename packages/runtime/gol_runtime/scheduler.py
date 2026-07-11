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
from dataclasses import dataclass
from typing import Any

import torch
from gol_brains.base import Brain
from gol_brains.precision import PrecisionPolicy, configure_process_precision
from gol_brains.registry import (
    body_from_config,
    build_brain,
    is_learning_kind,
    resolve_brain_config,
)
from gol_world.interface import Observation
from gol_world.sensing import observe
from gol_world.world import World

from gol_runtime.config import RunConfig

PendingDeath = tuple[Brain, threading.Lock, Observation, bool, int]


class Population:
    def __init__(self, world: World, run_cfg: RunConfig) -> None:
        self.world = world
        self.cfg = run_cfg
        self.brains: dict[str, Brain] = {}
        self.kinds: dict[str, str] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.last_obs: dict[str, Observation] = {}
        self._specs_by_kind = {
            str(resolve_brain_config(entry["brain"]).get("kind")): entry["brain"]
            for entry in run_cfg.population.mix
        }
        self._configure_precision_runtime()
        self._next_idx = 0
        self._respawn_queue: list[tuple[int, str]] = []  # (due_tick, brain kind)
        # Proposal 004 — earned reproduction. Per-kind founder counts (the caps
        # budding fills toward, and the counts scripted anchors are held at) and
        # each parent's last-bud tick (the cooldown clock).
        self._kind_targets: dict[str, int] = {
            str(resolve_brain_config(e["brain"]).get("kind")): int(e.get("count", 0))
            for e in run_cfg.population.mix
        }
        self._last_bud: dict[str, int] = {}
        # Waking from dormancy interrupts the stream of experience (the gap
        # itself is never observed); those brains get wake() before their next
        # act. The default wake() is a stream cut (reset_stream); brains that
        # price the blackout keep the gap as one visible transition instead.
        self._dormant_ids: set[str] = set()
        self._pending_wake: set[str] = set()
        self._dormant_steps: dict[str, int] = {}
        # Stateful inheritance modes keep a dead learning brain here until its
        # replacement is born. "lineage" reuses the same mind; "descendant"
        # copies its learned substrate into a distinct newborn brain.
        self._lineage_stash: dict[str, list[Brain]] = {}
        self._stash_parent_ids: dict[str, list[str]] = {}
        # Deaths whose final observation has not reached the brain yet. Normal
        # tick processing retries without blocking; checkpointing waits for an
        # in-flight update so terminal replay evidence cannot be orphaned.
        self._pending_deaths: list[PendingDeath] = []
        # Sticky per-robot last observation: last_obs holds only THIS step's
        # awake obs (the frame logger's contract), so a hibernating body —
        # the dominant death mode — vanishes from it long before it dies.
        # Its final pre-blackout view survives here for record_death and rides
        # the scheduler checkpoint so a dormant resume cannot lose terminal
        # replay evidence before the body wakes or dies.
        self._death_obs: dict[str, Observation] = {}
        # Action-latch accounting: act-steps where the learner held the lock
        # and the robot repeated its previous command. The artifact grows with
        # world speed (a 200ms update spans more ticks the faster ticks go);
        # act_latched_frac in metrics keeps it visible instead of anecdotal.
        self._act_attempts: dict[str, int] = {}
        self._act_latched: dict[str, int] = {}
        self._act_seconds: dict[str, float] = {}
        if world.robots:
            self._rebuild_brains()
            self._next_idx = (
                max(int(rid.rsplit("_", 1)[1]) for rid in world.robots) + 1 if world.robots else 0
            )
        else:
            self._spawn_initial()

    def _seed_for(self, robot_id: str) -> int:
        # crc32, not hash(): stable across processes, so resumed brains that
        # reseed anything derive the same stream.
        return (self.world.cfg.seed * 100003 + zlib.crc32(robot_id.encode())) % (2**31)

    def _configure_precision_runtime(self) -> None:
        policies = []
        for entry in self.cfg.population.mix:
            spec = entry["brain"]
            if not is_learning_kind(spec):
                continue
            brain_cfg = resolve_brain_config(spec)
            training = dict(brain_cfg.get("training", {}))
            for device in self.cfg.devices.learning_devices():
                policies.append(PrecisionPolicy.from_config(training, torch.device(device)))
        configure_process_precision(policies)

    def _device_for(self, spec: str | dict[str, Any], robot_id: str) -> str:
        """Learning brains live on the learning device (act runs there too;
        a few ms of act latency at 4 Hz is nothing next to update speed).
        Benchmarked on M1 Pro: nano is fastest on cpu, small+ on mps."""
        if not is_learning_kind(spec):
            return self.cfg.devices.inference
        devices = self.cfg.devices.learning_devices()
        try:
            index = int(robot_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            index = zlib.crc32(robot_id.encode())
        return devices[index % len(devices)]

    def _spawn_initial(self) -> None:
        for entry in self.cfg.population.mix:
            kind = str(resolve_brain_config(entry["brain"]).get("kind"))
            for _ in range(int(entry.get("count", 1))):
                self._spawn(kind)

    def _rebuild_brains(self) -> None:
        """On resume: robots already exist; rebuild each brain from its kind."""
        for robot in self.world.robots.values():
            spec = self._specs_by_kind.get(robot.brain_name, {"kind": robot.brain_name})
            # Saved robots carry no BodySpec; restore the body the config says
            # this kind wears (senses must match what the brain was sized for).
            robot.body = body_from_config(resolve_brain_config(spec))
            self.brains[robot.id] = build_brain(
                spec, seed=self._seed_for(robot.id), device=self._device_for(spec, robot.id)
            )
            self.kinds[robot.id] = robot.brain_name
            self.locks[robot.id] = threading.Lock()

    def restore_brain_states(self, blobs: dict[str, bytes]) -> None:
        for robot_id, blob in blobs.items():
            if robot_id == "__scheduler__":
                state = pickle.loads(blob)
                self._next_idx = state["next_idx"]
                self._respawn_queue = state["respawn_queue"]
                self._last_bud = state.get("last_bud", {})
                self._dormant_ids = set(state.get("dormant_ids", ()))
                self._pending_wake = set(state.get("pending_wake", ()))
                self._dormant_steps = dict(state.get("dormant_steps", {}))
                self._death_obs = dict(state.get("death_obs", {}))
                self._stash_parent_ids = {
                    kind: list(parent_ids)
                    for kind, parent_ids in state.get("stash_parent_ids", {}).items()
                }
            elif robot_id.startswith("__lineage__"):
                kind = robot_id.split("__")[2]
                spec = self._specs_by_kind.get(kind, {"kind": kind})
                brain = build_brain(spec, seed=0, device=self._device_for(spec, robot_id))
                brain.load_state_dict(pickle.loads(blob))
                self._lineage_stash.setdefault(kind, []).append(brain)
            elif robot_id in self.brains:
                self.brains[robot_id].load_state_dict(pickle.loads(blob))

    def brain_states(self) -> dict[str, bytes]:
        # A death can land between act boundaries. Reconcile it before the
        # atomic world+brain snapshot so no blob remains keyed to a body that
        # the checkpointed world has already removed.
        self.on_world_tick(self.world)
        self._flush_pending_deaths_for_checkpoint()
        blobs = {}
        for rid, brain in self.brains.items():
            # Block act() while the brain takes a coherent snapshot. Legacy
            # learners share this lock; snapshot-capable learners serialize
            # training internally inside state_dict().
            with self.locks[rid]:
                blobs[rid] = pickle.dumps(brain.state_dict())
        for kind, stash in self._lineage_stash.items():
            for i, brain in enumerate(stash):
                blobs[f"__lineage__{kind}__{i}"] = pickle.dumps(brain.state_dict())
        blobs["__scheduler__"] = pickle.dumps(
            {
                "next_idx": self._next_idx,
                "respawn_queue": self._respawn_queue,
                "last_bud": self._last_bud,
                "dormant_ids": tuple(self._dormant_ids),
                "pending_wake": tuple(self._pending_wake),
                "dormant_steps": dict(self._dormant_steps),
                "death_obs": dict(self._death_obs),
                "stash_parent_ids": {
                    kind: tuple(parent_ids)
                    for kind, parent_ids in self._stash_parent_ids.items()
                },
            }
        )
        return blobs

    def _collect_deaths(self, world: World) -> None:
        """Detach brains for bodies removed by the just-completed world tick."""
        budding = self.cfg.reproduction.mode == "budding"
        died = [rid for rid in self.brains if rid not in world.robots]
        next_act_tick = (
            (world.tick + self.cfg.act_every - 1) // self.cfg.act_every
        ) * self.cfg.act_every
        for rid in died:
            kind = self.kinds.pop(rid, "random_walker")
            brain = self.brains.pop(rid)
            lock = self.locks.pop(rid, None)
            self.last_obs.pop(rid, None)
            last = self._death_obs.pop(rid, None)
            # The body's end is real experience the brain could never sense
            # (dormant bodies don't act, and the death tick removes the robot
            # before observation) — deliver the final state so terminal-aware
            # brains can record it (Brain.record_death). Only meaningful when
            # the brain's memory outlives the body (lineage).
            if (
                last is not None
                and lock is not None
                and self.cfg.population.inherit_weights in ("lineage", "descendant")
            ):
                was_dormant = rid in self._dormant_ids
                self._pending_deaths.append(
                    (brain, lock, last, was_dormant, self._dormant_steps.get(rid, 0))
                )
            if self.cfg.population.inherit_weights in ("lineage", "descendant"):
                self._lineage_stash.setdefault(kind, []).append(brain)
                self._stash_parent_ids.setdefault(kind, []).append(rid)
            self._dormant_ids.discard(rid)
            self._pending_wake.discard(rid)
            self._dormant_steps.pop(rid, None)
            self._act_attempts.pop(rid, None)
            self._act_latched.pop(rid, None)
            self._act_seconds.pop(rid, None)
            self._last_bud.pop(rid, None)
            # Evolving kinds under budding are NOT replaced on the timer — their
            # continuation is earned by a living body budding, not owed on death.
            if not (budding and self._is_evolving(kind)):
                self._respawn_queue.append(
                    (next_act_tick + self.cfg.population.respawn_delay_ticks, kind)
                )

    def on_world_tick(self, world: World) -> None:
        """Reconcile deaths every tick without advancing spawn lifecycle work."""
        self._collect_deaths(world)
        self._deliver_deaths()

    def _process_lifecycle(self, world: World) -> None:
        """Replace the dead. Legacy mode respawns on a timer; budding mode has
        evolving lineages continue only by thriving bodies budding (proposal 004)."""
        budding = self.cfg.reproduction.mode == "budding"
        self.on_world_tick(world)
        if budding:
            self._maintain_budding(world)
        elif len(world.robots) < self.cfg.population.target:
            due = [entry for entry in self._respawn_queue if entry[0] <= world.tick]
            for entry in due:
                self._respawn_queue.remove(entry)
                _, kind = entry
                self._spawn(kind)

    def _is_evolving(self, kind: str) -> bool:
        """Evolving kinds reproduce by budding; scripted anchors (foragers,
        walkers) are instruments, held at their mix count by respawn."""
        return kind not in ("scripted_forager", "random_walker")

    def _is_thriving(self, robot: Any, tick: int) -> bool:
        """A body eligible to bud: awake, past juvenile age, well-fed and intact,
        and off cooldown. No fitness score — a physiological gate on state."""
        r = self.cfg.reproduction
        if robot.dormant or robot.age_ticks < r.min_bud_age:
            return False
        if robot.energy < r.thrive_energy or robot.integrity < r.thrive_integrity:
            return False
        return tick - self._last_bud.get(robot.id, -r.bud_cooldown) >= r.bud_cooldown

    def _maintain_budding(self, world: World) -> None:
        """Keep scripted anchors at count via respawn; let evolving kinds bud
        from thriving parents toward their cap, with a low floor against extinction."""
        for kind, cap in self._kind_targets.items():
            living = [rid for rid, k in self.kinds.items() if k == kind and rid in world.robots]
            if not self._is_evolving(kind):
                if len(living) < cap:  # anchor: refill from the respawn queue
                    due = [e for e in self._respawn_queue if e[1] == kind and e[0] <= world.tick]
                    for entry in due[: cap - len(living)]:
                        self._respawn_queue.remove(entry)
                        self._spawn(kind)
                continue
            # evolving: thriving parents spend surplus to bud toward the cap
            for pid in list(living):
                if len(living) >= cap:
                    break
                if self._is_thriving(world.robots[pid], world.tick):
                    living.append(self._bud(world, pid, kind))
            # extinction guard: a hibernating generation buds nobody
            while len(living) < self.cfg.reproduction.floor:
                living.append(self._spawn(kind))

    def _bud(self, world: World, parent_id: str, kind: str) -> str:
        """A thriving parent buds a child that inherits its mutated genome, and
        pays for it out of its own body (reproduction is costly, not free)."""
        r = self.cfg.reproduction
        prefix = kind.rsplit("_", 1)[-1] if kind else "bot"
        child_id = f"{prefix}_{self._next_idx:03d}"
        self._next_idx += 1
        spec = self._specs_by_kind.get(kind, {"kind": kind})
        body = body_from_config(resolve_brain_config(spec))
        world.spawn_robot(child_id, brain_name=kind, body=body)
        brain = build_brain(
            spec, seed=self._seed_for(child_id), device=self._device_for(spec, child_id)
        )
        parent_brain = self.brains.get(parent_id)
        if parent_brain is not None:
            brain.inherit(parent_brain.state_dict())
        self.brains[child_id] = brain
        self.kinds[child_id] = kind
        self.locks[child_id] = threading.Lock()
        parent = world.robots[parent_id]
        parent.energy_ledger["bud"] += min(parent.energy, r.bud_cost_energy)
        parent.energy = max(0.0, parent.energy - r.bud_cost_energy)
        parent.integrity = max(0.0, parent.integrity - r.bud_cost_integrity)
        self._last_bud[parent_id] = world.tick
        self._last_bud[child_id] = world.tick
        world._emit("bud", parent, child=child_id)
        return child_id

    def _deliver_deaths(self) -> None:
        """Hand each dead body's final observation to its brain, non-blocking.

        The worker thread may be mid-learn() holding the brain's lock; the sim
        tick path does not wait, so an undeliverable death stays pending and
        retries next tick. The worker exits once it sees the body gone, so an
        entry outlives at most one in-flight update. Checkpointing flushes the
        same entry synchronously before serializing the stashed lineage.
        """
        still_pending: list[PendingDeath] = []
        for entry in self._pending_deaths:
            brain, lock, obs, dormant, dormant_steps = entry
            if lock.acquire(blocking=False):
                try:
                    brain.record_death(obs, dormant=dormant, dormant_steps=dormant_steps)
                finally:
                    lock.release()
            else:
                still_pending.append(entry)
        self._pending_deaths = still_pending

    def _flush_pending_deaths_for_checkpoint(self) -> None:
        """Apply every terminal record before an atomic world+brain snapshot."""
        pending, self._pending_deaths = self._pending_deaths, []
        for brain, lock, obs, dormant, dormant_steps in pending:
            with lock:
                brain.record_death(obs, dormant=dormant, dormant_steps=dormant_steps)

    def _flush_death(self, brain: Brain) -> None:
        """Last call for a stashed brain's pending death before it respawns.

        The record must precede the newborn's first step or it would land out
        of order in the replay stream. After a full respawn delay the old
        worker is normally long gone. If its final update is still completing,
        wait here so the newborn cannot overtake the parent's terminal record.
        """
        for entry in list(self._pending_deaths):
            if entry[0] is brain:
                self._pending_deaths.remove(entry)
                _, lock, obs, dormant, dormant_steps = entry
                with lock:
                    brain.record_death(obs, dormant=dormant, dormant_steps=dormant_steps)

    def _spawn(self, kind: str) -> str:
        # Ids carry the kind so every surface (labels, charts, events) reads
        # meaningfully: dreamer_000, forager_001, walker_002.
        prefix = kind.rsplit("_", 1)[-1] if kind else "bot"
        robot_id = f"{prefix}_{self._next_idx:03d}"
        self._next_idx += 1
        spec = self._specs_by_kind.get(kind, {"kind": kind})
        body = body_from_config(resolve_brain_config(spec))
        self.world.spawn_robot(robot_id, brain_name=kind, body=body)
        mode = self.cfg.population.inherit_weights
        stash = self._lineage_stash.get(kind, [])
        parent_ids = self._stash_parent_ids.get(kind, [])
        if mode == "lineage" and stash:
            # Reincarnation: same mind, new body.
            brain = stash.pop(0)
            if parent_ids:
                parent_ids.pop(0)
            self._flush_death(brain)
            brain.reset_stream()
        elif mode == "descendant" and stash:
            # A body is one organism. Its learned weights, replay, optimizer,
            # and temperament seed a distinct descendant; live recurrent state
            # and per-life affect reset through inherit().
            parent = stash.pop(0)
            parent_id = parent_ids.pop(0) if parent_ids else "unknown"
            self._flush_death(parent)
            brain = build_brain(
                spec, seed=self._seed_for(robot_id), device=self._device_for(spec, robot_id)
            )
            brain.inherit(parent.state_dict())
            self.world._emit(
                "inherit",
                self.world.robots[robot_id],
                parent=parent_id,
                mode="descendant",
            )
        else:
            brain = build_brain(
                spec, seed=self._seed_for(robot_id), device=self._device_for(spec, robot_id)
            )
            if mode == "random_living":
                donors = [
                    b
                    for rid2, b in self.brains.items()
                    if self.kinds.get(rid2) == kind and rid2 in self.world.robots
                ]
                if donors:
                    donor = donors[int(self.world.rng.integers(0, len(donors)))]
                    # inherit, not load: newborns keep the donor's mind but
                    # mutate heritable traits (temperament) on the way in.
                    brain.inherit(donor.state_dict())
        self.brains[robot_id] = brain
        self.kinds[robot_id] = kind
        self.locks[robot_id] = threading.Lock()
        return robot_id

    def act_step(self, world: World) -> None:
        """One perception-action cycle for every awake robot.

        The act path never waits: legacy brains latch their previous command if the
        learner holds their lock. Snapshot-capable brains learn concurrently,
        so their lock remains available for every perception-action cycle.
        """
        self._process_lifecycle(world)
        obs = observe(
            world.grid.blocks,
            list(world.robots.values()),
            world.light_level,
            world.active_sounds(),
            toxic_mimic=world.cfg.ecology.toxic_mimic,
            senescence_halflife=world.cfg.economy.senescence_halflife,
        )
        self.last_obs = obs
        self._death_obs.update(obs)
        # Count simulated perception/action opportunities while each mind is
        # offline. Aion uses this duration to decay persistent S5 modes by
        # world time rather than freezing memory while reality advances.
        dormant_now = {r.id for r in world.robots.values() if r.dormant}
        for robot_id in dormant_now:
            self._dormant_steps[robot_id] = self._dormant_steps.get(robot_id, 0) + 1
        # Robots observable again after a dormant spell just woke up.
        self._pending_wake |= self._dormant_ids & set(obs)
        self._dormant_ids = dormant_now
        for robot_id, o in obs.items():
            lock = self.locks[robot_id]
            self._act_attempts[robot_id] = self._act_attempts.get(robot_id, 0) + 1
            if not lock.acquire(blocking=False):
                self._act_latched[robot_id] = self._act_latched.get(robot_id, 0) + 1
                continue
            try:
                act_began = time.monotonic()
                if robot_id in self._pending_wake:
                    dormant_steps = self._dormant_steps.pop(robot_id, 0)
                    self.brains[robot_id].wake(dormant_steps)
                    self._pending_wake.discard(robot_id)
                action = self.brains[robot_id].act(o)
                elapsed = time.monotonic() - act_began
                previous = self._act_seconds.get(robot_id, 0.0)
                self._act_seconds[robot_id] = (
                    elapsed if previous == 0.0 else 0.9 * previous + 0.1 * elapsed
                )
            finally:
                lock.release()
            world.apply_action(robot_id, action)

    def can_fast_forward_dormant(self) -> bool:
        """Whether skipped act boundaries are guaranteed to be no-ops."""
        living_ids = set(self.world.robots)
        if not living_ids or living_ids != set(self.brains):
            return False
        if self._pending_deaths or any(not robot.dormant for robot in self.world.robots.values()):
            return False
        if self.cfg.reproduction.mode == "budding":
            for kind, floor in self._kind_targets.items():
                living = sum(1 for current in self.kinds.values() if current == kind)
                required = floor if not self._is_evolving(kind) else self.cfg.reproduction.floor
                if living < required:
                    return False
        return True

    def next_lifecycle_tick(self) -> int | None:
        if not self._respawn_queue:
            return None
        due = min(entry[0] for entry in self._respawn_queue)
        every = self.cfg.act_every
        return ((max(self.world.tick + 1, due) + every - 1) // every) * every

    def advance_dormant_opportunities(self, start_tick: int, end_tick: int) -> None:
        """Account act boundaries crossed by an event-free world jump."""
        if end_tick < start_tick:
            raise ValueError("dormant opportunity interval cannot run backward")
        opportunities = end_tick // self.cfg.act_every - start_tick // self.cfg.act_every
        if opportunities < 1:
            return
        dormant_ids = set(self.world.robots)
        for robot_id in dormant_ids:
            self._dormant_steps[robot_id] = (
                self._dormant_steps.get(robot_id, 0) + opportunities
            )
        self._dormant_ids = dormant_ids
        self.last_obs = {}

    def sync_learn(self) -> None:
        """Inline learning (--sync): deterministic, single-threaded."""
        for brain in self.brains.values():
            while brain.pending_update_credit() >= 1.0:
                if brain.learn() is None:
                    break

    def learning_ids(self) -> list[str]:
        return [rid for rid, brain in self.brains.items() if brain.target_train_ratio() > 0.0]

    def awake_learning_ids(self) -> list[str]:
        return [
            rid
            for rid in self.learning_ids()
            if rid in self.world.robots and not self.world.robots[rid].dormant
        ]

    def learning_precision_modes(self) -> tuple[str, ...]:
        return tuple(sorted({self.brains[rid].precision_mode() for rid in self.learning_ids()}))

    def introspection(self) -> dict[str, dict[str, float]]:
        out = {}
        for rid, brain in self.brains.items():
            m = dict(brain.introspect())
            attempts = self._act_attempts.get(rid, 0)
            if m and attempts:
                m["act_latched_frac"] = self._act_latched.get(rid, 0) / attempts
                m["action_seconds"] = self._act_seconds.get(rid, 0.0)
            out[rid] = m
        return out

    def stats(self) -> dict[str, Any]:
        robots = self.world.robots.values()
        return {
            "population": len(self.world.robots),
            "awake": sum(1 for r in robots if not r.dormant),
        }


@dataclass(frozen=True)
class LearnerSnapshot:
    aggregate_updates_per_second: float
    debt_by_brain: dict[str, float]
    update_seconds_by_brain: dict[str, float]
    dropped_credit_by_brain: dict[str, float]


class LearnerThread:
    """Background learning, paced to lived experience, one worker per brain.

    Each brain declares a target train_ratio (updates per recorded act-step);
    its worker accrues update debt as act-steps land and pays it down one
    learn() at a time. Workers are per-brain because a single mind's updates
    are inherently serial (each needs the previous one's weights) while
    *siblings* are independent — three brains learn concurrently instead of
    taking turns. A supervisor spawns workers for newborns and reaps them on
    death (stashed lineage brains don't learn, same as before).

    In causal research mode, owed updates remain checkpoint-coherent in each
    brain and the world governor supplies backpressure before lag exceeds its
    configured bound. Credit shedding exists only in explicit ``drop`` mode
    and is exposed as a metric.
    """

    def __init__(
        self,
        population: Population,
        idle_seconds: float = 0.1,
        debt_policy: str = "backpressure",
        max_debt: float = 4.0,
    ) -> None:
        if debt_policy not in ("backpressure", "drop"):
            raise ValueError("debt_policy must be 'backpressure' or 'drop'")
        if max_debt < 1.0:
            raise ValueError("max_debt must be at least one update")
        self.population = population
        self.idle_seconds = idle_seconds
        self.debt_policy = debt_policy
        self.max_debt = max_debt
        self._stop = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._workers: dict[str, threading.Thread] = {}
        self._owed: dict[str, float] = {}
        self._learn_seconds: dict[str, float] = {}
        self._dropped: dict[str, float] = {}
        self._metrics_lock = threading.Lock()
        self.rounds = 0  # supervisor sweeps (observability/tests)

    def start(self) -> None:
        self._supervisor = threading.Thread(target=self._supervise, daemon=True, name="learner")
        self._supervisor.start()

    def stop(self) -> None:
        self._stop.set()
        if self._supervisor is not None:
            self._supervisor.join(timeout=30)
        for worker in list(self._workers.values()):
            worker.join(timeout=30)

    def _accrue(self, rid: str, brain: Brain) -> float:
        """Read checkpointed credit and apply explicit drop policy if selected."""
        owed = brain.pending_update_credit()
        if self.debt_policy == "drop" and owed > self.max_debt:
            dropped = owed - self.max_debt
            brain.drop_update_credit(dropped)
            owed = brain.pending_update_credit()
            with self._metrics_lock:
                self._dropped[rid] = self._dropped.get(rid, 0.0) + dropped
        with self._metrics_lock:
            self._owed[rid] = owed
        return owed

    def snapshot(self) -> LearnerSnapshot:
        living = self.population.learning_ids()
        debts = {
            rid: self.population.brains[rid].pending_update_credit()
            for rid in living
            if rid in self.population.brains
        }
        with self._metrics_lock:
            update_seconds = {
                rid: self._learn_seconds[rid] for rid in debts if rid in self._learn_seconds
            }
            dropped = {rid: self._dropped.get(rid, 0.0) for rid in debts}
        capacity = sum(1.0 / seconds for seconds in update_seconds.values() if seconds > 0.0)
        return LearnerSnapshot(capacity, debts, update_seconds, dropped)

    def has_payable_debt(self) -> bool:
        return any(debt >= 1.0 for debt in self.snapshot().debt_by_brain.values())

    def _supervise(self) -> None:
        while not self._stop.is_set():
            living = set(self.population.learning_ids())
            for rid in living - self._workers.keys():
                worker = threading.Thread(
                    target=self._work, args=(rid,), daemon=True, name=f"learner-{rid}"
                )
                self._workers[rid] = worker
                worker.start()
            # Reap finished workers and drop pacing state for the dead (also
            # covers ids that never got a worker) so nothing accumulates.
            for rid, worker in list(self._workers.items()):
                if not worker.is_alive():
                    self._workers.pop(rid, None)
            with self._metrics_lock:
                tracked = set(self._owed) | set(self._learn_seconds) | set(self._dropped)
            for rid in list(tracked):
                if rid not in self.population.brains:
                    with self._metrics_lock:
                        self._owed.pop(rid, None)
                        self._learn_seconds.pop(rid, None)
                        self._dropped.pop(rid, None)
            self.rounds += 1
            self._stop.wait(0.25)

    def _work(self, rid: str) -> None:
        """Pace one brain's updates to its lived experience, until it dies."""
        while not self._stop.is_set():
            brain = self.population.brains.get(rid)
            lock = self.population.locks.get(rid)
            if brain is None or lock is None:
                # Body died: this worker's watch has ended. Pacing state is
                # pruned by the supervisor; a respawn gets a fresh worker.
                return
            if self._accrue(rid, brain) < 1.0:
                self._stop.wait(self.idle_seconds)
                continue
            if brain.allows_concurrent_learning():
                # The learner mutates training weights while act() reads an
                # immutable published snapshot. The brain's own lock makes
                # checkpoints coherent; the population lock stays free, so
                # embodied control never latches behind a long update.
                began = time.monotonic()
                result = brain.learn()
            else:
                with lock:
                    began = time.monotonic()
                    result = brain.learn()
            elapsed = time.monotonic() - began
            if rid not in self.population.brains:
                return
            if result is None:
                # Credit is scientific state; a temporarily unavailable replay
                # sample must not silently erase it.
                self._stop.wait(self.idle_seconds)
            else:
                with self._metrics_lock:
                    previous = self._learn_seconds.get(rid, 0.0)
                    self._learn_seconds[rid] = (
                        elapsed if previous == 0.0 else 0.9 * previous + 0.1 * elapsed
                    )
                self._accrue(rid, brain)
