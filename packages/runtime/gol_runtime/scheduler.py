"""Population management: robots and their brains.

Builds the population from the run config's mix on a fresh world, or rebuilds
brains for the robots already living in a resumed one. Runs the act-step
(observe -> act -> apply) every k ticks, and serializes brain state for the
joint world+brain checkpoint.
"""

from __future__ import annotations

import pickle
import threading
import zlib
from typing import Any

from gol_brains.base import Brain
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
        # inherit_weights == "lineage": dead learning brains wait here for a
        # new body — weights and replay memory persist across deaths.
        self._lineage_stash: dict[str, list[Brain]] = {}
        # Deaths whose final observation hasn't reached the brain yet: the
        # learner worker may hold the brain's lock mid-update when the body
        # dies, and the sim never waits, so delivery is non-blocking with a
        # retry each act-step. Tuple: brain, lock, last obs, died dormant,
        # and missed perception cycles before death.
        self._pending_deaths: list[tuple[Brain, threading.Lock, Observation, bool, int]] = []
        # Sticky per-robot last observation: last_obs holds only THIS step's
        # awake obs (the frame logger's contract), so a hibernating body —
        # the dominant death mode — vanishes from it long before it dies.
        # Its final pre-blackout view survives here for record_death. Not
        # checkpointed: a resume during a dormant spell sheds that body's
        # death record if it never wakes — skip, don't stall.
        self._death_obs: dict[str, Observation] = {}
        # Action-latch accounting: act-steps where the learner held the lock
        # and the robot repeated its previous command. The artifact grows with
        # world speed (a 200ms update spans more ticks the faster ticks go);
        # act_latched_frac in metrics keeps it visible instead of anecdotal.
        self._act_attempts: dict[str, int] = {}
        self._act_latched: dict[str, int] = {}
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

    def _device_for(self, spec: str | dict[str, Any]) -> str:
        """Learning brains live on the learning device (act runs there too;
        a few ms of act latency at 4 Hz is nothing next to update speed).
        Benchmarked on M1 Pro: nano is fastest on cpu, small+ on mps."""
        return self.cfg.devices.learning if is_learning_kind(spec) else self.cfg.devices.inference

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
                spec, seed=self._seed_for(robot.id), device=self._device_for(spec)
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
            elif robot_id.startswith("__lineage__"):
                kind = robot_id.split("__")[2]
                spec = self._specs_by_kind.get(kind, {"kind": kind})
                brain = build_brain(spec, seed=0, device=self._device_for(spec))
                brain.load_state_dict(pickle.loads(blob))
                self._lineage_stash.setdefault(kind, []).append(brain)
            elif robot_id in self.brains:
                self.brains[robot_id].load_state_dict(pickle.loads(blob))

    def brain_states(self) -> dict[str, bytes]:
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
            }
        )
        return blobs

    def _process_lifecycle(self, world: World) -> None:
        """Replace the dead. Legacy mode respawns on a timer; budding mode has
        evolving lineages continue only by thriving bodies budding (proposal 004)."""
        budding = self.cfg.reproduction.mode == "budding"
        died = [rid for rid in self.brains if rid not in world.robots]
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
                and self.cfg.population.inherit_weights == "lineage"
            ):
                was_dormant = rid in self._dormant_ids
                self._pending_deaths.append(
                    (brain, lock, last, was_dormant, self._dormant_steps.get(rid, 0))
                )
            if self.cfg.population.inherit_weights == "lineage":
                self._lineage_stash.setdefault(kind, []).append(brain)
            self._dormant_ids.discard(rid)
            self._pending_wake.discard(rid)
            self._dormant_steps.pop(rid, None)
            self._act_attempts.pop(rid, None)
            self._act_latched.pop(rid, None)
            self._last_bud.pop(rid, None)
            # Evolving kinds under budding are NOT replaced on the timer — their
            # continuation is earned by a living body budding, not owed on death.
            if not (budding and self._is_evolving(kind)):
                self._respawn_queue.append(
                    (world.tick + self.cfg.population.respawn_delay_ticks, kind)
                )
        self._deliver_deaths()
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
        brain = build_brain(spec, seed=self._seed_for(child_id), device=self._device_for(spec))
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
        never waits, so an undeliverable death stays pending and retries next
        act-step. The worker exits once it sees the body gone, so an entry
        outlives at most one in-flight update. (Not checkpointed: a checkpoint
        landing inside that window sheds one death record — skip, don't stall.)
        """
        still_pending = []
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

    def _flush_death(self, brain: Brain) -> None:
        """Last call for a stashed brain's pending death before it respawns.

        The record must precede the newborn's first step or it would land out
        of order in the replay stream. After a full respawn delay the old
        worker is long gone, so the acquire only fails if something is badly
        wedged — then the record is shed rather than stalling the sim.
        """
        for entry in list(self._pending_deaths):
            if entry[0] is brain:
                self._pending_deaths.remove(entry)
                _, lock, obs, dormant, dormant_steps = entry
                if lock.acquire(blocking=False):
                    try:
                        brain.record_death(obs, dormant=dormant, dormant_steps=dormant_steps)
                    finally:
                        lock.release()

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
        if mode == "lineage" and stash:
            # Reincarnation: same mind, new body.
            brain = stash.pop(0)
            self._flush_death(brain)
            brain.reset_stream()
        else:
            brain = build_brain(spec, seed=self._seed_for(robot_id), device=self._device_for(spec))
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

        The sim never waits: legacy brains latch their previous command if the
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
                if robot_id in self._pending_wake:
                    dormant_steps = self._dormant_steps.pop(robot_id, 0)
                    self.brains[robot_id].wake(dormant_steps)
                    self._pending_wake.discard(robot_id)
                action = self.brains[robot_id].act(o)
            finally:
                lock.release()
            world.apply_action(robot_id, action)

    def sync_learn(self) -> None:
        """Inline learning (--sync): deterministic, single-threaded."""
        for brain in self.brains.values():
            brain.learn()

    def learning_ids(self) -> list[str]:
        return [rid for rid, brain in self.brains.items() if brain.target_train_ratio() > 0.0]

    def introspection(self) -> dict[str, dict[str, float]]:
        out = {}
        for rid, brain in self.brains.items():
            m = dict(brain.introspect())
            attempts = self._act_attempts.get(rid, 0)
            if m and attempts:
                m["act_latched_frac"] = self._act_latched.get(rid, 0) / attempts
            out[rid] = m
        return out

    def stats(self) -> dict[str, Any]:
        robots = self.world.robots.values()
        return {
            "population": len(self.world.robots),
            "awake": sum(1 for r in robots if not r.dormant),
        }


class LearnerThread:
    """Background learning, paced to lived experience, one worker per brain.

    Each brain declares a target train_ratio (updates per recorded act-step);
    its worker accrues update debt as act-steps land and pays it down one
    learn() at a time. Workers are per-brain because a single mind's updates
    are inherently serial (each needs the previous one's weights) while
    *siblings* are independent — three brains learn concurrently instead of
    taking turns. A supervisor spawns workers for newborns and reaps them on
    death (stashed lineage brains don't learn, same as before).

    Backpressure rule: the learner skips, the sim never waits — debt is
    capped, so a world outrunning the learner sheds updates instead of
    banking an unpayable backlog, and a fast learner idles instead of
    over-training stale data.
    """

    # An indebted brain may owe at most this many updates; anything beyond is
    # dropped (that's the "skip" in skip-don't-stall). Sized to hold roughly a
    # full awake burst (~3-8k ticks ≈ 600-1600 act-steps) so that when the
    # body hibernates, the learner works through the day's banked experience —
    # sleep pays the day's debt, which is the Dreamer premise. Too small and
    # dormancy-heavy lives (beta_07: ~93% dormant) starve the learner while
    # the GPU idles; unbounded and a sprinting world banks an unpayable
    # backlog of stale data.
    MAX_DEBT = 1024.0

    def __init__(self, population: Population, idle_seconds: float = 0.1) -> None:
        self.population = population
        self.idle_seconds = idle_seconds
        self._stop = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._workers: dict[str, threading.Thread] = {}
        self._owed: dict[str, float] = {}
        self._seen_acts: dict[str, int] = {}
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
        """Fold newly lived act-steps into rid's update debt."""
        acts = brain.experience_count()
        prev = self._seen_acts.get(rid)
        if prev is None:
            # First sight (fresh spawn or resume): pace from here, no
            # retroactive debt for experience lived before we were watching.
            self._seen_acts[rid] = acts
            self._owed[rid] = 0.0
            return 0.0
        owed = self._owed.get(rid, 0.0) + brain.target_train_ratio() * (acts - prev)
        owed = min(owed, self.MAX_DEBT)
        self._seen_acts[rid] = acts
        self._owed[rid] = owed
        return owed

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
            for rid in list(self._seen_acts.keys() | self._owed.keys()):
                if rid not in self.population.brains:
                    self._seen_acts.pop(rid, None)
                    self._owed.pop(rid, None)
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
                result = brain.learn()
            else:
                with lock:
                    result = brain.learn()
            # learn() runs long (0.25-0.7s on GPU); the body may have died and
            # the supervisor pruned our pacing state in the meantime. End the
            # watch instead of raising on (or resurrecting) the missing key.
            owed = self._owed.get(rid)
            if owed is None:
                return
            if result is None:
                # Nothing learnable yet (warmup): don't bank debt for it.
                self._owed[rid] = 0.0
            else:
                self._owed[rid] = owed - 1.0
