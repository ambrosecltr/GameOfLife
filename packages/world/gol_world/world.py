"""The persistent world: grid + time + ecology.

One World instance lives for the entire life of a save dir. There is no reset.
Entities and physics attach here in later milestones; M0 is terrain, the
day/night cycle, and bush regrowth.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from gol_world import physics, terrain
from gol_world.blocks import DIGGABLE, Block
from gol_world.config import WorldConfig
from gol_world.entities import EV_ATE, EV_DIG_SUCCESS, EV_TOOK_DAMAGE, Robot
from gol_world.grid import VoxelGrid
from gol_world.interface import (
    GAZE_DIM,
    GRIP_DIG,
    GRIP_EAT,
    GRIP_NOOP,
    GRIP_PLACE,
    SIGNAL_DIM,
    Action,
    BodySpec,
)

RegrowEntry = tuple[int, int, int, int]  # due_tick, x, y, z
WorldEvent = dict[str, Any]
TransientSound = tuple[float, float, float, float, int]  # x, y, sig0, sig1, expires_tick

# Involuntary cry patterns on the 2-dim signal channel. Fixed world physics
# (a crash has a sound), not innate vocabulary: nothing stops an agent from
# emitting the same pattern itself.
DEATH_CRY = (-1.0, -1.0)
HURT_CRY = (-0.6, 0.6)

EDIBLE = (Block.BUSH_RIPE, Block.BUSH_TOXIC)
BUSH_BLOCKS = (Block.BUSH_RIPE, Block.BUSH_TOXIC, Block.BUSH_EMPTY)


class World:
    def __init__(
        self,
        cfg: WorldConfig,
        grid: VoxelGrid,
        tick: int = 0,
        regrow_heap: list[RegrowEntry] | None = None,
        rng: np.random.Generator | None = None,
        wither_heap: list[RegrowEntry] | None = None,
        sprout_heap: list[int] | None = None,
    ) -> None:
        self.cfg = cfg
        self.grid = grid
        self.tick = tick
        self.regrow_heap: list[RegrowEntry] = regrow_heap if regrow_heap is not None else []
        heapq.heapify(self.regrow_heap)
        # Bush senescence: standing sites die on the wither heap; each loss
        # (and every held bush that leaves the system) queues a sprout, so
        # standing + held + pending is a conserved bush budget.
        self.wither_heap: list[RegrowEntry] = wither_heap if wither_heap is not None else []
        heapq.heapify(self.wither_heap)
        self.sprout_heap: list[int] = sprout_heap if sprout_heap is not None else []
        heapq.heapify(self.sprout_heap)
        # World-event randomness (regrow jitter, future weather). Seeded off the
        # world seed but separate from terrain generation, which already consumed
        # the plain seed.
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed + 1)
        self.robots: dict[str, Robot] = {}
        self.dt = 1.0 / 20.0  # fixed physics timestep (one tick)
        self._events: list[WorldEvent] = []
        self.transient_sounds: list[TransientSound] = []

    @classmethod
    def new(cls, cfg: WorldConfig) -> World:
        world = cls(cfg, terrain.generate(cfg))
        # Depleted bushes from generation get a regrowth due in the first day.
        sx, sy, _ = cfg.size
        empties = np.argwhere(world.grid.blocks == Block.BUSH_EMPTY)
        for x, y, z in empties:
            due = int(world.rng.integers(1, cfg.day_length_ticks))
            heapq.heappush(world.regrow_heap, (due, int(x), int(y), int(z)))
        world.seed_wither_entries()
        return world

    def seed_wither_entries(self) -> None:
        """Give every standing bush site a death date, uniform across a full
        lifespan so the generation-0 bushes don't all die in one wave."""
        eco = self.cfg.ecology
        if eco.bush_lifespan_ticks <= 0:
            return
        for x, y, z in np.argwhere(np.isin(self.grid.blocks, BUSH_BLOCKS)):
            due = self.tick + int(self.rng.integers(1, eco.bush_lifespan_ticks + 1))
            heapq.heappush(self.wither_heap, (due, int(x), int(y), int(z)))

    # ------------------------------------------------------------------ time

    @property
    def day_fraction(self) -> float:
        return (self.tick % self.cfg.day_length_ticks) / self.cfg.day_length_ticks

    @property
    def sun_height(self) -> float:
        """-1..1; positive during the first half of the day (daytime)."""
        return math.sin(2 * math.pi * self.day_fraction)

    @property
    def is_day(self) -> bool:
        return self.sun_height > 0

    @property
    def light_level(self) -> float:
        """0..1 with quick dawn/dusk ramps (~6% of the day each)."""
        return float(np.clip(self.sun_height * 2.5, 0.0, 1.0))

    def next_dawn_tick(self) -> int:
        day = self.cfg.day_length_ticks
        return (self.tick // day + 1) * day

    # --------------------------------------------------------------- ecology

    def schedule_regrow(self, x: int, y: int, z: int) -> None:
        eco = self.cfg.ecology
        jitter = int(self.rng.integers(-eco.regrow_jitter, eco.regrow_jitter + 1))
        due = self.tick + max(1, eco.regrow_ticks + jitter)
        heapq.heappush(self.regrow_heap, (due, x, y, z))

    def _process_regrowth(self) -> None:
        eco = self.cfg.ecology
        while self.regrow_heap and self.regrow_heap[0][0] <= self.tick:
            due, x, y, z = heapq.heappop(self.regrow_heap)
            if eco.regrow_daytime_only and not self.is_day:
                # Spread over the first quarter of the next day so a whole
                # night's backlog doesn't pop in one tick (and never lands in
                # the following night).
                morning = max(1, self.cfg.day_length_ticks // 4)
                dawn = self.next_dawn_tick() + int(self.rng.integers(0, morning))
                heapq.heappush(self.regrow_heap, (dawn, x, y, z))
                continue
            if self.grid.get_block(x, y, z) == Block.BUSH_EMPTY:
                toxic = float(self.rng.random()) < eco.toxic_fraction
                self.grid.set_block(x, y, z, Block.BUSH_TOXIC if toxic else Block.BUSH_RIPE)

    def schedule_wither(self, x: int, y: int, z: int) -> None:
        """Stamp a new bush site with its death date."""
        eco = self.cfg.ecology
        if eco.bush_lifespan_ticks <= 0:
            return
        jitter = int(self.rng.integers(-eco.bush_lifespan_jitter, eco.bush_lifespan_jitter + 1))
        due = self.tick + max(1, eco.bush_lifespan_ticks + jitter)
        heapq.heappush(self.wither_heap, (due, x, y, z))

    def schedule_sprout(self) -> None:
        """Queue a replacement bush for one that left the world for good
        (withered, eaten from the hand, spoiled, or died with its carrier)."""
        eco = self.cfg.ecology
        jitter = int(self.rng.integers(-eco.regrow_jitter, eco.regrow_jitter + 1))
        heapq.heappush(self.sprout_heap, self.tick + max(1, eco.regrow_ticks + jitter))

    def _process_wither(self) -> None:
        eco = self.cfg.ecology
        if eco.bush_lifespan_ticks <= 0:
            return
        while self.wither_heap and self.wither_heap[0][0] <= self.tick:
            due, x, y, z = heapq.heappop(self.wither_heap)
            # Stale entries (the bush was dug up, or died early to an older
            # stamp on a replanted cell) just drop; the site travels as a
            # held item and is accounted for when it leaves the system.
            if self.grid.get_block(x, y, z) in BUSH_BLOCKS:
                self.grid.set_block(x, y, z, Block.AIR)
                self._emit("wither", pos=[x, y, z])
                self.schedule_sprout()

    def _process_sprouts(self) -> None:
        eco = self.cfg.ecology
        while self.sprout_heap and self.sprout_heap[0] <= self.tick:
            if eco.regrow_daytime_only and not self.is_day:
                heapq.heappop(self.sprout_heap)
                morning = max(1, self.cfg.day_length_ticks // 4)
                dawn = self.next_dawn_tick() + int(self.rng.integers(0, morning))
                heapq.heappush(self.sprout_heap, dawn)
                continue
            heapq.heappop(self.sprout_heap)
            site = self._find_sprout_site()
            if site is None:  # nowhere to grow right now; try again tomorrow
                heapq.heappush(self.sprout_heap, self.tick + self.cfg.day_length_ticks)
                continue
            x, y, z = site
            toxic = float(self.rng.random()) < eco.toxic_fraction
            self.grid.set_block(x, y, z, Block.BUSH_TOXIC if toxic else Block.BUSH_RIPE)
            self.schedule_wither(x, y, z)
            self._emit("sprout", pos=[x, y, z])

    def _find_sprout_site(self) -> tuple[int, int, int] | None:
        """A grass column with air above: near an existing bush (patches stay
        spottable by ray fans) or anywhere, per sprout_clump_bias."""
        eco = self.cfg.ecology
        sx, sy, sz = self.cfg.size
        near: tuple[int, int] | None = None
        if float(self.rng.random()) < eco.sprout_clump_bias:
            bushes = np.argwhere(np.isin(self.grid.blocks, BUSH_BLOCKS))
            if len(bushes):
                bx, by, _ = bushes[int(self.rng.integers(0, len(bushes)))]
                near = (int(bx), int(by))
        for _ in range(64):
            if near is not None:
                x = near[0] + int(self.rng.integers(-2, 3))
                y = near[1] + int(self.rng.integers(-2, 3))
                if not (0 <= x < sx and 0 <= y < sy):
                    continue
            else:
                x = int(self.rng.integers(1, sx - 1))
                y = int(self.rng.integers(1, sy - 1))
            h = self.grid.column_height(x, y)
            z = h + 1
            if (
                h > 0
                and z < sz
                and self.grid.get_block(x, y, h) == Block.GRASS
                and self.grid.get_block(x, y, z) == Block.AIR
                and not self._cell_overlaps_robot(x, y, z)
            ):
                return (x, y, z)
            near = None  # clump spot failed; fall back to anywhere
        return None

    def _cell_overlaps_robot(self, x: int, y: int, z: int) -> bool:
        cell_lo = np.array([x, y, z], dtype=np.float64)
        for robot in self.robots.values():
            lo, hi = robot.aabb
            if bool(np.all(lo < cell_lo + 1.0) and np.all(hi > cell_lo)):
                return True
        return False

    # ---------------------------------------------------------------- robots

    def find_spawn(self) -> tuple[float, float, float]:
        """A dry grass spot with headroom, chosen with the world's own rng."""
        sx, sy, sz = self.cfg.size
        for _ in range(4096):
            x = int(self.rng.integers(2, sx - 2))
            y = int(self.rng.integers(2, sy - 2))
            h = self.grid.column_height(x, y)
            if h < 1 or h + 3 >= sz:
                continue
            if self.grid.get_block(x, y, h) != Block.GRASS:
                continue
            if any(self.grid.get_block(x, y, h + dz) != Block.AIR for dz in (1, 2)):
                continue
            return (x + 0.5, y + 0.5, float(h + 1))
        raise RuntimeError("no spawnable grass found (world all water/rock?)")

    def spawn_robot(self, robot_id: str, brain_name: str, body: BodySpec | None = None) -> Robot:
        pos = self.find_spawn()
        robot = Robot(
            id=robot_id,
            pos=np.array(pos, dtype=np.float64),
            yaw=float(self.rng.uniform(0, 2 * math.pi)),
            brain_name=brain_name,
            body=body or BodySpec(),
            energy=self.cfg.economy.energy_max,
            integrity=self.cfg.economy.integrity_max,
        )
        robot.fall_peak_z = float(pos[2])
        self.robots[robot_id] = robot
        self._emit("spawn", robot, brain=brain_name)
        return robot

    def apply_action(self, robot_id: str, action: Action) -> None:
        """Latch a brain's command; physics applies it every tick until the
        next act-step. Gripper actions are executed in the next step()."""
        robot = self.robots[robot_id]
        if robot.dormant:
            return
        robot.drive[:] = np.clip(np.asarray(action.drive, dtype=np.float64), -1.0, 1.0)
        if action.signal is not None:
            robot.signal[:] = np.clip(
                np.asarray(action.signal, dtype=np.float64)[:SIGNAL_DIM], -1.0, 1.0
            )
        else:
            robot.signal[:] = 0.0
        if action.gaze is not None:
            robot.gaze[:] = np.clip(np.asarray(action.gaze, dtype=np.float64)[:GAZE_DIM], -1.0, 1.0)
        else:
            robot.gaze[:] = 0.0
        robot.pending_grip = int(action.gripper)

    # --------------------------------------------------------------- gripper

    def _faced_cells(
        self, robot: Robot
    ) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
        """(first non-air cell, last air cell before it) along the gaze, within reach."""
        eye = robot.eye
        dx, dy = math.cos(robot.yaw), math.sin(robot.yaw)
        last_air: tuple[int, int, int] | None = None
        prev_cell: tuple[int, int, int] | None = None
        steps = max(2, int(robot.body.reach * 4))
        for i in range(1, steps + 1):
            t = robot.body.reach * i / steps
            cell = (int(eye[0] + dx * t), int(eye[1] + dy * t), int(eye[2]))
            if cell == prev_cell:
                continue
            prev_cell = cell
            if not self.grid.in_bounds(*cell):
                break
            if self.grid.get_block(*cell) == Block.AIR:
                last_air = cell
            else:
                return cell, last_air
        return None, last_air

    def _faced_edible(self, robot: Robot) -> tuple[int, int, int] | None:
        """First edible cell along the gaze within reach, also checking one
        block below and above eye level — a mouth can dip to a bush at the
        feet. The eye-height-only scan of _faced_cells left robots starving in
        front of downhill bushes their pitched rays could plainly see.
        A solid non-edible block at eye level still blocks the reach."""
        eye = robot.eye
        dx, dy = math.cos(robot.yaw), math.sin(robot.yaw)
        z_eye = int(eye[2])
        steps = max(2, int(robot.body.reach * 4))
        prev_xy: tuple[int, int] | None = None
        for i in range(1, steps + 1):
            t = robot.body.reach * i / steps
            xy = (int(eye[0] + dx * t), int(eye[1] + dy * t))
            if xy == prev_xy:
                continue
            prev_xy = xy
            blocked = False
            for dz in (0, -1, 1):
                cell = (xy[0], xy[1], z_eye + dz)
                if not self.grid.in_bounds(*cell):
                    continue
                block = self.grid.get_block(*cell)
                if block in EDIBLE:
                    return cell
                if dz == 0 and block != Block.AIR:
                    blocked = True
            if blocked:
                return None
        return None

    def _faced_dormant(self, robot: Robot) -> Robot | None:
        """Nearest dormant robot whose body crosses the gaze ray within reach."""
        eye = robot.eye
        direction = np.array([math.cos(robot.yaw), math.sin(robot.yaw), 0.0])
        safe = np.where(direction == 0, 1e-12, direction)
        best: Robot | None = None
        best_t = robot.body.reach
        for other in self.robots.values():
            if other.id == robot.id or not other.dormant:
                continue
            lo, hi = other.aabb
            t1 = (lo - eye) / safe
            t2 = (hi - eye) / safe
            tmin = float(np.minimum(t1, t2).max())
            tmax = float(np.maximum(t1, t2).min())
            if tmax >= max(tmin, 0.0) and tmin <= best_t:
                best = other
                best_t = tmin
        return best

    def _execute_grip(self, robot: Robot) -> None:
        grip = robot.pending_grip
        robot.pending_grip = GRIP_NOOP
        if grip == GRIP_NOOP or robot.dormant:
            return
        eco = self.cfg.economy
        target, air_before = self._faced_cells(robot)

        if grip == GRIP_EAT:
            meal = self._faced_edible(robot)
            if meal is not None:
                toxic = self.grid.get_block(*meal) == Block.BUSH_TOXIC
                self.grid.set_block(*meal, Block.BUSH_EMPTY)
                self.schedule_regrow(*meal)
                self._ingest(robot, toxic, pos=list(meal))
            elif robot.held is not None and robot.held in EDIBLE:
                toxic = robot.held == Block.BUSH_TOXIC
                robot.held = None
                self.schedule_sprout()  # the carried bush leaves the system
                self._ingest(robot, toxic, held=True)

        elif grip == GRIP_DIG:
            if (
                target is not None
                and robot.held is None
                and bool(DIGGABLE[self.grid.get_block(*target)])
            ):
                block = self.grid.get_block(*target)
                self.grid.set_block(*target, Block.AIR)
                robot.held = block
                robot.energy_ledger["dig"] += min(robot.energy, eco.dig_cost)
                robot.energy = max(0.0, robot.energy - eco.dig_cost)
                robot.events[EV_DIG_SUCCESS] = 1.0
                self._emit("dig", robot, pos=list(target), block=block)

        elif grip == GRIP_PLACE and robot.held is not None:
            # Holding food and facing a dormant body: the food goes into them,
            # not onto the ground. The only way to help someone who can't act —
            # and toxic food goes in just the same. Rescue and murder share a verb.
            recipient = self._faced_dormant(robot) if robot.held in EDIBLE else None
            if recipient is not None:
                toxic = robot.held == Block.BUSH_TOXIC
                robot.held = None
                self.schedule_sprout()  # eaten by the sleeper: leaves the system
                self._emit("feed", robot, to=recipient.id)
                self._ingest(recipient, toxic)
            elif air_before is not None:
                block = robot.held
                self.grid.set_block(*air_before, block)
                if block == Block.BUSH_EMPTY:
                    self.schedule_regrow(*air_before)
                if block in BUSH_BLOCKS:
                    self.schedule_wither(*air_before)  # a transplant restarts its clock
                robot.held = None
                robot.energy_ledger["place"] += min(robot.energy, eco.place_cost)
                robot.energy = max(0.0, robot.energy - eco.place_cost)
                self._emit("place", robot, pos=list(air_before), block=block)

    def _ingest(self, robot: Robot, toxic: bool, **evdata: Any) -> None:
        """A meal lands in a robot: nourishing or poisonous, eaten or fed."""
        eco = self.cfg.economy
        if toxic:
            banked = min(eco.energy_max, robot.energy + eco.toxic_energy)
            robot.energy_ledger["eaten"] += banked - robot.energy
            robot.energy = banked
            robot.integrity = max(0.0, robot.integrity - eco.toxic_integrity_damage)
            robot.ledger["poison"] += eco.toxic_integrity_damage
            robot.events[EV_ATE] = 1.0
            robot.events[EV_TOOK_DAMAGE] = 1.0
            self._cry(robot, HURT_CRY, self.cfg.sounds.hurt_cry_ticks)
            self._emit("poisoned", robot, **evdata)
        else:
            banked = min(eco.energy_max, robot.energy + eco.eat_energy)
            robot.energy_ledger["eaten"] += banked - robot.energy
            robot.energy = banked
            robot.events[EV_ATE] = 1.0
            self._emit("eat", robot, **evdata)

    # ---------------------------------------------------------------- sounds

    def _cry(self, robot: Robot, pattern: tuple[float, float], duration_ticks: int) -> None:
        if duration_ticks <= 0:
            return
        x, y = float(robot.pos[0]), float(robot.pos[1])
        self.transient_sounds.append((x, y, pattern[0], pattern[1], self.tick + duration_ticks))

    def active_sounds(self) -> list[tuple[float, float, float, float]]:
        """Unexpired transient sounds as (x, y, sig0, sig1) for sensing."""
        return [
            (x, y, s0, s1) for x, y, s0, s1, expires in self.transient_sounds if expires > self.tick
        ]

    # ---------------------------------------------------------------- events

    def _emit(self, kind: str, robot: Robot | None = None, **data: Any) -> None:
        event: WorldEvent = {"tick": self.tick, "kind": kind, **data}
        if robot is not None:
            event["robot"] = robot.id
            event.setdefault("pos", [round(float(p), 2) for p in robot.pos])
        self._events.append(event)

    def consume_events(self) -> list[WorldEvent]:
        events, self._events = self._events, []
        return events

    # --------------------------------------------------------------- economy

    def _actuation(self, robot: Robot) -> float:
        """Energy brownout: full actuation above the threshold, linear fade to
        the floor at zero. Depletion degrades the body's own dynamics, giving
        the world model a smooth gradient to learn before stasis hits."""
        eco = self.cfg.economy
        if eco.brownout_threshold <= 0.0 or robot.energy >= eco.brownout_threshold:
            return 1.0
        frac = robot.energy / eco.brownout_threshold
        return eco.brownout_floor + (1.0 - eco.brownout_floor) * frac

    def _account_energy(self, robot: Robot, costs: dict[str, float]) -> None:
        eco = self.cfg.economy
        if robot.dormant:
            # Hibernation trades integrity for time; sunlight (or a feeding
            # peer) is the only way back out. It is at least restful.
            robot.integrity = max(0.0, robot.integrity - eco.hibernate_integrity_drain)
            robot.ledger["hibernation"] += eco.hibernate_integrity_drain
            gained = min(eco.energy_max, robot.energy + eco.solar_trickle * self.light_level)
            robot.energy_ledger["solar"] += gained - robot.energy
            robot.energy = gained
            robot.fatigue = max(0.0, robot.fatigue - eco.fatigue_recover)
            if robot.energy >= eco.wake_energy:
                robot.dormant = False
                self._emit("wake", robot)
            return
        activity = float(np.abs(robot.drive).max())
        resting = activity < eco.rest_drive_threshold
        # Sleep is a night thing, discovered rather than scripted: the same
        # stillness recovers faster in the dark.
        night_factor = 1.0 + eco.night_rest_bonus * (1.0 - self.light_level)
        was_exhausted = robot.fatigue >= eco.exhaustion_threshold
        if resting:
            robot.fatigue = max(0.0, robot.fatigue - eco.fatigue_recover * night_factor)
        else:
            robot.fatigue = min(
                1.0, robot.fatigue + eco.fatigue_rise_base + eco.fatigue_rise_active * activity
            )
        exhausted = robot.fatigue >= eco.exhaustion_threshold
        if exhausted and not was_exhausted:
            self._emit("exhausted", robot)
        parts = {
            "basal": eco.basal_drain * (eco.rest_basal_mult if resting else 1.0),
            "move": eco.move_cost * costs["moved"],
            "turn": eco.turn_cost * costs["turned"],
            "climb": eco.climb_cost * costs["climbed"],
            "signal": eco.signal_cost * float(np.abs(robot.signal).max()),
        }
        drain = sum(parts.values())
        if exhausted:
            parts["exhaustion"] = drain * (eco.exhaustion_drain_mult - 1.0)
            drain *= eco.exhaustion_drain_mult
            robot.integrity = max(0.0, robot.integrity - eco.exhaustion_integrity_drain)
            robot.ledger["exhaustion"] += eco.exhaustion_integrity_drain
        if robot.in_water:
            parts["water"] = drain * (eco.water_drain_mult - 1.0)
            drain *= eco.water_drain_mult
        for cause, cost in parts.items():
            robot.energy_ledger[cause] += cost
        robot.energy = max(0.0, robot.energy - drain)
        if costs["fall_damage"] > 0:
            fall = eco.fall_damage_per_block * costs["fall_damage"]
            robot.integrity = max(0.0, robot.integrity - fall)
            robot.ledger["fall"] += fall
            self._emit("fall_damage", robot, blocks=round(costs["fall_damage"], 2))
            self._cry(robot, HURT_CRY, self.cfg.sounds.hurt_cry_ticks)
        # Chronic wear, and repair funded by energy surplus. Repair efficiency
        # halves per senescence half-life of age: young bodies mend, old ones
        # can no longer keep up with their own wear — death arrives when the
        # curves cross, sooner for lives that banked more unrepaired damage.
        robot.integrity = max(0.0, robot.integrity - eco.awake_wear)
        robot.ledger["wear"] += eco.awake_wear
        if (
            eco.repair_rate > 0.0
            and robot.integrity < eco.integrity_max
            and robot.energy > eco.repair_threshold
        ):
            efficiency = (
                0.5 ** (robot.age_ticks / eco.senescence_halflife)
                if eco.senescence_halflife > 0
                else 1.0
            )
            rest_mult = eco.rest_repair_mult * night_factor if resting else 1.0
            rate = eco.repair_rate * efficiency * rest_mult
            amount = min(rate, eco.integrity_max - robot.integrity)
            if eco.repair_energy_per_point > 0:
                surplus = robot.energy - eco.repair_threshold
                amount = min(amount, surplus / eco.repair_energy_per_point)
            if amount > 0.0:
                robot.integrity += amount
                robot.energy -= amount * eco.repair_energy_per_point
                robot.ledger["repaired"] += amount
                robot.energy_ledger["repair"] += amount * eco.repair_energy_per_point
        if robot.energy <= 0.0 and not robot.dormant:
            robot.dormant = True
            self._emit("hibernate", robot)

    def _drop_scrap(self, robot: Robot) -> None:
        x, y = int(robot.pos[0]), int(robot.pos[1])
        for z in range(int(robot.pos[2]), min(int(robot.pos[2]) + 4, self.cfg.size[2])):
            if self.grid.in_bounds(x, y, z) and self.grid.get_block(x, y, z) == Block.AIR:
                self.grid.set_block(x, y, z, Block.SCRAP)
                return

    # ------------------------------------------------------------------ step

    def step(self) -> None:
        """Advance the world by one tick. Never resets, never ends."""
        self.tick += 1
        self._process_regrowth()
        self._process_wither()
        self._process_sprouts()
        if self.transient_sounds:
            self.transient_sounds = [s for s in self.transient_sounds if s[4] > self.tick]
        eco = self.cfg.ecology
        dead: list[str] = []
        for robot in self.robots.values():
            self._execute_grip(robot)
            # A carried bush perishes; the slot goes back to the world.
            if robot.held is not None and robot.held in BUSH_BLOCKS:
                robot.held_age_ticks += 1
                if 0 < eco.held_spoil_ticks <= robot.held_age_ticks:
                    robot.held = None
                    robot.held_age_ticks = 0
                    self.schedule_sprout()
                    self._emit("spoil", robot)
            else:
                robot.held_age_ticks = 0
            costs = physics.step_robot(
                self.grid,
                robot,
                self.dt,
                self._actuation(robot),
                water_speed_mult=self.cfg.economy.water_speed_mult,
            )
            self._account_energy(robot, costs)
            robot.age_ticks += 1
            if robot.integrity <= 0.0:
                dead.append(robot.id)
        for robot_id in dead:
            robot = self.robots.pop(robot_id)
            if robot.held is not None and robot.held in BUSH_BLOCKS:
                self.schedule_sprout()  # the carried bush dies with its carrier
            self._drop_scrap(robot)
            self._cry(robot, DEATH_CRY, self.cfg.sounds.death_cry_ticks)
            self._emit(
                "death",
                robot,
                age_ticks=robot.age_ticks,
                ledger={k: round(v, 2) for k, v in robot.ledger.items()},
                energy_ledger={k: round(v, 2) for k, v in robot.energy_ledger.items()},
            )
        if len(self.robots) > 1:
            physics.resolve_robot_overlaps(list(self.robots.values()))
