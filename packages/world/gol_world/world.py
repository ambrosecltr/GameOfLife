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
from gol_world.entities import EV_ATE, EV_DIG_SUCCESS, Robot
from gol_world.grid import VoxelGrid
from gol_world.interface import (
    GRIP_DIG,
    GRIP_EAT,
    GRIP_NOOP,
    GRIP_PLACE,
    SIGNAL_DIM,
    Action,
)

RegrowEntry = tuple[int, int, int, int]  # due_tick, x, y, z
WorldEvent = dict[str, Any]


class World:
    def __init__(
        self,
        cfg: WorldConfig,
        grid: VoxelGrid,
        tick: int = 0,
        regrow_heap: list[RegrowEntry] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.cfg = cfg
        self.grid = grid
        self.tick = tick
        self.regrow_heap: list[RegrowEntry] = regrow_heap if regrow_heap is not None else []
        heapq.heapify(self.regrow_heap)
        # World-event randomness (regrow jitter, future weather). Seeded off the
        # world seed but separate from terrain generation, which already consumed
        # the plain seed.
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed + 1)
        self.robots: dict[str, Robot] = {}
        self.dt = 1.0 / 20.0  # fixed physics timestep (one tick)
        self._events: list[WorldEvent] = []

    @classmethod
    def new(cls, cfg: WorldConfig) -> World:
        world = cls(cfg, terrain.generate(cfg))
        # Depleted bushes from generation get a regrowth due in the first day.
        sx, sy, _ = cfg.size
        empties = np.argwhere(world.grid.blocks == Block.BUSH_EMPTY)
        for x, y, z in empties:
            due = int(world.rng.integers(1, cfg.day_length_ticks))
            heapq.heappush(world.regrow_heap, (due, int(x), int(y), int(z)))
        return world

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
                self.grid.set_block(x, y, z, Block.BUSH_RIPE)

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

    def spawn_robot(self, robot_id: str, brain_name: str) -> Robot:
        pos = self.find_spawn()
        robot = Robot(
            id=robot_id,
            pos=np.array(pos, dtype=np.float64),
            yaw=float(self.rng.uniform(0, 2 * math.pi)),
            brain_name=brain_name,
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

    def _execute_grip(self, robot: Robot) -> None:
        grip = robot.pending_grip
        robot.pending_grip = GRIP_NOOP
        if grip == GRIP_NOOP or robot.dormant:
            return
        eco = self.cfg.economy
        target, air_before = self._faced_cells(robot)

        if grip == GRIP_EAT:
            if target is not None and self.grid.get_block(*target) == Block.BUSH_RIPE:
                self.grid.set_block(*target, Block.BUSH_EMPTY)
                self.schedule_regrow(*target)
                robot.energy = min(eco.energy_max, robot.energy + eco.eat_energy)
                robot.events[EV_ATE] = 1.0
                self._emit("eat", robot, pos=list(target))
            elif robot.held == Block.BUSH_RIPE:
                robot.held = None
                robot.energy = min(eco.energy_max, robot.energy + eco.eat_energy)
                robot.events[EV_ATE] = 1.0
                self._emit("eat", robot, held=True)

        elif grip == GRIP_DIG:
            if (
                target is not None
                and robot.held is None
                and bool(DIGGABLE[self.grid.get_block(*target)])
            ):
                block = self.grid.get_block(*target)
                self.grid.set_block(*target, Block.AIR)
                robot.held = block
                robot.energy = max(0.0, robot.energy - eco.dig_cost)
                robot.events[EV_DIG_SUCCESS] = 1.0
                self._emit("dig", robot, pos=list(target), block=block)

        elif grip == GRIP_PLACE and robot.held is not None and air_before is not None:
            block = robot.held
            self.grid.set_block(*air_before, block)
            if block == Block.BUSH_EMPTY:
                self.schedule_regrow(*air_before)
            robot.held = None
            robot.energy = max(0.0, robot.energy - eco.place_cost)
            self._emit("place", robot, pos=list(air_before), block=block)

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

    def _account_energy(self, robot: Robot, costs: dict[str, float]) -> None:
        eco = self.cfg.economy
        if robot.dormant:
            robot.integrity = max(0.0, robot.integrity - eco.hibernate_integrity_drain)
            return
        drain = (
            eco.basal_drain
            + eco.move_cost * costs["moved"]
            + eco.climb_cost * costs["climbed"]
            + eco.signal_cost * float(np.abs(robot.signal).max())
        )
        if robot.in_water:
            drain *= eco.water_drain_mult
        robot.energy = max(0.0, robot.energy - drain)
        if costs["fall_damage"] > 0:
            robot.integrity = max(
                0.0, robot.integrity - eco.fall_damage_per_block * costs["fall_damage"]
            )
            self._emit("fall_damage", robot, blocks=round(costs["fall_damage"], 2))
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
        dead: list[str] = []
        for robot in self.robots.values():
            self._execute_grip(robot)
            costs = physics.step_robot(self.grid, robot, self.dt)
            self._account_energy(robot, costs)
            robot.age_ticks += 1
            if robot.integrity <= 0.0:
                dead.append(robot.id)
        for robot_id in dead:
            robot = self.robots.pop(robot_id)
            self._drop_scrap(robot)
            self._emit("death", robot, age_ticks=robot.age_ticks)
        if len(self.robots) > 1:
            physics.resolve_robot_overlaps(list(self.robots.values()))
