"""The persistent world: grid + time + ecology.

One World instance lives for the entire life of a save dir. There is no reset.
Entities and physics attach here in later milestones; M0 is terrain, the
day/night cycle, and bush regrowth.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from gol_world import physics, terrain
from gol_world.blocks import Block
from gol_world.config import WorldConfig
from gol_world.entities import Robot
from gol_world.grid import VoxelGrid
from gol_world.interface import SIGNAL_DIM, Action

RegrowEntry = tuple[int, int, int, int]  # due_tick, x, y, z


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
        # Gripper (dig/place/eat) lands in M2.

    # ------------------------------------------------------------------ step

    def step(self) -> None:
        """Advance the world by one tick. Never resets, never ends."""
        self.tick += 1
        self._process_regrowth()
        for robot in self.robots.values():
            physics.step_robot(self.grid, robot, self.dt)
            robot.age_ticks += 1
        if len(self.robots) > 1:
            physics.resolve_robot_overlaps(list(self.robots.values()))
