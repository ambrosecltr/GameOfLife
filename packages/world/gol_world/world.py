"""The persistent world: grid + time + ecology.

One World instance lives for the entire life of a save dir. There is no reset.
Entities and physics attach here in later milestones; M0 is terrain, the
day/night cycle, and bush regrowth.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from gol_world import terrain
from gol_world.blocks import Block
from gol_world.config import WorldConfig
from gol_world.grid import VoxelGrid

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

    # ------------------------------------------------------------------ step

    def step(self) -> None:
        """Advance the world by one tick. Never resets, never ends."""
        self.tick += 1
        self._process_regrowth()
