"""Spatial visit heatmap: where does the population actually live?

A slowly decaying (x, y) grid of robot visits — congregation sites, foraging
routes, and dead zones show up here long before they're visible in raw events.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from gol_world.world import World


class VisitHeatmap:
    def __init__(self, size: tuple[int, int, int], half_life_ticks: int = 20_000) -> None:
        self.grid: npt.NDArray[np.float32] = np.zeros((size[0], size[1]), dtype=np.float32)
        # Apply decay in one multiply every `stride` ticks.
        self.stride = 100
        self._decay = float(0.5 ** (self.stride / half_life_ticks))

    def on_tick(self, world: World) -> None:
        sx, sy = self.grid.shape
        for robot in world.robots.values():
            x, y = int(robot.pos[0]), int(robot.pos[1])
            if 0 <= x < sx and 0 <= y < sy:
                self.grid[x, y] += 1.0
        if world.tick % self.stride == 0:
            self.grid *= self._decay

    def image(self) -> npt.NDArray[np.uint8]:
        """Log-scaled grayscale image (y-major for image viewers)."""
        scaled = np.log1p(self.grid)
        top = float(scaled.max())
        if top > 0:
            scaled = scaled / top
        return (scaled.T * 255).astype(np.uint8)
