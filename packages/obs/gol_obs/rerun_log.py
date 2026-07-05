"""Live world visualization via Rerun.

One RerunLogger per running world. Terrain is logged as per-chunk meshes and
only dirty chunks are re-meshed each frame; charts get scalar streams. Both a
"tick" sequence timeline and a "sim_time" duration timeline are set, so runs
are scrubbable in either domain.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rerun as rr
from gol_world.blocks import Block
from gol_world.grid import CHUNK
from gol_world.world import World

from gol_obs.mesher import chunk_mesh

APP_ID = "gameoflife"


class RerunLogger:
    def __init__(
        self,
        world: World,
        tick_rate: int,
        spawn: bool = True,
        save_path: Path | None = None,
    ) -> None:
        self.tick_rate = tick_rate
        rr.init(APP_ID, spawn=spawn)
        if save_path is not None:
            rr.save(str(save_path))
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._set_time(world)
        self._log_all_chunks(world)
        self._sun_radius = max(world.cfg.size[0], world.cfg.size[1]) * 0.6

    def _set_time(self, world: World) -> None:
        rr.set_time("tick", sequence=world.tick)
        rr.set_time("sim_time", duration=world.tick / self.tick_rate)

    def _log_chunk(self, world: World, cx: int, cy: int) -> None:
        path = f"world/terrain/chunk_{cx}_{cy}"
        mesh = chunk_mesh(world.grid.blocks, cx, cy)
        if mesh is None:
            rr.log(path, rr.Clear(recursive=False))
            return
        vertices, triangles, colors = mesh
        rr.log(
            path,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles,
                vertex_colors=colors,
            ),
        )

    def _log_all_chunks(self, world: World) -> None:
        sx, sy, _ = world.cfg.size
        for cx in range(math.ceil(sx / CHUNK)):
            for cy in range(math.ceil(sy / CHUNK)):
                self._log_chunk(world, cx, cy)
        world.grid.consume_dirty_chunks()

    def _log_sun(self, world: World) -> None:
        sx, sy, sz = world.cfg.size
        angle = 2 * math.pi * world.day_fraction
        pos = [
            sx / 2 + self._sun_radius * math.cos(angle),
            sy / 2,
            sz / 2 + self._sun_radius * math.sin(angle),
        ]
        warm = np.array([255, 220, 120], dtype=np.uint8)
        rr.log("world/sun", rr.Points3D([pos], radii=[6.0], colors=[warm]))

    def log_frame(self, world: World) -> None:
        self._set_time(world)
        for cx, cy in world.grid.consume_dirty_chunks():
            self._log_chunk(world, cx, cy)
        self._log_sun(world)
        rr.log("charts/light_level", rr.Scalars([world.light_level]))
        rr.log(
            "charts/ripe_bushes",
            rr.Scalars([float((world.grid.blocks == Block.BUSH_RIPE).sum())]),
        )
