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
from gol_world.blocks import COLOR, Block
from gol_world.grid import CHUNK
from gol_world.interface import (
    NUM_RAY_CLASSES,
    RAY_CLASS_DORMANT,
    RAY_CLASS_ITEM,
    RAY_CLASS_NOTHING,
    RAY_CLASS_ROBOT,
    Observation,
)
from gol_world.sensing import ray_directions
from gol_world.world import World

from gol_obs.mesher import chunk_mesh

APP_ID = "gameoflife"

ROBOT_COLOR = np.array([255, 140, 30], dtype=np.uint8)  # awake: robot orange
DORMANT_COLOR = np.array([110, 90, 70], dtype=np.uint8)

# Ray line colors by hit class: block hits use the block palette.
RAY_CLASS_COLOR = np.zeros((NUM_RAY_CLASSES, 3), dtype=np.uint8)
RAY_CLASS_COLOR[: len(COLOR)] = COLOR
RAY_CLASS_COLOR[RAY_CLASS_ROBOT] = ROBOT_COLOR
RAY_CLASS_COLOR[RAY_CLASS_DORMANT] = DORMANT_COLOR
RAY_CLASS_COLOR[RAY_CLASS_ITEM] = (90, 220, 220)
RAY_CLASS_COLOR[RAY_CLASS_NOTHING] = (70, 70, 80)


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

    def _log_robots(self, world: World) -> None:
        robots = list(world.robots.values())
        if not robots:
            return
        centers = np.array([[r.pos[0], r.pos[1], r.pos[2] + r.body.height / 2] for r in robots])
        half_sizes = np.array(
            [[r.body.width / 2, r.body.width / 2, r.body.height / 2] for r in robots]
        )
        angles = [rr.RotationAxisAngle(axis=(0, 0, 1), radians=r.yaw) for r in robots]
        colors = np.array([DORMANT_COLOR if r.dormant else ROBOT_COLOR for r in robots])
        rr.log(
            "world/robots",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                rotation_axis_angles=angles,
                colors=colors,
                labels=[r.id for r in robots],
                fill_mode="solid",
            ),
        )
        # Heading arrows make yaw legible at a glance.
        origins = centers.copy()
        vectors = np.array([[math.cos(r.yaw) * 1.2, math.sin(r.yaw) * 1.2, 0.0] for r in robots])
        rr.log("world/robots/heading", rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))

    def _log_rays(self, world: World, obs: dict[str, Observation]) -> None:
        for robot_id, o in obs.items():
            robot = world.robots.get(robot_id)
            if robot is None:
                continue
            dirs = ray_directions(
                robot.yaw, robot.body.ray_pitches_deg, robot.body.rays_per_row, robot.body.fov_deg
            )
            depths = o["rays"][:, 0:1].astype(np.float64) * robot.body.ray_range
            ends = robot.eye[None, :] + dirs * depths
            strips = np.stack(
                [np.repeat(robot.eye[None, :], len(ends), axis=0), ends], axis=1
            ).astype(np.float32)
            classes = o["rays"][:, 1:].argmax(axis=1)
            colors = RAY_CLASS_COLOR[classes]
            rr.log(f"world/rays/{robot_id}", rr.LineStrips3D(strips, colors=colors, radii=0.02))

    def log_frame(
        self,
        world: World,
        obs: dict[str, Observation] | None = None,
        introspection: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._set_time(world)
        for cx, cy in world.grid.consume_dirty_chunks():
            self._log_chunk(world, cx, cy)
        self._log_sun(world)
        self._log_robots(world)
        if obs:
            self._log_rays(world, obs)
        rr.log("charts/light_level", rr.Scalars([world.light_level]))
        rr.log("charts/population", rr.Scalars([float(len(world.robots))]))
        rr.log(
            "charts/ripe_bushes",
            rr.Scalars([float((world.grid.blocks == Block.BUSH_RIPE).sum())]),
        )
        for robot in world.robots.values():
            rr.log(f"charts/energy/{robot.id}", rr.Scalars([robot.energy]))
        if introspection:
            for robot_id, metrics in introspection.items():
                for name, value in metrics.items():
                    rr.log(f"charts/brains/{robot_id}/{name}", rr.Scalars([value]))
