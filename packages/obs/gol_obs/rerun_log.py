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
import numpy.typing as npt
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

from gol_obs.blueprint import build_blueprint
from gol_obs.mesher import chunk_mesh

APP_ID = "gameoflife"

ROBOT_COLOR = np.array([255, 140, 30], dtype=np.uint8)  # ray-hit class color
DORMANT_COLOR = np.array([110, 90, 70], dtype=np.uint8)

# Stable per-robot identity colors, shared by the 3D body and every chart line.
ROBOT_PALETTE = np.array(
    [
        [230, 80, 60],  # red
        [70, 160, 235],  # blue
        [110, 205, 90],  # green
        [240, 190, 60],  # gold
        [180, 110, 235],  # violet
        [250, 140, 190],  # pink
        [90, 220, 210],  # teal
        [250, 160, 60],  # orange
        [165, 165, 175],  # gray
        [200, 220, 120],  # lime
    ],
    dtype=np.uint8,
)


def robot_color(robot_id: str) -> npt.NDArray[np.uint8]:
    try:
        idx = int(robot_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        idx = abs(hash(robot_id))
    color: npt.NDArray[np.uint8] = ROBOT_PALETTE[idx % len(ROBOT_PALETTE)]
    return color


# World events as feed entries: severity + a human sentence.
_EVENT_LEVELS = {
    "death": rr.TextLogLevel.ERROR,
    "poisoned": rr.TextLogLevel.WARN,
    "hibernate": rr.TextLogLevel.WARN,
    "fall_damage": rr.TextLogLevel.WARN,
    "exhausted": rr.TextLogLevel.WARN,
    "dig": rr.TextLogLevel.TRACE,
    "place": rr.TextLogLevel.TRACE,
}


def _event_text(event: dict[str, object]) -> str:
    kind = str(event["kind"])
    rid = event.get("robot", "?")
    if kind == "spawn":
        return f"{rid} spawned ({event.get('brain', '?')})"
    if kind == "eat":
        return f"{rid} ate a berry"
    if kind == "poisoned":
        return f"{rid} ate TOXIC food — integrity damage"
    if kind == "feed":
        return f"{rid} fed {event.get('to', '?')}"
    if kind == "hibernate":
        return f"{rid} collapsed — out of energy, hibernating"
    if kind == "wake":
        return f"{rid} woke from hibernation"
    if kind == "exhausted":
        return f"{rid} is exhausted — integrity bleeding"
    if kind == "fall_damage":
        return f"{rid} fell {event.get('blocks', '?')} blocks"
    if kind == "death":
        return f"{rid} DIED (age {event.get('age_ticks', '?')} ticks)"
    if kind in ("dig", "place"):
        block = event.get("block")
        name = Block(int(block)).name if isinstance(block, int) else "?"
        return f"{rid} {kind} {name}"
    return f"{rid} {kind}"

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
        rotate_ticks: int = 0,
    ) -> None:
        self.tick_rate = tick_rate
        self.save_path = save_path
        # Rotation only makes sense when recording to files: each .rrd is a
        # self-contained scrubbable slice of the run.
        self.rotate_ticks = rotate_ticks if save_path is not None else 0
        self._rotation_index = 0
        self._sun_radius = max(world.cfg.size[0], world.cfg.size[1]) * 0.6
        self._styled: set[str] = set()
        self._dreamers: tuple[str, ...] = ()
        self._robot_paths: set[str] = set()
        rr.init(APP_ID, spawn=spawn)
        if save_path is not None:
            rr.save(str(self._rotation_path(world.tick)))
        self._log_scene_base(world)

    def _rotation_path(self, tick: int) -> Path:
        assert self.save_path is not None
        return self.save_path.with_name(f"{self.save_path.stem}_{tick:012d}.rrd")

    def _log_scene_base(self, world: World) -> None:
        """Everything a fresh recording needs to stand alone."""
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._styled.clear()
        self._style_world_series()
        self._style_robot_series(world)
        self._send_blueprint(world)
        self._set_time(world)
        self._log_all_chunks(world)

    def _style_world_series(self) -> None:
        for path, name, color in (
            ("charts/light_level", "daylight", (255, 220, 120)),
            ("charts/population", "population", (235, 235, 235)),
            ("charts/ripe_bushes", "ripe bushes", (196, 74, 60)),
            ("charts/toxic_bushes", "toxic bushes", (148, 70, 168)),
        ):
            rr.log(path, rr.SeriesLines(colors=[color], names=[name], widths=[1.5]), static=True)

    def _style_robot_series(self, world: World) -> None:
        """Name and color each robot's chart lines once, in its identity color."""
        for robot in world.robots.values():
            if robot.id in self._styled:
                continue
            self._styled.add(robot.id)
            color = robot_color(robot.id)
            for metric in ("energy", "integrity", "fatigue"):
                rr.log(
                    f"charts/{metric}/{robot.id}",
                    rr.SeriesLines(colors=[color], names=[robot.id], widths=[1.5]),
                    static=True,
                )

    def _send_blueprint(self, world: World) -> None:
        dreamers = tuple(sorted(r.id for r in world.robots.values() if r.brain_name == "dreamer"))
        self._dreamers = dreamers
        rr.send_blueprint(build_blueprint(list(dreamers)))

    def _maybe_rotate(self, world: World) -> None:
        if not self.rotate_ticks:
            return
        index = world.tick // self.rotate_ticks
        if index > self._rotation_index:
            self._rotation_index = index
            rr.save(str(self._rotation_path(world.tick)))
            self._log_scene_base(world)

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
        # One entity per robot (not one batch): individually selectable, and
        # double-click in any 3D view makes the camera track that body.
        for r in world.robots.values():
            center = [r.pos[0], r.pos[1], r.pos[2] + r.body.height / 2]
            color = (robot_color(r.id) * (0.4 if r.dormant else 1.0)).astype(np.uint8)
            rr.log(
                f"world/robots/{r.id}",
                rr.Boxes3D(
                    centers=[center],
                    half_sizes=[[r.body.width / 2, r.body.width / 2, r.body.height / 2]],
                    rotation_axis_angles=[rr.RotationAxisAngle(axis=(0, 0, 1), radians=r.yaw)],
                    colors=[color],
                    labels=[r.id],
                    fill_mode="solid",
                ),
            )
            rr.log(
                f"world/robots/{r.id}/heading",
                rr.Arrows3D(
                    origins=[center],
                    vectors=[[math.cos(r.yaw) * 1.2, math.sin(r.yaw) * 1.2, 0.0]],
                    colors=[color],
                ),
            )
        # The dead leave the stage: clear entities for departed robots.
        alive = set(world.robots)
        for gone in self._robot_paths - alive:
            rr.log(f"world/robots/{gone}", rr.Clear(recursive=True))
        self._robot_paths = alive

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

    def _log_sounds(self, world: World) -> None:
        """Transient cries as pulses: red for death, amber for hurt."""
        sounds = world.active_sounds()
        if not sounds:
            rr.log("world/sounds", rr.Clear(recursive=False))
            return
        points = []
        colors = []
        for x, y, s0, s1, *_ in sounds:
            z = world.grid.column_height(int(x), int(y)) + 1.5
            points.append([x, y, z])
            colors.append([235, 40, 40] if s0 <= -0.9 and s1 <= -0.9 else [245, 200, 70])
        rr.log("world/sounds", rr.Points3D(points, radii=0.5, colors=colors))

    def log_events(self, events: list[dict[str, object]]) -> None:
        """World events as a human-readable feed, globally and per robot."""
        for event in events:
            tick = int(str(event["tick"]))
            rr.set_time("tick", sequence=tick)
            rr.set_time("sim_time", duration=tick / self.tick_rate)
            level = _EVENT_LEVELS.get(str(event["kind"]), rr.TextLogLevel.INFO)
            rid = str(event.get("robot", "world"))
            rr.log(f"events/{rid}", rr.TextLog(_event_text(event), level=level))
            if event["kind"] == "feed" and "to" in event:
                rr.log(
                    f"events/{event['to']}",
                    rr.TextLog(f"{event['to']} was fed by {rid}", level=rr.TextLogLevel.INFO),
                )

    def log_frame(
        self,
        world: World,
        obs: dict[str, Observation] | None = None,
        introspection: dict[str, dict[str, float]] | None = None,
        heatmap: np.ndarray | None = None,
    ) -> None:
        self._maybe_rotate(world)
        self._set_time(world)
        # New bodies (respawns) get chart styles; a changed set of learning
        # minds gets a fresh layout with the right dashboard tabs.
        self._style_robot_series(world)
        dreamers = tuple(sorted(r.id for r in world.robots.values() if r.brain_name == "dreamer"))
        if dreamers != self._dreamers:
            self._send_blueprint(world)
        if heatmap is not None:
            rr.log("charts/visit_heatmap", rr.Image(heatmap))
        for cx, cy in world.grid.consume_dirty_chunks():
            self._log_chunk(world, cx, cy)
        self._log_sun(world)
        self._log_robots(world)
        self._log_sounds(world)
        if obs:
            self._log_rays(world, obs)
        rr.log("charts/light_level", rr.Scalars([world.light_level]))
        rr.log("charts/population", rr.Scalars([float(len(world.robots))]))
        rr.log(
            "charts/ripe_bushes",
            rr.Scalars([float((world.grid.blocks == Block.BUSH_RIPE).sum())]),
        )
        rr.log(
            "charts/toxic_bushes",
            rr.Scalars([float((world.grid.blocks == Block.BUSH_TOXIC).sum())]),
        )
        for robot in world.robots.values():
            rr.log(f"charts/energy/{robot.id}", rr.Scalars([robot.energy]))
            rr.log(f"charts/integrity/{robot.id}", rr.Scalars([robot.integrity]))
            rr.log(f"charts/fatigue/{robot.id}", rr.Scalars([robot.fatigue]))
        if introspection:
            for robot_id, metrics in introspection.items():
                for name, value in metrics.items():
                    rr.log(f"charts/brains/{robot_id}/{name}", rr.Scalars([value]))
