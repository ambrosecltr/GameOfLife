"""Robot senses: vectorized voxel raycasting and observation assembly.

Raycasting is Amanatides–Woo DDA stepped simultaneously for every ray of every
robot in one set of numpy ops — the sim's hottest path stays O(range), not
O(robots x rays x range) Python loops.

Vision is color (obs v3): each ray returns depth plus the shaded RGB of what
it struck — block palette color x face shade x per-voxel grain x daylight —
and a small hit-kind one-hot (block / robot / dormant / nothing). Block
identity is carried only by appearance; misses see the sky. Gaze offsets aim
the whole fan without turning the body.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from gol_world.blocks import COLOR, FACE_SHADE, Block, light_factor, sky_color, tint_factors
from gol_world.entities import DORMANT_DIM, Robot, robot_color
from gol_world.interface import (
    PROPRIO_DIM,
    RAY_DIM,
    RAY_KIND_BLOCK,
    RAY_KIND_DORMANT,
    RAY_KIND_NOTHING,
    RAY_KIND_ROBOT,
    SOUND_DIM,
    Observation,
)

F32 = np.float32
MISS = -1  # internal marker for "no block hit"


def ray_directions(
    yaw: float, pitches_deg: tuple[float, ...], n: int, fov_deg: float
) -> npt.NDArray[np.float64]:
    """Unit direction vectors for a fan of rays: (rows * n, 3), row-major."""
    half = math.radians(fov_deg) / 2
    azimuths = yaw + np.linspace(-half, half, n)
    dirs = []
    for pitch_deg in pitches_deg:
        pitch = math.radians(pitch_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        dirs.append(
            np.stack([np.cos(azimuths) * cp, np.sin(azimuths) * cp, np.full(n, sp)], axis=1)
        )
    return np.concatenate(dirs)


def robot_ray_dirs(robot: Robot) -> npt.NDArray[np.float64]:
    """The robot's current fan: body frame plus its gaze offset (eyes/head)."""
    body = robot.body
    yaw = robot.yaw + math.radians(body.gaze_yaw_max_deg) * float(robot.gaze[1])
    pitch_off = body.gaze_pitch_max_deg * float(robot.gaze[0])
    pitches = tuple(p + pitch_off for p in body.ray_pitches_deg)
    return ray_directions(yaw, pitches, body.rays_per_row, body.fov_deg)


def cast_rays(
    blocks: npt.NDArray[np.uint8],
    origins: npt.NDArray[np.float64],
    dirs: npt.NDArray[np.float64],
    max_range: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """March all rays through the grid at once.

    Returns (depth, hit, hit_cell, normal_axis, normal_sign) where hit is a
    block id or MISS (-1) beyond range; hit_cell is the struck voxel (only
    meaningful for block hits; may be out of bounds for border hits) and the
    normal is the face the ray entered through. Rays leaving the world
    sideways hit ROCK (the unbreakable border); leaving vertically is a miss
    (sky / the void below bedrock is empty).
    """
    n = len(origins)
    sx, sy, sz = blocks.shape
    cell = np.floor(origins).astype(np.int64)
    step = np.sign(dirs).astype(np.int64)
    safe = np.where(dirs == 0, 1e-12, dirs)
    t_delta = np.abs(1.0 / safe)
    next_boundary = cell + (step > 0)
    t_max = np.where(step != 0, (next_boundary - origins) / safe, np.inf)

    depth = np.full(n, max_range, dtype=np.float64)
    hit = np.full(n, MISS, dtype=np.int64)
    hit_cell = np.zeros((n, 3), dtype=np.int64)
    normal_axis = np.zeros(n, dtype=np.int64)
    normal_sign = np.ones(n, dtype=np.int64)

    # Active-set compaction: each iteration works only on rays still flying,
    # so per-step cost shrinks as rays land (most terminate within a few
    # steps — ground below, walls, nearby terrain).
    idx = np.arange(n)
    max_steps = int(max_range * 3) + 3
    for _ in range(max_steps):
        if idx.size == 0:
            break
        axis = np.argmin(t_max[idx], axis=1)
        t = t_max[idx, axis]
        # Past max range: those rays end as misses at full depth.
        live = t <= max_range
        idx, axis, t = idx[live], axis[live], t[live]
        if idx.size == 0:
            break

        cell[idx, axis] += step[idx, axis]
        cx, cy, cz = cell[idx, 0], cell[idx, 1], cell[idx, 2]

        oob_side = (cx < 0) | (cx >= sx) | (cy < 0) | (cy >= sy)
        if oob_side.any():
            g = idx[oob_side]
            hit[g] = int(Block.ROCK)
            depth[g] = t[oob_side]
            hit_cell[g] = cell[g]
            normal_axis[g] = axis[oob_side]
            normal_sign[g] = -step[g, axis[oob_side]]

        oob_vert = ~oob_side & ((cz < 0) | (cz >= sz))  # miss: stays MISS at max_range

        struck = np.zeros(idx.size, dtype=np.bool_)
        inb = ~oob_side & ~oob_vert
        if inb.any():
            solid = np.zeros(idx.size, dtype=np.bool_)
            solid[inb] = blocks[cx[inb], cy[inb], cz[inb]] != Block.AIR
            struck = inb & solid
            if struck.any():
                g = idx[struck]
                hit[g] = blocks[cx[struck], cy[struck], cz[struck]]
                depth[g] = t[struck]
                hit_cell[g] = cell[g]
                normal_axis[g] = axis[struck]
                normal_sign[g] = -step[g, axis[struck]]

        flying = ~oob_side & ~oob_vert & ~struck
        idx, axis = idx[flying], axis[flying]
        t_max[idx, axis] += t_delta[idx, axis]

    return depth, hit, hit_cell, normal_axis, normal_sign


def _block_appearance(
    hit: npt.NDArray[np.int64],
    hit_cell: npt.NDArray[np.int64],
    normal_axis: npt.NDArray[np.int64],
    normal_sign: npt.NDArray[np.int64],
    light_level: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Shaded, tinted, daylight-scaled RGB (n, 3) + hit kind (n,) for all rays."""
    n = len(hit)
    rgb = np.empty((n, 3), dtype=F32)
    kind = np.full(n, RAY_KIND_BLOCK, dtype=np.int64)

    miss = hit == MISS
    kind[miss] = RAY_KIND_NOTHING
    rgb[miss] = sky_color(light_level)

    struck = ~miss
    if struck.any():
        base = COLOR[hit[struck]].astype(F32) / 255.0
        shade = FACE_SHADE[normal_axis[struck], (normal_sign[struck] > 0).astype(np.int64)]
        # Border hits carry out-of-bounds cells; the grain hash is total, so
        # clipping isn't needed for correctness — the wall just has grain too.
        tint = tint_factors(hit_cell[struck])
        factor = shade * tint * light_factor(light_level)
        rgb[struck] = np.clip(base * factor[:, None], 0.0, 1.0)
    return rgb, kind


def _overlay_robots(
    viewer: Robot,
    others: list[Robot],
    origins: npt.NDArray[np.float64],
    dirs: npt.NDArray[np.float64],
    depth: npt.NDArray[np.float64],
    rgb: npt.NDArray[np.float32],
    kind: npt.NDArray[np.int64],
    light_level: float,
) -> None:
    """Replace block hits with robot hits where a body is closer along the ray."""
    lit = light_factor(light_level)
    for other in others:
        lo, hi = other.aabb
        d = other.pos[:2] - viewer.pos[:2]
        if float(np.hypot(d[0], d[1])) > viewer.body.ray_range + 1:
            continue
        safe = np.where(dirs == 0, 1e-12, dirs)
        t1 = (lo[None, :] - origins) / safe
        t2 = (hi[None, :] - origins) / safe
        tmin = np.minimum(t1, t2).max(axis=1)
        tmax = np.maximum(t1, t2).min(axis=1)
        hits = (tmax >= tmin) & (tmax >= 0) & (tmin < depth) & (tmin > 0)
        if not hits.any():
            continue
        color = robot_color(other.id).astype(F32) / 255.0
        if other.dormant:
            color = color * DORMANT_DIM
        depth[hits] = tmin[hits]
        kind[hits] = RAY_KIND_DORMANT if other.dormant else RAY_KIND_ROBOT
        rgb[hits] = np.clip(color * lit, 0.0, 1.0)


def _proprio(robot: Robot, light_level: float) -> npt.NDArray[np.float32]:
    body = robot.body
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    vx, vy, vz = robot.vel
    out = np.empty(PROPRIO_DIM, dtype=F32)
    out[0] = (c * vx + s * vy) / body.max_speed
    out[1] = (-s * vx + c * vy) / body.max_speed
    out[2] = vz / body.max_speed
    out[3] = s
    out[4] = c
    out[5] = robot.energy / 100.0
    out[6] = robot.integrity / 100.0
    out[7] = 0.0 if robot.held is None else 1.0
    out[8] = 0.0 if robot.held is None else robot.held / 16.0
    out[9:13] = robot.touch.astype(F32)
    out[13] = light_level
    out[14] = robot.fatigue
    out[15] = robot.gaze[0]  # head pitch, fraction of gaze_pitch_max
    out[16] = robot.gaze[1]  # head yaw, fraction of gaze_yaw_max
    return out


def _sound(
    robot: Robot,
    others: list[Robot],
    world_sounds: Sequence[tuple[float, float, float, float]] = (),
) -> npt.NDArray[np.float32]:
    out = np.zeros(SOUND_DIM, dtype=F32)
    radius = robot.body.hear_radius
    total_w = 0.0
    mix = np.zeros(2)
    loudest = 0.0
    loudest_bearing: float | None = None
    # Robot signals and transient world sounds (cries) mix identically.
    sources = [(other.pos[:2], other.signal) for other in others]
    sources += [(np.array([x, y]), np.array([s0, s1])) for x, y, s0, s1 in world_sounds]
    for pos, signal in sources:
        d = pos - robot.pos[:2]
        dist = float(np.hypot(d[0], d[1]))
        if dist > radius:
            continue
        w = 1.0 - dist / radius
        mix += w * signal
        total_w += w
        volume = w * float(np.abs(signal).max())
        if volume > loudest:
            loudest = volume
            loudest_bearing = math.atan2(d[1], d[0]) - robot.yaw
    if total_w > 0:
        out[0:2] = (mix / total_w).astype(F32)
    if loudest_bearing is not None:
        out[2] = math.sin(loudest_bearing)
        out[3] = math.cos(loudest_bearing)
    return out


def observe(
    blocks: npt.NDArray[np.uint8],
    robots: list[Robot],
    light_level: float,
    world_sounds: Sequence[tuple[float, float, float, float]] = (),
    toxic_mimic: bool = False,
) -> dict[str, Observation]:
    """Build observations for every awake robot (one batched raycast)."""
    awake = [r for r in robots if not r.dormant]
    if not awake:
        return {}

    all_origins = []
    all_dirs = []
    counts = []
    for robot in awake:
        dirs = robot_ray_dirs(robot)
        all_origins.append(np.repeat(robot.eye[None, :], len(dirs), axis=0))
        all_dirs.append(dirs)
        counts.append(len(dirs))
    origins = np.concatenate(all_origins)
    dirs = np.concatenate(all_dirs)
    max_range = max(r.body.ray_range for r in awake)
    depth, hit, hit_cell, normal_axis, normal_sign = cast_rays(blocks, origins, dirs, max_range)
    if toxic_mimic:
        # Ablation: perfect mimicry — toxic bushes wear ripe-red, learnable
        # only through consequence and place memory.
        hit[hit == Block.BUSH_TOXIC] = Block.BUSH_RIPE
    all_rgb, all_kind = _block_appearance(hit, hit_cell, normal_axis, normal_sign, light_level)

    obs: dict[str, Observation] = {}
    offset = 0
    for robot, count in zip(awake, counts, strict=True):
        sl = slice(offset, offset + count)
        offset += count
        r_depth = depth[sl].copy()
        r_rgb = all_rgb[sl].copy()
        r_kind = all_kind[sl].copy()
        others = [o for o in robots if o.id != robot.id]
        _overlay_robots(
            robot, others, origins[sl], dirs[sl], r_depth, r_rgb, r_kind, light_level
        )

        rays = np.zeros((count, RAY_DIM), dtype=F32)
        rays[:, 0] = (r_depth / robot.body.ray_range).astype(F32)
        rays[:, 1:4] = r_rgb
        rays[np.arange(count), 4 + r_kind] = 1.0

        obs[robot.id] = Observation(
            rays=rays,
            proprio=_proprio(robot, light_level),
            sound=_sound(robot, others, world_sounds),
            events=robot.drain_events().astype(F32),
        )
    return obs
