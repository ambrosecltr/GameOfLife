"""Robot senses: vectorized voxel raycasting and observation assembly.

Raycasting is Amanatides–Woo DDA stepped simultaneously for every ray of every
robot in one set of numpy ops — the sim's hottest path stays O(range), not
O(robots x rays x range) Python loops.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from gol_world.blocks import Block
from gol_world.entities import Robot
from gol_world.interface import (
    NUM_RAY_CLASSES,
    PROPRIO_DIM,
    RAY_CLASS_DORMANT,
    RAY_CLASS_NOTHING,
    RAY_CLASS_ROBOT,
    SOUND_DIM,
    Observation,
)

F32 = np.float32
MISS = -1  # internal marker before one-hot encoding


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


def cast_rays(
    blocks: npt.NDArray[np.uint8],
    origins: npt.NDArray[np.float64],
    dirs: npt.NDArray[np.float64],
    max_range: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """March all rays through the grid at once.

    Returns (depth, hit) where hit is a block id, or MISS (-1) beyond range.
    Rays leaving the world sideways hit ROCK (the unbreakable border); leaving
    vertically is a miss (sky / the void below bedrock is empty).
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
    active = np.ones(n, dtype=np.bool_)

    max_steps = int(max_range * 3) + 3
    for _ in range(max_steps):
        if not active.any():
            break
        axis = np.argmin(t_max, axis=1)
        idx = np.arange(n)
        t = t_max[idx, axis]
        over = active & (t > max_range)
        active &= ~over

        move = active
        cell[move, axis[move]] += step[move, axis[move]]
        cx, cy, cz = cell[:, 0], cell[:, 1], cell[:, 2]

        oob_side = move & ((cx < 0) | (cx >= sx) | (cy < 0) | (cy >= sy))
        hit[oob_side] = int(Block.ROCK)
        depth[oob_side] = t[oob_side]
        active &= ~oob_side

        oob_vert = move & ~oob_side & ((cz < 0) | (cz >= sz))
        active &= ~oob_vert  # miss: stays MISS at max_range

        inb = active & move & ~oob_side & ~oob_vert
        if inb.any():
            solid = np.zeros(n, dtype=np.bool_)
            solid[inb] = blocks[cx[inb], cy[inb], cz[inb]] != Block.AIR
            struck = inb & solid
            hit[struck] = blocks[cx[struck], cy[struck], cz[struck]]
            depth[struck] = t[struck]
            active &= ~struck

        t_max[idx, axis] += t_delta[idx, axis]

    return depth, hit


def _overlay_robots(
    viewer: Robot,
    others: list[Robot],
    origins: npt.NDArray[np.float64],
    dirs: npt.NDArray[np.float64],
    depth: npt.NDArray[np.float64],
    hit_class: npt.NDArray[np.int64],
) -> None:
    """Replace block hits with robot hits where a body is closer along the ray."""
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
        cls = RAY_CLASS_DORMANT if other.dormant else RAY_CLASS_ROBOT
        depth[hits] = tmin[hits]
        hit_class[hits] = cls


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
    return out


def _sound(robot: Robot, others: list[Robot]) -> npt.NDArray[np.float32]:
    out = np.zeros(SOUND_DIM, dtype=F32)
    radius = robot.body.hear_radius
    total_w = 0.0
    mix = np.zeros(2)
    loudest = 0.0
    loudest_bearing: float | None = None
    for other in others:
        d = other.pos[:2] - robot.pos[:2]
        dist = float(np.hypot(d[0], d[1]))
        if dist > radius:
            continue
        w = 1.0 - dist / radius
        mix += w * other.signal
        total_w += w
        volume = w * float(np.abs(other.signal).max())
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
) -> dict[str, Observation]:
    """Build observations for every awake robot (one batched raycast)."""
    awake = [r for r in robots if not r.dormant]
    if not awake:
        return {}

    all_origins = []
    all_dirs = []
    counts = []
    for robot in awake:
        dirs = ray_directions(
            robot.yaw, robot.body.ray_pitches_deg, robot.body.rays_per_row, robot.body.fov_deg
        )
        all_origins.append(np.repeat(robot.eye[None, :], len(dirs), axis=0))
        all_dirs.append(dirs)
        counts.append(len(dirs))
    origins = np.concatenate(all_origins)
    dirs = np.concatenate(all_dirs)
    max_range = max(r.body.ray_range for r in awake)
    depth, hit = cast_rays(blocks, origins, dirs, max_range)

    obs: dict[str, Observation] = {}
    offset = 0
    for robot, count in zip(awake, counts, strict=True):
        sl = slice(offset, offset + count)
        offset += count
        r_depth = depth[sl].copy()
        r_hit = hit[sl].copy()
        others = [o for o in robots if o.id != robot.id]
        _overlay_robots(robot, others, origins[sl], dirs[sl], r_depth, r_hit)

        rays = np.zeros((count, 1 + NUM_RAY_CLASSES), dtype=F32)
        rays[:, 0] = (r_depth / robot.body.ray_range).astype(F32)
        classes = np.where(r_hit == MISS, RAY_CLASS_NOTHING, r_hit)
        rays[np.arange(count), 1 + classes] = 1.0

        obs[robot.id] = Observation(
            rays=rays,
            proprio=_proprio(robot, light_level),
            sound=_sound(robot, others),
            events=robot.drain_events().astype(F32),
        )
    return obs
