"""Minecraft-style body physics: AABB-vs-voxel sweeps, gravity, step-climbing.

No rigid-body engine. Bodies are axis-aligned boxes that slide against the
voxel grid, auto-climb 1-block steps, take fall damage past 3 blocks, and slow
down in water. This is deliberately the physics of Minecraft, not MuJoCo: rich
enough to make space matter, cheap enough to never be the bottleneck.
"""

from __future__ import annotations

import math

import numpy as np

from gol_world.blocks import Block
from gol_world.entities import (
    EV_BUMPED_ROBOT,
    EV_TOOK_DAMAGE,
    TOUCH_FRONT,
    TOUCH_GROUND,
    TOUCH_LEFT,
    TOUCH_RIGHT,
    Robot,
)
from gol_world.grid import VoxelGrid

GRAVITY = 25.0  # blocks/s^2
TERMINAL_VELOCITY = -30.0  # blocks/s
WATER_SINK_SPEED = -1.5  # blocks/s
FALL_DAMAGE_THRESHOLD = 3.0  # blocks
STEP_HEIGHT = 1.0  # auto-climbable ledge height
EPS = 1e-6


def _box_collides(grid: VoxelGrid, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Does the AABB overlap any solid block (world border counts as solid)?"""
    x0, y0, z0 = np.floor(lo + EPS).astype(np.int64)
    x1, y1, z1 = np.floor(hi - EPS).astype(np.int64)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                if grid.is_solid(x, y, z):
                    return True
    return False


def _sweep_axis(
    grid: VoxelGrid, lo: np.ndarray, hi: np.ndarray, axis: int, delta: float
) -> tuple[float, bool]:
    """Move the box along one axis, clamping at the first solid block.

    Returns (achieved delta, collided). A single tick can cross more than one
    cell boundary (e.g. fast falls), so on collision we retry flush against
    each crossed boundary, farthest first, and take the first free position.
    """
    if abs(delta) < EPS:
        return 0.0, False

    def collides_at(d: float) -> bool:
        new_lo, new_hi = lo.copy(), hi.copy()
        new_lo[axis] += d
        new_hi[axis] += d
        return _box_collides(grid, new_lo, new_hi)

    if not collides_at(delta):
        return delta, False

    if delta > 0:
        face = hi[axis]
        boundaries = range(math.floor(face + delta), math.floor(face), -1)
        trials = [b - face - EPS for b in boundaries]
    else:
        face = lo[axis]
        boundaries = range(math.ceil(face + delta), math.ceil(face))
        trials = [b - face + EPS for b in boundaries]
    for trial in trials:
        if (delta > 0 and trial < 0) or (delta < 0 and trial > 0):
            continue
        if not collides_at(trial):
            return trial, True
    return 0.0, True


def feet_block(grid: VoxelGrid, robot: Robot) -> int:
    x, y, z = int(robot.pos[0]), int(robot.pos[1]), int(robot.pos[2] + 0.1)
    if not grid.in_bounds(x, y, z):
        return int(Block.AIR)
    return grid.get_block(x, y, z)


def step_robot(
    grid: VoxelGrid,
    robot: Robot,
    dt: float,
    actuation: float = 1.0,
    water_speed_mult: float = 0.5,
) -> dict[str, float]:
    """Advance one robot one tick. Returns physics costs for the economy layer.

    Costs: climbed (blocks stepped up), moved (fraction of full drive), fall
    damage is applied directly to integrity here. `actuation` scales speed and
    turn rate (energy brownout); costs charge the commanded effort, not the
    achieved motion — a browned-out body pays full price for less movement.
    """
    costs = {"climbed": 0.0, "moved": 0.0, "turned": 0.0, "fall_damage": 0.0}
    if robot.dormant:
        # Dormant bodies still fall and rest on the ground; nothing else.
        robot.drive[:] = 0.0

    body = robot.body
    robot.in_water = feet_block(grid, robot) == Block.WATER

    # --- steering
    forward_cmd = float(np.clip(robot.drive[0], -1.0, 1.0))
    turn_cmd = float(np.clip(robot.drive[1], -1.0, 1.0))
    robot.yaw = (robot.yaw + turn_cmd * body.max_turn * actuation * dt) % (2 * math.pi)
    speed = forward_cmd * body.max_speed * actuation
    if robot.in_water:
        speed *= water_speed_mult
    vx = math.cos(robot.yaw) * speed
    vy = math.sin(robot.yaw) * speed
    costs["moved"] = abs(forward_cmd)
    costs["turned"] = abs(turn_cmd)

    # --- gravity / buoyancy
    if robot.in_water:
        robot.vel[2] = max(robot.vel[2] - GRAVITY * dt * 0.3, WATER_SINK_SPEED)
    else:
        robot.vel[2] = max(robot.vel[2] - GRAVITY * dt, TERMINAL_VELOCITY)

    # --- axis-by-axis sweep (x, y with step-climb; then z)
    lo, hi = robot.aabb
    grounded_before = bool(robot.touch[TOUCH_GROUND])
    blocked = np.zeros(2, dtype=np.bool_)
    for axis, want in ((0, vx * dt), (1, vy * dt)):
        got, hit = _sweep_axis(grid, lo, hi, axis, want)
        if hit and grounded_before and not robot.dormant:
            # Try climbing a 1-block step: lift, retry, settle.
            lifted_lo, lifted_hi = lo.copy(), hi.copy()
            lifted_lo[2] += STEP_HEIGHT
            lifted_hi[2] += STEP_HEIGHT
            if not _box_collides(grid, lifted_lo, lifted_hi):
                got2, hit2 = _sweep_axis(grid, lifted_lo, lifted_hi, axis, want)
                if abs(got2) > abs(got) + EPS:
                    lo, hi = lifted_lo, lifted_hi
                    got, hit = got2, hit2
                    costs["climbed"] += STEP_HEIGHT
        lo[axis] += got
        hi[axis] += got
        blocked[axis] = hit and abs(got) < abs(want) - EPS

    got_z, hit_z = _sweep_axis(grid, lo, hi, 2, robot.vel[2] * dt)
    lo[2] += got_z
    hi[2] += got_z
    grounded = hit_z and robot.vel[2] < 0
    if hit_z:
        robot.vel[2] = 0.0

    robot.pos[:] = [lo[0] + body.width / 2, lo[1] + body.width / 2, lo[2]]

    # --- touch flags in body frame
    robot.touch[:] = False
    robot.touch[TOUCH_GROUND] = grounded
    if blocked.any():
        # World-frame direction we pushed into, classified against yaw.
        push = math.atan2(vy if blocked[1] else 0.0, vx if blocked[0] else 0.0)
        rel = (push - robot.yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(rel) < math.pi / 3:
            robot.touch[TOUCH_FRONT] = True
        elif rel > 0:
            robot.touch[TOUCH_LEFT] = True
        else:
            robot.touch[TOUCH_RIGHT] = True

    # --- fall damage
    if grounded:
        drop = robot.fall_peak_z - robot.pos[2]
        if drop > FALL_DAMAGE_THRESHOLD and not robot.in_water:
            costs["fall_damage"] = drop - FALL_DAMAGE_THRESHOLD
            robot.events[EV_TOOK_DAMAGE] = 1.0
        robot.fall_peak_z = robot.pos[2]
    else:
        robot.fall_peak_z = max(robot.fall_peak_z, float(robot.pos[2]))

    return costs


def resolve_robot_overlaps(robots: list[Robot]) -> None:
    """Push overlapping robots apart horizontally (they are not ghosts)."""
    for i, a in enumerate(robots):
        for b in robots[i + 1 :]:
            min_dist = (a.body.width + b.body.width) / 2
            d = a.pos[:2] - b.pos[:2]
            dist = float(np.hypot(d[0], d[1]))
            if dist >= min_dist or abs(a.pos[2] - b.pos[2]) > a.body.height:
                continue
            a.events[EV_BUMPED_ROBOT] = 1.0
            b.events[EV_BUMPED_ROBOT] = 1.0
            if dist < EPS:
                d = np.array([1.0, 0.0])
                dist = 1.0
            push = (min_dist - dist) / 2
            shift = d / dist * push
            a.pos[:2] += shift
            b.pos[:2] -= shift
