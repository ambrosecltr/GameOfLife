import numpy as np
from gol_world.blocks import Block
from gol_world.entities import TOUCH_FRONT, TOUCH_GROUND, Robot
from gol_world.grid import VoxelGrid
from gol_world.physics import step_robot

DT = 0.05


def flat_grid(size: tuple[int, int, int] = (32, 32, 16), floor_z: int = 3) -> VoxelGrid:
    grid = VoxelGrid.empty(size)
    grid.blocks[:, :, : floor_z + 1] = Block.ROCK
    return grid


def make_robot(x: float, y: float, z: float, yaw: float = 0.0) -> Robot:
    r = Robot(id="t", pos=np.array([x, y, z]), yaw=yaw, brain_name="test")
    r.fall_peak_z = z
    return r


def settle(grid: VoxelGrid, robot: Robot, ticks: int = 10) -> None:
    for _ in range(ticks):
        step_robot(grid, robot, DT)


def test_rests_on_ground() -> None:
    grid = flat_grid()
    robot = make_robot(16.5, 16.5, 8.0)
    settle(grid, robot, 40)
    np.testing.assert_allclose(robot.pos[2], 4.0, atol=0.01)
    assert robot.touch[TOUCH_GROUND]


def test_drives_forward() -> None:
    grid = flat_grid()
    robot = make_robot(10.5, 16.5, 4.0)
    settle(grid, robot)
    robot.drive[:] = [1.0, 0.0]  # yaw 0 -> +x
    for _ in range(40):  # 2 seconds at full speed (4 b/s)
        step_robot(grid, robot, DT)
    assert robot.pos[0] > 16.0
    assert abs(robot.pos[1] - 16.5) < 0.01


def test_wall_stops_and_touches_front() -> None:
    grid = flat_grid()
    grid.blocks[20, :, 4:7] = Block.ROCK  # 3-high wall: unclimbable
    robot = make_robot(18.5, 16.5, 4.0)
    settle(grid, robot)
    robot.drive[:] = [1.0, 0.0]
    for _ in range(40):
        step_robot(grid, robot, DT)
    assert robot.pos[0] < 20.0 - 0.4 + 0.01  # flush against the wall
    assert robot.touch[TOUCH_FRONT]


def test_climbs_single_step() -> None:
    grid = flat_grid()
    grid.blocks[20:, :, 4] = Block.ROCK  # one-block ledge
    robot = make_robot(18.5, 16.5, 4.0)
    settle(grid, robot)
    robot.drive[:] = [1.0, 0.0]
    climbed = 0.0
    for _ in range(60):
        climbed += step_robot(grid, robot, DT)["climbed"]
    assert robot.pos[0] > 20.5
    assert abs(robot.pos[2] - 5.0) < 0.01
    assert climbed >= 1.0


def test_two_block_wall_blocks() -> None:
    grid = flat_grid()
    grid.blocks[20:, :, 4:6] = Block.ROCK  # two-high ledge
    robot = make_robot(18.5, 16.5, 4.0)
    settle(grid, robot)
    robot.drive[:] = [1.0, 0.0]
    for _ in range(60):
        step_robot(grid, robot, DT)
    assert robot.pos[0] < 20.0
    assert abs(robot.pos[2] - 4.0) < 0.01


def test_fall_damage_beyond_three_blocks() -> None:
    grid = flat_grid()
    robot = make_robot(16.5, 16.5, 9.5)  # 5.5-block drop
    damage = 0.0
    for _ in range(40):
        damage += step_robot(grid, robot, DT)["fall_damage"]
    assert robot.touch[TOUCH_GROUND]
    assert damage > 1.5  # ~5.5 - 3 blocks over threshold


def test_short_fall_is_safe() -> None:
    grid = flat_grid()
    robot = make_robot(16.5, 16.5, 6.0)  # 2-block drop
    damage = 0.0
    for _ in range(40):
        damage += step_robot(grid, robot, DT)["fall_damage"]
    assert damage == 0.0


def test_water_slows() -> None:
    dry = flat_grid()
    wet = flat_grid()
    wet.blocks[:, :, 4:6] = Block.WATER

    fast = make_robot(8.5, 16.5, 4.0)
    slow = make_robot(8.5, 16.5, 4.0)
    settle(dry, fast)
    settle(wet, slow)
    fast.drive[:] = [1.0, 0.0]
    slow.drive[:] = [1.0, 0.0]
    for _ in range(40):
        step_robot(dry, fast, DT)
        step_robot(wet, slow, DT)
    assert slow.in_water
    dist_fast = fast.pos[0] - 8.5
    dist_slow = slow.pos[0] - 8.5
    assert dist_slow < dist_fast * 0.7


def test_world_border_is_wall() -> None:
    grid = flat_grid()
    robot = make_robot(1.5, 16.5, 4.0)
    settle(grid, robot)
    robot.yaw = np.pi  # face -x
    robot.drive[:] = [1.0, 0.0]
    for _ in range(60):
        step_robot(grid, robot, DT)
    assert robot.pos[0] > 0.39  # half-width flush
