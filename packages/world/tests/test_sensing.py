import numpy as np
from gol_world.blocks import Block
from gol_world.entities import Robot
from gol_world.grid import VoxelGrid
from gol_world.interface import (
    RAY_CLASS_NOTHING,
    RAY_CLASS_ROBOT,
)
from gol_world.sensing import cast_rays, observe, ray_directions


def scene() -> VoxelGrid:
    grid = VoxelGrid.empty((16, 16, 16))
    grid.blocks[:, :, 0] = Block.ROCK  # floor at z=0
    return grid


def test_ray_hits_wall_at_distance() -> None:
    grid = scene()
    grid.blocks[10, :, :] = Block.ORE  # wall plane at x=10
    origins = np.array([[2.5, 8.5, 8.5]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    depth, hit = cast_rays(grid.blocks, origins, dirs, max_range=24.0)
    assert hit[0] == Block.ORE
    np.testing.assert_allclose(depth[0], 7.5, atol=1e-6)


def test_ray_miss_within_range() -> None:
    grid = scene()
    origins = np.array([[2.5, 8.5, 8.5]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    depth, hit = cast_rays(grid.blocks, origins, dirs, max_range=8.0)
    assert hit[0] == -1
    assert depth[0] == 8.0


def test_ray_up_is_sky_miss_down_hits_floor() -> None:
    grid = scene()
    origins = np.array([[8.5, 8.5, 8.5], [8.5, 8.5, 8.5]])
    dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    depth, hit = cast_rays(grid.blocks, origins, dirs, max_range=24.0)
    assert hit[0] == -1
    assert hit[1] == Block.ROCK
    np.testing.assert_allclose(depth[1], 7.5, atol=1e-6)


def test_ray_leaving_sideways_hits_border_rock() -> None:
    grid = scene()
    origins = np.array([[8.5, 8.5, 8.5]])
    dirs = np.array([[-1.0, 0.0, 0.0]])
    depth, hit = cast_rays(grid.blocks, origins, dirs, max_range=24.0)
    assert hit[0] == Block.ROCK
    np.testing.assert_allclose(depth[0], 8.5, atol=1e-6)


def test_diagonal_ray_through_gap() -> None:
    grid = scene()
    grid.blocks[8, 8, 8] = Block.ROCK
    origins = np.array([[6.5, 6.5, 8.5]])
    dirs = np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2)
    depth, hit = cast_rays(grid.blocks, origins, dirs, max_range=24.0)
    assert hit[0] == Block.ROCK  # hits the diagonal block's corner cell


def test_ray_directions_fan() -> None:
    dirs = ray_directions(0.0, (0.0,), 16, 144.0)
    assert dirs.shape == (16, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    # Fan is symmetric around yaw.
    assert abs(dirs[0][1] + dirs[-1][1]) < 1e-9


def test_observe_sees_other_robot() -> None:
    grid = scene()
    viewer = Robot(id="a", pos=np.array([4.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    target = Robot(id="b", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    obs = observe(grid.blocks, [viewer, target], light_level=1.0)
    classes = obs["a"]["rays"][:, 1:].argmax(axis=1)
    assert (classes == RAY_CLASS_ROBOT).any(), "viewer should see the other robot"
    # And the robot hit is closer than background misses.
    robot_rays = classes == RAY_CLASS_ROBOT
    assert obs["a"]["rays"][robot_rays, 0].min() < 0.5


def test_observe_shapes_and_events_drain() -> None:
    grid = scene()
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.3, brain_name="t")
    robot.events[0] = 1.0
    obs = observe(grid.blocks, [robot], light_level=0.5)["a"]
    assert obs["rays"].shape == (robot.body.num_rays, 16)
    assert obs["proprio"].shape == (14,)
    assert obs["sound"].shape == (4,)
    assert obs["events"].shape == (4,)
    assert obs["events"][0] == 1.0
    assert robot.events[0] == 0.0, "events drain on observe"
    # Each ray is one-hot.
    np.testing.assert_allclose(obs["rays"][:, 1:].sum(axis=1), 1.0)


def test_dormant_robot_seen_not_seeing() -> None:
    grid = scene()
    viewer = Robot(id="a", pos=np.array([4.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    sleeper = Robot(id="b", pos=np.array([7.5, 8.5, 1.0]), yaw=0.0, brain_name="t", dormant=True)
    obs = observe(grid.blocks, [viewer, sleeper], light_level=1.0)
    assert "b" not in obs
    classes = obs["a"]["rays"][:, 1:].argmax(axis=1)
    assert (classes == RAY_CLASS_ROBOT + 1).any()  # RAY_CLASS_DORMANT


def test_sound_carries_signal_and_bearing() -> None:
    grid = scene()
    hearer = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    shouter = Robot(id="b", pos=np.array([8.5, 12.5, 1.0]), yaw=0.0, brain_name="t")
    shouter.signal[:] = [1.0, -0.5]
    obs = observe(grid.blocks, [hearer, shouter], light_level=1.0)["a"]
    assert obs["sound"][0] > 0.0 and obs["sound"][1] < 0.0
    # Bearing: shouter is at +y, hearer faces +x -> relative bearing +90 deg.
    np.testing.assert_allclose(obs["sound"][2], 1.0, atol=1e-6)
    np.testing.assert_allclose(obs["sound"][3], 0.0, atol=1e-6)


def test_silence_when_alone() -> None:
    grid = scene()
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    obs = observe(grid.blocks, [robot], light_level=1.0)["a"]
    assert (obs["sound"] == 0).all()
    classes = obs["rays"][:, 1:].argmax(axis=1)
    assert (classes != RAY_CLASS_NOTHING).sum() > 0  # floor/border are visible
