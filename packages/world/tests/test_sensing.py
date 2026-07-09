import numpy as np
from gol_world.blocks import COLOR, Block, sky_color, tint_factors
from gol_world.entities import Robot
from gol_world.grid import VoxelGrid
from gol_world.interface import (
    PROPRIO_DIM,
    RAY_DIM,
    RAY_KIND_BLOCK,
    RAY_KIND_DORMANT,
    RAY_KIND_NOTHING,
    RAY_KIND_ROBOT,
)
from gol_world.sensing import cast_rays, observe, ray_directions, robot_ray_dirs


def scene() -> VoxelGrid:
    grid = VoxelGrid.empty((16, 16, 16))
    grid.blocks[:, :, 0] = Block.ROCK  # floor at z=0
    return grid


def ray_kinds(rays: np.ndarray) -> np.ndarray:
    return rays[:, 4:].argmax(axis=1)


def chroma(rgb: np.ndarray) -> np.ndarray:
    s = rgb.sum(axis=-1, keepdims=True)
    return rgb / np.where(s <= 1e-6, 1.0, s)


def test_ray_hits_wall_at_distance() -> None:
    grid = scene()
    grid.blocks[10, :, :] = Block.ORE  # wall plane at x=10
    origins = np.array([[2.5, 8.5, 8.5]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    depth, hit, cell, n_axis, n_sign = cast_rays(grid.blocks, origins, dirs, max_range=24.0)
    assert hit[0] == Block.ORE
    np.testing.assert_allclose(depth[0], 7.5, atol=1e-6)
    # Entered through the -x face of cell (10, 8, 8).
    np.testing.assert_array_equal(cell[0], [10, 8, 8])
    assert n_axis[0] == 0 and n_sign[0] == -1


def test_ray_miss_within_range() -> None:
    grid = scene()
    origins = np.array([[2.5, 8.5, 8.5]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    depth, hit, *_ = cast_rays(grid.blocks, origins, dirs, max_range=8.0)
    assert hit[0] == -1
    assert depth[0] == 8.0


def test_ray_up_is_sky_miss_down_hits_floor() -> None:
    grid = scene()
    origins = np.array([[8.5, 8.5, 8.5], [8.5, 8.5, 8.5]])
    dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    depth, hit, _, n_axis, n_sign = cast_rays(grid.blocks, origins, dirs, max_range=24.0)
    assert hit[0] == -1
    assert hit[1] == Block.ROCK
    np.testing.assert_allclose(depth[1], 7.5, atol=1e-6)
    # Floor is struck through its top (+z) face.
    assert n_axis[1] == 2 and n_sign[1] == 1


def test_ray_leaving_sideways_hits_border_rock() -> None:
    grid = scene()
    origins = np.array([[8.5, 8.5, 8.5]])
    dirs = np.array([[-1.0, 0.0, 0.0]])
    depth, hit, *_ = cast_rays(grid.blocks, origins, dirs, max_range=24.0)
    assert hit[0] == Block.ROCK
    np.testing.assert_allclose(depth[0], 8.5, atol=1e-6)


def test_diagonal_ray_through_gap() -> None:
    grid = scene()
    grid.blocks[8, 8, 8] = Block.ROCK
    origins = np.array([[6.5, 6.5, 8.5]])
    dirs = np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2)
    depth, hit, *_ = cast_rays(grid.blocks, origins, dirs, max_range=24.0)
    assert hit[0] == Block.ROCK  # hits the diagonal block's corner cell


def test_ray_directions_fan() -> None:
    dirs = ray_directions(0.0, (0.0,), 16, 144.0)
    assert dirs.shape == (16, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    # Fan is symmetric around yaw.
    assert abs(dirs[0][1] + dirs[-1][1]) < 1e-9


def test_default_fan_includes_upward_pitches() -> None:
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    dirs = robot_ray_dirs(robot)
    assert dirs.shape == (robot.body.num_rays, 3)
    assert dirs[:, 2].max() > 0.3, "the fan must look up as well as down"


def test_gaze_pitches_the_whole_fan() -> None:
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    level = robot_ray_dirs(robot)
    robot.gaze[:] = [1.0, 0.0]  # look full up
    up = robot_ray_dirs(robot)
    assert up[:, 2].min() > level[:, 2].min()
    assert up[:, 2].max() > level[:, 2].max()


def test_gaze_yaw_turns_eyes_not_body() -> None:
    grid = scene()
    grid.blocks[8, 12, 1] = Block.ORE  # a block to the robot's left (+y)
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    ore_chroma = chroma(COLOR[Block.ORE].astype(np.float32) / 255.0)

    def sees_ore(rays: np.ndarray) -> bool:
        blocks_hit = ray_kinds(rays) == RAY_KIND_BLOCK
        if not blocks_hit.any():
            return False
        c = chroma(rays[blocks_hit, 1:4])
        return bool((((c - ore_chroma) ** 2).sum(axis=1) < 1e-4).any())

    ahead = observe(grid.blocks, [robot], light_level=1.0)["a"]["rays"]
    robot.gaze[:] = [0.0, 1.0]  # eyes hard left (+90 deg at default gaze range)
    left = observe(grid.blocks, [robot], light_level=1.0)["a"]["rays"]
    assert not sees_ore(ahead), "ore at +y is outside the forward fan"
    assert sees_ore(left), "gazing left must bring the ore into view"


def test_bush_up_a_hill_is_visible() -> None:
    grid = scene()
    # A mound at x=12 with a bush on top, above the robot's eye level: level
    # rays hit the mound face, only an upward pitch row reaches the bush.
    grid.blocks[12, 7:10, 1:3] = Block.ROCK
    grid.blocks[12, 8, 3] = Block.BUSH_RIPE
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    rays = observe(grid.blocks, [robot], light_level=1.0)["a"]["rays"]
    blocks_hit = ray_kinds(rays) == RAY_KIND_BLOCK
    c = chroma(rays[blocks_hit, 1:4])
    ripe = chroma(COLOR[Block.BUSH_RIPE].astype(np.float32) / 255.0)
    assert (((c - ripe) ** 2).sum(axis=1) < 1e-4).any(), "upward rays must see the hilltop bush"


def test_observe_sees_other_robot() -> None:
    grid = scene()
    viewer = Robot(id="a", pos=np.array([4.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    target = Robot(id="b", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    obs = observe(grid.blocks, [viewer, target], light_level=1.0)
    kinds = ray_kinds(obs["a"]["rays"])
    assert (kinds == RAY_KIND_ROBOT).any(), "viewer should see the other robot"
    # And the robot hit is closer than background misses.
    robot_rays = kinds == RAY_KIND_ROBOT
    assert obs["a"]["rays"][robot_rays, 0].min() < 0.5


def test_observe_shapes_and_events_drain() -> None:
    grid = scene()
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.3, brain_name="t")
    robot.events[0] = 1.0
    obs = observe(grid.blocks, [robot], light_level=0.5)["a"]
    assert obs["rays"].shape == (robot.body.num_rays, RAY_DIM)
    assert obs["proprio"].shape == (PROPRIO_DIM,)
    assert obs["sound"].shape == (4,)
    assert obs["events"].shape == (4,)
    assert obs["events"][0] == 1.0
    assert robot.events[0] == 0.0, "events drain on observe"
    # Kind channel is one-hot; colors stay in [0, 1].
    np.testing.assert_allclose(obs["rays"][:, 4:].sum(axis=1), 1.0)
    assert obs["rays"][:, 1:4].min() >= 0.0 and obs["rays"][:, 1:4].max() <= 1.0


def test_proprio_carries_gaze() -> None:
    grid = scene()
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    robot.gaze[:] = [0.5, -0.25]
    proprio = observe(grid.blocks, [robot], light_level=1.0)["a"]["proprio"]
    np.testing.assert_allclose(proprio[15], 0.5, atol=1e-6)
    np.testing.assert_allclose(proprio[16], -0.25, atol=1e-6)


def test_proprio_carries_in_water() -> None:
    grid = scene()
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    dry = observe(grid.blocks, [robot], light_level=1.0)["a"]["proprio"]
    assert dry[18] == 0.0  # IN_WATER_IDX
    robot.in_water = True
    wet = observe(grid.blocks, [robot], light_level=1.0)["a"]["proprio"]
    assert wet[18] == 1.0


def test_dormant_robot_seen_not_seeing() -> None:
    grid = scene()
    viewer = Robot(id="a", pos=np.array([4.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    sleeper = Robot(id="b", pos=np.array([7.5, 8.5, 1.0]), yaw=0.0, brain_name="t", dormant=True)
    obs = observe(grid.blocks, [viewer, sleeper], light_level=1.0)
    assert "b" not in obs
    assert (ray_kinds(obs["a"]["rays"]) == RAY_KIND_DORMANT).any()


def test_misses_see_the_sky_and_night_is_dark() -> None:
    grid = scene()
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    robot.gaze[:] = [1.0, 0.0]  # look up: the steep rows clear the border walls
    day = observe(grid.blocks, [robot], light_level=1.0)["a"]["rays"]
    night = observe(grid.blocks, [robot], light_level=0.0)["a"]["rays"]
    day_miss = ray_kinds(day) == RAY_KIND_NOTHING
    assert day_miss.any(), "upward rays over a flat floor must miss"
    assert np.allclose(day[day_miss, 1:4], sky_color(1.0), atol=1e-6)
    # The same scene at night: sky nearly black, lit surfaces dimmed, depth unchanged.
    assert night[day_miss, 1:4].max() < 0.1
    day_floor = ray_kinds(day) == RAY_KIND_BLOCK
    assert night[day_floor, 1:4].mean() < day[day_floor, 1:4].mean() * 0.2
    np.testing.assert_allclose(night[:, 0], day[:, 0], atol=1e-6)


def test_voxel_grain_is_deterministic_and_bounded() -> None:
    cells = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [-1, 0, 200]])
    t = tint_factors(cells)
    assert t[0] == t[1], "same voxel, same grain, forever"
    assert t[0] != t[2], "different voxels differ"
    assert (t >= 0.88 - 1e-6).all() and (t <= 1.12 + 1e-6).all()


def test_same_block_type_looks_different_across_voxels() -> None:
    grid = scene()
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    rays = observe(grid.blocks, [robot], light_level=1.0)["a"]["rays"]
    floor = ray_kinds(rays) == RAY_KIND_BLOCK
    # All floor hits are ROCK, yet the grain gives the view spatial structure.
    assert len(np.unique(rays[floor, 1:4].round(4), axis=0)) > 1


def test_toxic_bush_color_differs_unless_mimic() -> None:
    grid = scene()
    grid.blocks[12, 8, 1] = Block.BUSH_TOXIC  # eye-height, one toxic bush
    viewer = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    viewer.pos[2] = 0.5  # eye at z≈1.25, in the bush's cell layer

    toxic_chroma = chroma(COLOR[Block.BUSH_TOXIC].astype(np.float32) / 255.0)
    ripe_chroma = chroma(COLOR[Block.BUSH_RIPE].astype(np.float32) / 255.0)

    def nearest(rays: np.ndarray) -> np.ndarray:
        blocks_hit = ray_kinds(rays) == RAY_KIND_BLOCK
        c = chroma(rays[blocks_hit, 1:4])
        d_toxic = ((c - toxic_chroma) ** 2).sum(axis=1)
        d_ripe = ((c - ripe_chroma) ** 2).sum(axis=1)
        return np.stack([d_toxic, d_ripe])

    d_toxic, d_ripe = nearest(observe(grid.blocks, [viewer], light_level=1.0)["a"]["rays"])
    assert (d_toxic < 1e-4).any(), "the toxic bush wears its purple"
    assert not (d_ripe < 1e-4).any()

    d_toxic, d_ripe = nearest(
        observe(grid.blocks, [viewer], light_level=1.0, toxic_mimic=True)["a"]["rays"]
    )
    assert not (d_toxic < 1e-4).any(), "under mimicry the purple is gone"
    assert (d_ripe < 1e-4).any(), "the toxic bush wears ripe-red"


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


def test_world_sound_is_heard_with_bearing() -> None:
    grid = scene()
    hearer = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    # A death cry 4 blocks ahead (+x): mix carries its pattern, bearing is 0.
    cry = (12.5, 8.5, -1.0, -1.0)
    obs = observe(grid.blocks, [hearer], light_level=1.0, world_sounds=[cry])["a"]
    assert obs["sound"][0] < 0.0 and obs["sound"][1] < 0.0
    np.testing.assert_allclose(obs["sound"][2], 0.0, atol=1e-6)
    np.testing.assert_allclose(obs["sound"][3], 1.0, atol=1e-6)


def test_world_sound_beyond_hear_radius_is_silent() -> None:
    grid = scene()
    hearer = Robot(id="a", pos=np.array([1.5, 1.5, 1.0]), yaw=0.0, brain_name="t")
    cry = (14.5, 14.5, -1.0, -1.0)  # ~18 blocks away, radius is 12
    obs = observe(grid.blocks, [hearer], light_level=1.0, world_sounds=[cry])["a"]
    assert (obs["sound"] == 0).all()


def test_silence_when_alone() -> None:
    grid = scene()
    robot = Robot(id="a", pos=np.array([8.5, 8.5, 1.0]), yaw=0.0, brain_name="t")
    obs = observe(grid.blocks, [robot], light_level=1.0)["a"]
    assert (obs["sound"] == 0).all()
    assert (ray_kinds(obs["rays"]) != RAY_KIND_NOTHING).sum() > 0  # floor/border visible
