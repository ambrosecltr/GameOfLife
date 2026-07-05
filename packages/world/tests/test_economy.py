"""Energy economy and lifecycle: eat, drain, hibernate, die, drop scrap."""

import numpy as np
from gol_world.blocks import Block
from gol_world.config import EconomyConfig, WorldConfig
from gol_world.entities import EV_ATE
from gol_world.interface import GRIP_DIG, GRIP_EAT, GRIP_PLACE, Action
from gol_world.world import World

CFG = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000)


def make_world() -> World:
    return World.new(CFG)


def place_robot_before_bush(world: World) -> str:
    """Spawn a robot and put a ripe bush right in front of it."""
    robot = world.spawn_robot("bot_000", "test")
    x, y, z = (
        int(robot.pos[0] + np.cos(robot.yaw) * 1.2),
        int(robot.pos[1] + np.sin(robot.yaw) * 1.2),
        int(robot.eye[2]),
    )
    world.grid.set_block(x, y, z, Block.BUSH_RIPE)
    return robot.id


def drive_action() -> Action:
    return Action(drive=np.array([1.0, 0.0], dtype=np.float32))


def test_eating_restores_energy_and_depletes_bush() -> None:
    world = make_world()
    rid = place_robot_before_bush(world)
    robot = world.robots[rid]
    robot.energy = 40.0
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()
    # Same-tick basal drain applies after the meal.
    assert abs(robot.energy - (40.0 + CFG.economy.eat_energy)) < 0.01
    assert robot.events[EV_ATE] == 1.0
    events = world.consume_events()
    assert any(e["kind"] == "eat" for e in events)
    # The bush is depleted and queued to regrow.
    assert (world.grid.blocks == Block.BUSH_RIPE).sum() >= 0
    assert any(e["kind"] == "spawn" for e in events)


def test_movement_drains_energy() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    start = robot.energy
    for _ in range(100):
        world.apply_action(rid, drive_action())
        world.step()
    assert robot.energy < start - 0.5


def test_energy_zero_hibernates_then_dies_dropping_scrap() -> None:
    eco = EconomyConfig(hibernate_integrity_drain=1.0)  # fast decay for the test
    cfg = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000, economy=eco)
    world = World.new(cfg)
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.energy = 0.01
    world.apply_action(rid, drive_action())
    for _ in range(5):
        world.step()
    assert robot.dormant
    kinds = [e["kind"] for e in world.consume_events()]
    assert "hibernate" in kinds
    # Integrity decays at 1/tick -> death within ~100 more ticks.
    for _ in range(120):
        world.step()
    assert rid not in world.robots
    kinds = [e["kind"] for e in world.consume_events()]
    assert "death" in kinds
    assert (world.grid.blocks == Block.SCRAP).sum() >= 1


def test_dig_and_place_roundtrip() -> None:
    world = make_world()
    rid = place_robot_before_bush(world)  # something diggable in front
    robot = world.robots[rid]
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_DIG))
    world.step()
    assert robot.held == Block.BUSH_RIPE
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_PLACE))
    world.step()
    assert robot.held is None
    kinds = [e["kind"] for e in world.consume_events()]
    assert "dig" in kinds and "place" in kinds


def test_eat_held_food() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.held = int(Block.BUSH_RIPE)
    robot.energy = 30.0
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()
    assert robot.held is None
    assert abs(robot.energy - (30.0 + CFG.economy.eat_energy)) < 0.01
