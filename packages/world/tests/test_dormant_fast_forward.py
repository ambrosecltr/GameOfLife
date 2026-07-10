"""Differential validation for event-safe universal-dormancy acceleration."""

import copy

import numpy as np
import pytest
from gol_world.blocks import Block
from gol_world.config import EcologyConfig, EconomyConfig, WorldConfig
from gol_world.entities import TOUCH_FRONT
from gol_world.world import World


def _settled_dormant_world(
    *,
    day_length: int = 1000,
    economy: EconomyConfig | None = None,
    ecology: EcologyConfig | None = None,
) -> World:
    world = World.new(
        WorldConfig(
            seed=71,
            size=(32, 32, 40),
            day_length_ticks=day_length,
            economy=economy or EconomyConfig(),
            ecology=ecology or EcologyConfig(),
        )
    )
    robot = world.spawn_robot("sleeper_000", "test")
    for _ in range(30):
        world.step()
    world.consume_events()
    robot.dormant = True
    robot.energy = 1.0
    world.step()
    world.consume_events()
    assert world.can_fast_forward_dormant()
    return world


def _assert_robot_equal(expected: World, actual: World, robot_id: str) -> None:
    left = expected.robots[robot_id]
    right = actual.robots[robot_id]
    np.testing.assert_allclose(right.pos, left.pos, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(right.vel, left.vel, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(right.drive, left.drive, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(right.signal, left.signal, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(right.gaze, left.gaze, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(right.touch, left.touch)
    np.testing.assert_array_equal(right.events, left.events)
    assert right.yaw == pytest.approx(left.yaw, abs=0.0)
    assert right.pending_grip == left.pending_grip
    assert right.in_water == left.in_water
    assert right.fall_peak_z == pytest.approx(left.fall_peak_z, abs=0.0)
    assert right.energy == pytest.approx(left.energy, abs=1e-12)
    assert right.integrity == pytest.approx(left.integrity, abs=1e-9)
    assert right.fatigue == pytest.approx(left.fatigue, abs=1e-12)
    assert right.age_ticks == left.age_ticks
    assert right.held == left.held
    assert right.held_age_ticks == left.held_age_ticks
    assert right.dormant == left.dormant
    assert right.ledger == pytest.approx(left.ledger, abs=1e-9)
    assert right.energy_ledger == pytest.approx(left.energy_ledger, abs=1e-12)


def test_dormant_fast_forward_matches_ordinary_ticks_and_rng() -> None:
    baseline = _settled_dormant_world()
    ordinary = copy.deepcopy(baseline)
    accelerated = copy.deepcopy(baseline)

    advanced = accelerated.fast_forward_dormant(500)
    assert advanced > 0
    for _ in range(advanced):
        ordinary.step()

    assert accelerated.tick == ordinary.tick
    np.testing.assert_array_equal(accelerated.grid.blocks, ordinary.grid.blocks)
    _assert_robot_equal(ordinary, accelerated, "sleeper_000")
    assert accelerated.regrow_heap == ordinary.regrow_heap
    assert accelerated.wither_heap == ordinary.wither_heap
    assert accelerated.sprout_heap == ordinary.sprout_heap
    assert accelerated.transient_sounds == ordinary.transient_sounds
    assert accelerated.consume_events() == ordinary.consume_events() == []
    assert accelerated.rng.bit_generator.state == ordinary.rng.bit_generator.state


def test_fast_forward_stops_before_wake_and_scalar_step_emits_it() -> None:
    world = _settled_dormant_world(day_length=400)
    world.tick = 100  # noon
    robot = world.robots["sleeper_000"]
    robot.energy = world.cfg.economy.wake_energy - 0.025
    ordinary = copy.deepcopy(world)

    advanced = world.fast_forward_dormant(100)
    assert 0 < advanced < 5
    for _ in range(advanced):
        ordinary.step()
    assert robot.dormant
    while robot.dormant:
        world.step()
        ordinary.step()
    assert world.tick == ordinary.tick
    events = world.consume_events()
    assert events == ordinary.consume_events()
    assert [event["kind"] for event in events] == ["wake"]


def test_fast_forward_stops_before_death_spoil_and_ecology_boundaries() -> None:
    economy = EconomyConfig(hibernate_integrity_drain=0.5, solar_trickle=0.0)
    ecology = EcologyConfig(held_spoil_ticks=5)
    world = _settled_dormant_world(economy=economy, ecology=ecology)
    robot = world.robots["sleeper_000"]
    robot.integrity = 10.0
    robot.held = int(Block.BUSH_RIPE)
    robot.held_age_ticks = 2
    world.regrow_heap = [(world.tick + 10, 0, 0, 0)]

    assert world.fast_forward_dormant(100) == 2
    assert robot.held == Block.BUSH_RIPE
    world.step()
    assert robot.held is None
    assert any(event["kind"] == "spoil" for event in world.consume_events())


def test_falling_or_awake_body_disables_fast_forward() -> None:
    world = _settled_dormant_world()
    robot = world.robots["sleeper_000"]
    robot.vel[2] = -1.0
    assert not world.can_fast_forward_dormant()
    robot.vel[2] = 0.0
    robot.dormant = False
    assert not world.can_fast_forward_dormant()


def test_newly_dormant_physics_state_normalizes_before_fast_forward() -> None:
    world = _settled_dormant_world()
    robot = world.robots["sleeper_000"]

    robot.drive[0] = 1.0
    assert not world.can_fast_forward_dormant()
    world.step()
    world.consume_events()
    assert world.can_fast_forward_dormant()

    robot.touch[TOUCH_FRONT] = True
    assert not world.can_fast_forward_dormant()
    world.step()
    world.consume_events()
    assert world.can_fast_forward_dormant()

    robot.fall_peak_z += 1.0
    assert not world.can_fast_forward_dormant()
    world.step()
    world.consume_events()
    assert world.can_fast_forward_dormant()


def test_dormant_fast_forward_matches_across_dusk_and_dawn() -> None:
    world = _settled_dormant_world(day_length=200)
    world.regrow_heap.clear()
    world.wither_heap.clear()
    world.sprout_heap.clear()
    world.tick = 90
    ordinary = copy.deepcopy(world)
    accelerated = copy.deepcopy(world)

    assert accelerated.fast_forward_dormant(120) == 120
    for _ in range(120):
        ordinary.step()

    _assert_robot_equal(ordinary, accelerated, "sleeper_000")
    assert accelerated.tick == ordinary.tick == 210


def test_dormant_fast_forward_matches_energy_saturation() -> None:
    economy = EconomyConfig(energy_max=2.0, wake_energy=100.0, solar_trickle=0.5)
    world = _settled_dormant_world(day_length=200, economy=economy)
    world.tick = 50
    robot = world.robots["sleeper_000"]
    robot.energy = 1.0
    ordinary = copy.deepcopy(world)
    accelerated = copy.deepcopy(world)

    assert accelerated.fast_forward_dormant(20) == 20
    for _ in range(20):
        ordinary.step()

    _assert_robot_equal(ordinary, accelerated, "sleeper_000")
    assert accelerated.robots["sleeper_000"].energy == 2.0


def test_transient_expiry_runs_on_scalar_boundary() -> None:
    world = _settled_dormant_world()
    expires = world.tick + 7
    world.transient_sounds = [(1.0, 2.0, -1.0, 1.0, expires)]
    ordinary = copy.deepcopy(world)

    assert world.fast_forward_dormant(100) == 6
    for _ in range(6):
        ordinary.step()
    assert world.transient_sounds == ordinary.transient_sounds
    world.step()
    ordinary.step()
    assert world.tick == ordinary.tick == expires
    assert world.transient_sounds == ordinary.transient_sounds == []


def test_multiple_sleepers_stop_at_first_integrity_death() -> None:
    economy = EconomyConfig(hibernate_integrity_drain=0.1, solar_trickle=0.0)
    world = _settled_dormant_world(economy=economy)
    second = world.spawn_robot("sleeper_001", "test")
    for _ in range(30):
        world.step()
    world.consume_events()
    first = world.robots["sleeper_000"]
    first.dormant = True
    second.dormant = True
    first.energy = second.energy = 0.0
    first.integrity = 1.0
    second.integrity = 2.0
    world.step()
    world.consume_events()
    ordinary = copy.deepcopy(world)
    accelerated = copy.deepcopy(world)

    advanced = accelerated.fast_forward_dormant(100)
    assert 0 < advanced < 9
    for _ in range(advanced):
        ordinary.step()
    while "sleeper_000" in accelerated.robots:
        accelerated.step()
        ordinary.step()

    assert set(accelerated.robots) == set(ordinary.robots) == {"sleeper_001"}
    assert accelerated.consume_events() == ordinary.consume_events()
    assert accelerated.rng.bit_generator.state == ordinary.rng.bit_generator.state


def test_ecology_collision_runs_scalar_with_identical_rng_and_event_order() -> None:
    world = _settled_dormant_world()
    world.regrow_heap.clear()
    world.wither_heap.clear()
    world.sprout_heap.clear()
    regrow = (1, 1, 2)
    wither = (2, 2, 2)
    world.grid.set_block(*regrow, Block.BUSH_EMPTY)
    world.grid.set_block(*wither, Block.BUSH_RIPE)
    due = world.tick + 6
    world.regrow_heap = [(due, *regrow)]
    world.wither_heap = [(due, *wither)]
    world.sprout_heap = [due]
    ordinary = copy.deepcopy(world)
    accelerated = copy.deepcopy(world)

    assert accelerated.fast_forward_dormant(100) == 5
    for _ in range(5):
        ordinary.step()
    accelerated.step()
    ordinary.step()

    np.testing.assert_array_equal(accelerated.grid.blocks, ordinary.grid.blocks)
    assert accelerated.regrow_heap == ordinary.regrow_heap
    assert accelerated.wither_heap == ordinary.wither_heap
    assert accelerated.sprout_heap == ordinary.sprout_heap
    assert accelerated.consume_events() == ordinary.consume_events()
    assert accelerated.rng.bit_generator.state == ordinary.rng.bit_generator.state
