"""Skipped dormant act opportunities preserve Aion's first post-wake transition."""

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from gol_brains.aion.brain import AionBrain
from gol_runtime.config import PopulationConfig, RunConfig
from gol_runtime.scheduler import Population
from gol_world import persistence
from gol_world.config import EconomyConfig, WorldConfig
from gol_world.world import World

TINY_AION = {
    "kind": "aion",
    "preset": "nano",
    "world_model": {"s5": {"model_dim": 16, "state_dim": 4, "blocks": 1}},
    "replay": {"capacity": 64, "batch_size": 1, "seq_len": 4, "warmup_steps": 8},
    "actor_critic": {"imagination_horizon": 2},
}
RUN = RunConfig(
    act_every=5,
    population=PopulationConfig(
        target=1,
        inherit_weights="lineage",
        mix=({"brain": TINY_AION, "count": 1},),
    ),
)


def _prepare() -> tuple[World, Population]:
    world = World.new(
        WorldConfig(
            seed=81,
            size=(32, 32, 40),
            day_length_ticks=1000,
            economy=EconomyConfig(solar_trickle=0.0),
        )
    )
    population = Population(world, RUN)
    robot = next(iter(world.robots.values()))
    for _ in range(30):
        world.step()
    world.consume_events()
    population.act_step(world)
    robot.dormant = True
    robot.energy = 0.0
    world.step()
    world.consume_events()
    population.act_step(world)
    return world, population


def test_fast_forward_and_scalar_dormancy_produce_same_first_wake_action() -> None:
    ordinary_world, ordinary_population = _prepare()
    state = ordinary_population.brain_states()
    accelerated_world = copy.deepcopy(ordinary_world)
    accelerated_population = Population(accelerated_world, RUN)
    accelerated_population.restore_brain_states(state)
    ordinary_population.restore_brain_states(state)
    ordinary_world.regrow_heap.clear()
    ordinary_world.wither_heap.clear()
    ordinary_world.sprout_heap.clear()
    accelerated_world.regrow_heap.clear()
    accelerated_world.wither_heap.clear()
    accelerated_world.sprout_heap.clear()

    start_tick = ordinary_world.tick
    raw_ticks = 100
    for _ in range(raw_ticks):
        ordinary_world.step()
        if ordinary_world.tick % RUN.act_every == 0:
            ordinary_population.act_step(ordinary_world)
    advanced = accelerated_world.fast_forward_dormant(raw_ticks)
    assert advanced == raw_ticks
    accelerated_population.advance_dormant_opportunities(
        start_tick, accelerated_world.tick
    )

    ordinary_robot = next(iter(ordinary_world.robots.values()))
    accelerated_robot = next(iter(accelerated_world.robots.values()))
    ordinary_robot.dormant = False
    accelerated_robot.dormant = False
    ordinary_robot.energy = accelerated_robot.energy = 50.0
    torch.manual_seed(811)
    ordinary_population.act_step(ordinary_world)
    torch.manual_seed(811)
    accelerated_population.act_step(accelerated_world)

    ordinary_brain = next(iter(ordinary_population.brains.values()))
    accelerated_brain = next(iter(accelerated_population.brains.values()))
    assert isinstance(ordinary_brain, AionBrain)
    assert isinstance(accelerated_brain, AionBrain)
    np.testing.assert_allclose(accelerated_robot.drive, ordinary_robot.drive, atol=1e-7)
    np.testing.assert_allclose(accelerated_robot.signal, ordinary_robot.signal, atol=1e-7)
    np.testing.assert_allclose(accelerated_robot.gaze, ordinary_robot.gaze, atol=1e-7)
    torch.testing.assert_close(accelerated_brain.h, ordinary_brain.h, atol=2e-6, rtol=2e-5)
    accelerated_index = len(accelerated_brain.buffer) - 1
    ordinary_index = len(ordinary_brain.buffer) - 1
    assert (
        accelerated_brain.buffer.wake[accelerated_index]
        == ordinary_brain.buffer.wake[ordinary_index]
        == 1
    )
    assert accelerated_brain.buffer.step_scale[accelerated_index] == pytest.approx(
        ordinary_brain.buffer.step_scale[ordinary_index]
    )


def test_dormant_fast_forward_checkpoint_resume_preserves_first_wake(
    tmp_path: Path,
) -> None:
    world, population = _prepare()
    world.regrow_heap.clear()
    world.wither_heap.clear()
    world.sprout_heap.clear()
    start_tick = world.tick
    assert world.fast_forward_dormant(100) == 100
    population.advance_dormant_opportunities(start_tick, world.tick)

    save = tmp_path / "dormant-resume"
    persistence.create_save(save, world.cfg, run_config=dataclasses.asdict(RUN))
    checkpoint = persistence.save_checkpoint(save, world, population.brain_states())
    resumed_world = persistence.load_world(save, checkpoint)
    resumed_population = Population(resumed_world, RUN)
    resumed_population.restore_brain_states(persistence.load_brain_states(checkpoint))

    reference_robot = next(iter(world.robots.values()))
    resumed_robot = next(iter(resumed_world.robots.values()))
    reference_robot.dormant = resumed_robot.dormant = False
    reference_robot.energy = resumed_robot.energy = 50.0
    torch.manual_seed(812)
    population.act_step(world)
    torch.manual_seed(812)
    resumed_population.act_step(resumed_world)

    reference_brain = next(iter(population.brains.values()))
    resumed_brain = next(iter(resumed_population.brains.values()))
    assert isinstance(reference_brain, AionBrain)
    assert isinstance(resumed_brain, AionBrain)
    np.testing.assert_allclose(resumed_robot.drive, reference_robot.drive, atol=1e-7)
    np.testing.assert_allclose(resumed_robot.signal, reference_robot.signal, atol=1e-7)
    np.testing.assert_allclose(resumed_robot.gaze, reference_robot.gaze, atol=1e-7)
    torch.testing.assert_close(resumed_brain.h, reference_brain.h, atol=2e-6, rtol=2e-5)
    assert resumed_brain._step_scale == reference_brain._step_scale == 1.0
