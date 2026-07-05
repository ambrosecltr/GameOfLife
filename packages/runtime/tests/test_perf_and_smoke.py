"""Perf floor + full-stack smoke: the tests that keep long runs viable."""

import dataclasses
import time
from pathlib import Path

import pytest
from gol_runtime.config import PopulationConfig, RunConfig
from gol_runtime.loop import SimLoop
from gol_runtime.scheduler import Population
from gol_world import persistence
from gol_world.config import WorldConfig
from gol_world.world import World

# The sim must comfortably outrun real time (20 ticks/s) even with a full
# scripted population, or paced runs starve the learner of wall-clock.
PERF_FLOOR_TICKS_PER_SEC = 400.0


@pytest.mark.slow
def test_perf_floor_16_agents(tmp_path: Path) -> None:
    cfg = WorldConfig(seed=13, size=(256, 256, 64))
    run = RunConfig(
        checkpoint_interval_ticks=10**9,
        population=PopulationConfig(
            target=16,
            mix=(
                {"brain": {"kind": "scripted_forager"}, "count": 8},
                {"brain": {"kind": "random_walker"}, "count": 8},
            ),
        ),
    )
    save = tmp_path / "perf"
    persistence.create_save(save, cfg, run_config=dataclasses.asdict(run))
    world = World.new(cfg)
    population = Population(world, run)
    loop = SimLoop(world, save, run, act_step=population.act_step)
    loop.run(max_ticks=200, paced=False)  # warm caches

    began = time.perf_counter()
    loop.run(max_ticks=1000, paced=False)
    rate = 1000 / (time.perf_counter() - began)
    assert rate > PERF_FLOOR_TICKS_PER_SEC, (
        f"sim too slow: {rate:.0f} ticks/s with 16 agents "
        f"(floor {PERF_FLOOR_TICKS_PER_SEC}); an O(world) scan probably crept in"
    )


@pytest.mark.slow
def test_smoke_mixed_population_with_learner(tmp_path: Path) -> None:
    """Headless end-to-end: scripted + dreamer, checkpoint, resume."""
    cfg = WorldConfig(seed=17, size=(96, 96, 48), day_length_ticks=4000)
    dreamer_spec = {
        "kind": "dreamer",
        "preset": "nano",
        "replay": {"capacity": 5000, "batch_size": 4, "seq_len": 16, "warmup_steps": 100},
        "training": {"imag_starts": 32},
        "actor_critic": {"imagination_horizon": 5},
    }
    run = RunConfig(
        checkpoint_interval_ticks=2000,
        population=PopulationConfig(
            target=4,
            respawn_delay_ticks=200,
            inherit_weights="lineage",
            mix=(
                {"brain": dreamer_spec, "count": 1},
                {"brain": {"kind": "scripted_forager"}, "count": 2},
                {"brain": {"kind": "random_walker"}, "count": 1},
            ),
        ),
    )
    save = tmp_path / "smoke"
    persistence.create_save(save, cfg, run_config=dataclasses.asdict(run))
    world = World.new(cfg)
    population = Population(world, run)

    def act_and_learn(w: World) -> None:
        population.act_step(w)
        population.sync_learn()  # inline: deterministic and thread-free in tests

    loop = SimLoop(world, save, run, brain_states=population.brain_states, act_step=act_and_learn)
    loop.run(max_ticks=5000, paced=False)

    assert len(world.robots) > 0, "population died out entirely"
    assert persistence.latest_checkpoint(save) is not None
    dreamer_ids = population.learning_ids()
    assert dreamer_ids, "the dreamer lineage should still be embodied"
    metrics = population.brains[dreamer_ids[0]].introspect()
    assert metrics.get("updates", 0) > 0, "the dreamer should have learned"

    # Resume and keep going.
    resumed = persistence.load_world(save)
    population2 = Population(resumed, run)
    ckpt = persistence.latest_checkpoint(save)
    assert ckpt is not None
    population2.restore_brain_states(persistence.load_brain_states(ckpt))
    loop2 = SimLoop(resumed, save, run, act_step=population2.act_step)
    loop2.run(max_ticks=500, paced=False)
    # run() leaves a final checkpoint at the last tick, so resume starts there.
    assert resumed.tick == world.tick + 500
