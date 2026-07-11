"""Population lifecycle: deaths lead to delayed respawns with fresh brains."""

import dataclasses
import pickle
from pathlib import Path
from typing import Any

import pytest
from gol_brains.base import Brain
from gol_runtime.config import PopulationConfig, RunConfig
from gol_runtime.loop import SimLoop
from gol_runtime.scheduler import Population
from gol_world import persistence
from gol_world.config import EconomyConfig, WorldConfig
from gol_world.interface import Action
from gol_world.world import World

CFG = WorldConfig(
    seed=21,
    size=(64, 64, 40),
    day_length_ticks=1000,
    economy=EconomyConfig(hibernate_integrity_drain=2.0),  # dormancy kills fast
)
RUN = RunConfig(
    checkpoint_interval_ticks=100_000,
    population=PopulationConfig(
        target=3,
        respawn_delay_ticks=50,
        mix=({"brain": {"kind": "random_walker"}, "count": 3},),
    ),
)


def test_death_triggers_respawn(tmp_path: Path) -> None:
    save = tmp_path / "save"
    persistence.create_save(save, CFG, run_config=dataclasses.asdict(RUN))
    world = World.new(CFG)
    population = Population(world, RUN)
    assert set(world.robots) == {"walker_000", "walker_001", "walker_002"}

    # Starve one robot to death.
    world.robots["walker_001"].energy = 0.0
    world.robots["walker_001"].integrity = 0.5

    loop = SimLoop(
        world,
        save,
        RUN,
        act_step=population.act_step,
        after_world_step=population.on_world_tick,
    )
    loop.run(max_ticks=300, paced=False)

    assert "walker_001" not in world.robots
    assert "walker_003" in world.robots, "a replacement should have spawned"
    assert len(world.robots) == 3
    assert "walker_003" in population.brains
    assert "walker_001" not in population.brains


def test_descendant_inheritance_creates_a_distinct_brain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InheritingBrain(Brain):
        def __init__(self) -> None:
            self.learned = 0
            self.inherited_from: int | None = None

        def act(self, obs: Any) -> Action:
            del obs
            return Action.idle()

        def state_dict(self) -> dict[str, Any]:
            return {"learned": self.learned}

        def inherit(self, state: dict[str, Any]) -> None:
            self.inherited_from = int(state["learned"])
            self.learned = self.inherited_from

    monkeypatch.setattr(
        "gol_runtime.scheduler.build_brain", lambda spec, seed, device="cpu": InheritingBrain()
    )
    run = RunConfig(
        checkpoint_interval_ticks=100_000,
        population=PopulationConfig(
            target=1,
            respawn_delay_ticks=5,
            inherit_weights="descendant",
            mix=({"brain": {"kind": "random_walker"}, "count": 1},),
        ),
    )
    world = World.new(WorldConfig(seed=25, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, run)
    parent = population.brains["walker_000"]
    assert isinstance(parent, InheritingBrain)
    parent.learned = 73
    population.act_step(world)
    del world.robots["walker_000"]
    population.on_world_tick(world)
    world.tick = 5
    population.act_step(world)

    child = population.brains["walker_001"]
    assert isinstance(child, InheritingBrain)
    assert child is not parent
    assert child.inherited_from == 73
    inheritance = next(event for event in world.consume_events() if event["kind"] == "inherit")
    assert inheritance["parent"] == "walker_000"
    assert inheritance["robot"] == "walker_001"


def test_wake_from_dormancy_calls_wake_hook() -> None:
    world = World.new(WorldConfig(seed=21, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, RUN)
    rid = "walker_000"
    calls: list[tuple[str, int]] = []
    population.brains[rid].wake = (  # type: ignore[method-assign]
        lambda dormant_steps=0: calls.append((rid, dormant_steps))
    )

    population.act_step(world)  # awake baseline: no wake
    assert calls == []
    robot = world.robots[rid]
    robot.dormant = True
    population.act_step(world)  # dormant: unobserved, still no wake
    assert calls == []
    robot.dormant = False
    robot.energy = 50.0
    population.act_step(world)  # first awake cycle after the gap
    assert calls == [(rid, 1)]
    population.act_step(world)  # wake fires once, not every cycle
    assert calls == [(rid, 1)]


def test_dormant_duration_survives_population_checkpoint() -> None:
    world = World.new(WorldConfig(seed=22, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, RUN)
    rid = "walker_000"
    world.robots[rid].dormant = True
    population.act_step(world)
    population.act_step(world)

    restored = Population(world, RUN)
    restored.restore_brain_states(population.brain_states())
    calls: list[int] = []
    restored.brains[rid].wake = (  # type: ignore[method-assign]
        lambda dormant_steps=0: calls.append(dormant_steps)
    )
    world.robots[rid].dormant = False
    world.robots[rid].energy = 50.0
    restored.act_step(world)
    assert calls == [2]


def test_dormant_terminal_observation_survives_population_checkpoint() -> None:
    world = World.new(WorldConfig(seed=23, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, LINEAGE_RUN)
    rid = "walker_000"
    population.act_step(world)
    world.robots[rid].dormant = True
    population.act_step(world)

    restored = Population(world, LINEAGE_RUN)
    restored.restore_brain_states(population.brain_states())
    calls: list[tuple[bool, int]] = []
    restored.brains[rid].record_death = (  # type: ignore[method-assign]
        lambda obs, dormant=False, dormant_steps=0: calls.append((dormant, dormant_steps))
    )
    del world.robots[rid]
    restored.on_world_tick(world)

    assert calls == [(True, 1)]


def test_default_wake_is_a_stream_cut() -> None:
    """Brains that don't override wake() keep the legacy reset semantics."""
    world = World.new(WorldConfig(seed=21, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, RUN)
    rid = "walker_000"
    resets: list[str] = []
    population.brains[rid].reset_stream = lambda: resets.append(rid)  # type: ignore[method-assign]
    robot = world.robots[rid]
    robot.dormant = True
    population.act_step(world)
    robot.dormant = False
    robot.energy = 50.0
    population.act_step(world)
    assert resets == [rid]


LINEAGE_RUN = RunConfig(
    checkpoint_interval_ticks=100_000,
    population=PopulationConfig(
        target=3,
        respawn_delay_ticks=50,
        inherit_weights="lineage",
        mix=({"brain": {"kind": "random_walker"}, "count": 3},),
    ),
)


def test_death_delivers_final_observation_to_brain() -> None:
    """A body's end is real experience the brain could never sense (dormant
    bodies don't act; the death tick removes the robot before observation) —
    the scheduler delivers the last observation via record_death, flagged
    with how the body died (hibernating or awake)."""
    world = World.new(WorldConfig(seed=21, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, LINEAGE_RUN)
    rid = "walker_000"
    calls: list[tuple[bool, int]] = []
    brain = population.brains[rid]
    brain.record_death = (  # type: ignore[method-assign]
        lambda obs, dormant=False, dormant_steps=0: calls.append((dormant, dormant_steps))
    )
    population.act_step(world)  # records a last observation while awake
    world.robots[rid].dormant = True
    population.act_step(world)  # dormant: unobserved, tracked as hibernating
    del world.robots[rid]  # the body dies on the hibernation clock
    population.act_step(world)
    assert calls == [(True, 1)]


def test_death_delivery_never_blocks_on_a_busy_learner() -> None:
    """The learner worker may hold the brain's lock mid-update when the body
    dies: the sim never waits — delivery defers and retries next act-step."""
    world = World.new(WorldConfig(seed=21, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, LINEAGE_RUN)
    rid = "walker_000"
    calls: list[tuple[bool, int]] = []
    population.brains[rid].record_death = (  # type: ignore[method-assign]
        lambda obs, dormant=False, dormant_steps=0: calls.append((dormant, dormant_steps))
    )
    population.act_step(world)
    lock = population.locks[rid]
    lock.acquire()  # a learn() in flight
    del world.robots[rid]  # an awake death (fall/poison)
    population.act_step(world)
    assert calls == [], "delivery must not block the sim"
    lock.release()
    population.act_step(world)
    assert calls == [(False, 0)]


def test_checkpoint_reconciles_death_between_act_boundaries() -> None:
    world = World.new(WorldConfig(seed=24, size=(64, 64, 40), day_length_ticks=1000))
    population = Population(world, LINEAGE_RUN)
    rid = "walker_000"
    population.act_step(world)
    world.tick = 1
    del world.robots[rid]

    blobs = population.brain_states()

    assert rid not in population.brains
    assert "__lineage__random_walker__0" in blobs
    scheduler = pickle.loads(blobs["__scheduler__"])
    assert (55, "random_walker") in scheduler["respawn_queue"]
    assert scheduler["stash_parent_ids"] == {"random_walker": (rid,)}
