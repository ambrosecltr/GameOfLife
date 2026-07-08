"""Earned reproduction (proposal 004): budding replaces the respawn timer for
evolving lineages, gated on a thriving body, with a low extinction floor."""

from __future__ import annotations

from gol_runtime.config import PopulationConfig, ReproductionConfig, RunConfig
from gol_runtime.scheduler import Population
from gol_world.config import WorldConfig
from gol_world.world import World

CFG = WorldConfig(seed=21, size=(64, 64, 40), day_length_ticks=1000)
PLASTIC = {"brain": {"kind": "plastic", "core": {"hidden": 32}}, "count": 4}
FORAGER = {"brain": {"kind": "scripted_forager"}, "count": 2}


def _run(**repro: object) -> RunConfig:
    return RunConfig(
        checkpoint_interval_ticks=100_000,
        population=PopulationConfig(target=6, mix=(PLASTIC, FORAGER)),
        reproduction=ReproductionConfig(
            mode="budding",
            min_bud_age=0,
            bud_cooldown=0,
            thrive_energy=10.0,
            thrive_integrity=10.0,
            floor=2,
            **repro,  # type: ignore[arg-type]
        ),
    )


def _plastic(world: World) -> list[str]:
    return [rid for rid in world.robots if rid.startswith("plastic")]


def test_thriving_parent_buds_to_refill_cap() -> None:
    world = World.new(CFG)
    pop = Population(world, _run())
    assert len(_plastic(world)) == 4  # founders at the cap

    victim = _plastic(world)[0]
    del world.robots[victim]  # simulate a death (evolving kinds don't timer-respawn)
    world.consume_events()

    pop.act_step(world)  # a thriving survivor should bud back to the cap
    assert victim not in world.robots
    assert len(_plastic(world)) == 4
    events = world.consume_events()
    assert any(e["kind"] == "bud" for e in events), "a bud event should fire"
    # the child inherited a brain and is tracked
    child = next(e["child"] for e in events if e["kind"] == "bud")
    assert child in pop.brains and pop.kinds[child] == "plastic"


def test_budding_charges_the_parent() -> None:
    world = World.new(CFG)
    run = _run(bud_cost_energy=30.0, bud_cost_integrity=5.0)
    pop = Population(world, run)
    del world.robots[_plastic(world)[0]]
    world.consume_events()
    pop.act_step(world)
    bud = next(e for e in world.consume_events() if e["kind"] == "bud")
    parent = world.robots[bud["robot"]]
    # a full-tank founder that just paid to bud is below the max it spawned at
    assert parent.energy <= world.cfg.economy.energy_max - 30.0 + 1e-6


def test_dormant_body_is_not_thriving() -> None:
    world = World.new(CFG)
    pop = Population(world, _run())
    r = world.robots[_plastic(world)[0]]
    r.energy = world.cfg.economy.energy_max
    r.integrity = world.cfg.economy.integrity_max
    assert pop._is_thriving(r, tick=100)
    r.dormant = True
    assert not pop._is_thriving(r, tick=100)  # a sleeping body cannot reproduce


def test_extinction_floor_respawns_when_no_one_thrives() -> None:
    world = World.new(CFG)
    pop = Population(world, _run())  # floor=2 from the helper
    # wipe all but one plastic, and make survivors non-thriving (starving)
    plastic = _plastic(world)
    for rid in plastic[1:]:
        del world.robots[rid]
    world.robots[plastic[0]].energy = 0.0  # not thriving ⇒ won't bud
    world.consume_events()
    pop.act_step(world)
    # floor guarantees a minimum even with no thriving parent
    assert len(_plastic(world)) >= 2


def test_respawn_mode_unchanged() -> None:
    world = World.new(CFG)
    run = RunConfig(
        population=PopulationConfig(
            target=6, respawn_delay_ticks=1, mix=(PLASTIC, FORAGER)
        ),
    )  # default reproduction.mode == "respawn"
    pop = Population(world, run)
    victim = _plastic(world)[0]
    del world.robots[victim]
    pop.act_step(world)
    # legacy path: a timer respawn (not a bud) — no bud events
    assert all(e["kind"] != "bud" for e in world.consume_events())
