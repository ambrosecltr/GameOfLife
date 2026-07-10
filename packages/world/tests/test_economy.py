"""Energy economy and lifecycle: eat, drain, hibernate, die, drop scrap."""

import numpy as np
import pytest
from gol_world.blocks import Block
from gol_world.config import EcologyConfig, EconomyConfig, WorldConfig
from gol_world.entities import EV_ATE, EV_TOOK_DAMAGE
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


def test_dormant_robot_recharges_in_daylight_and_wakes() -> None:
    world = make_world()
    world.tick = CFG.day_length_ticks // 4  # noon: light_level == 1.0
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.energy = CFG.economy.wake_energy - 0.5
    robot.dormant = True
    for _ in range(300):  # 0.5 energy at solar_trickle*1.0 needs ~167 ticks
        world.step()
    assert not robot.dormant
    assert any(e["kind"] == "wake" for e in world.consume_events())


def test_dormant_robot_stays_down_at_night() -> None:
    world = make_world()
    world.tick = CFG.day_length_ticks * 3 // 4  # midnight: light_level == 0.0
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.energy = 0.0
    robot.dormant = True
    start_integrity = robot.integrity
    for _ in range(200):
        world.step()
    assert robot.dormant
    assert robot.energy == 0.0
    assert robot.integrity < start_integrity


def test_feeding_wakes_a_dormant_robot() -> None:
    world = make_world()
    feeder = world.robots[world.spawn_robot("bot_000", "test").id]
    feeder.held = int(Block.BUSH_RIPE)
    # A collapsed body one block ahead, in the feeder's gaze.
    pos = feeder.pos + np.array([np.cos(feeder.yaw), np.sin(feeder.yaw), 0.0]) * 1.2
    sleeper = world.spawn_robot("bot_001", "test")
    sleeper.pos[:] = pos
    sleeper.energy = 0.0
    sleeper.dormant = True
    world.apply_action(
        feeder.id, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_PLACE)
    )
    world.step()
    assert feeder.held is None
    assert abs(sleeper.energy - CFG.economy.eat_energy) < 0.5
    kinds = [e["kind"] for e in world.consume_events()]
    assert "feed" in kinds
    world.step()  # wake happens in the recipient's own energy accounting
    assert not sleeper.dormant


def test_place_still_places_blocks_when_no_dormant_target() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.held = int(Block.BUSH_RIPE)
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_PLACE))
    world.step()
    assert robot.held is None
    assert any(e["kind"] == "place" for e in world.consume_events())


def test_eating_toxic_bush_poisons() -> None:
    world = make_world()
    rid = place_robot_before_bush(world)
    robot = world.robots[rid]
    # place_robot_before_bush put a ripe bush in the gaze; swap it for a toxic one.
    bx = int(robot.pos[0] + np.cos(robot.yaw) * 1.2)
    by = int(robot.pos[1] + np.sin(robot.yaw) * 1.2)
    world.grid.set_block(bx, by, int(robot.eye[2]), Block.BUSH_TOXIC)
    robot.energy = 40.0
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()
    assert abs(robot.energy - (40.0 + CFG.economy.toxic_energy)) < 0.01
    assert abs(robot.integrity - (100.0 - CFG.economy.toxic_integrity_damage)) < 0.01
    assert robot.events[EV_ATE] == 1.0 and robot.events[EV_TOOK_DAMAGE] == 1.0
    assert any(e["kind"] == "poisoned" for e in world.consume_events())
    assert world.active_sounds()  # the hurt cry is audible


def test_feeding_toxic_food_poisons_the_sleeper() -> None:
    world = make_world()
    feeder = world.robots[world.spawn_robot("bot_000", "test").id]
    feeder.held = int(Block.BUSH_TOXIC)
    pos = feeder.pos + np.array([np.cos(feeder.yaw), np.sin(feeder.yaw), 0.0]) * 1.2
    sleeper = world.spawn_robot("bot_001", "test")
    sleeper.pos[:] = pos
    sleeper.energy = 0.0
    sleeper.dormant = True
    world.apply_action(
        feeder.id, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_PLACE)
    )
    world.step()
    assert feeder.held is None
    assert abs(sleeper.energy - CFG.economy.toxic_energy) < 0.5
    assert sleeper.integrity < 100.0 - CFG.economy.toxic_integrity_damage + 0.5
    kinds = [e["kind"] for e in world.consume_events()]
    assert "feed" in kinds and "poisoned" in kinds


def test_regrowth_can_come_back_toxic() -> None:
    eco = EcologyConfig(regrow_ticks=10, regrow_jitter=0, toxic_fraction=1.0)
    cfg = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000, ecology=eco)
    world = World.new(cfg)
    rid = place_robot_before_bush(world)
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()  # eat -> BUSH_EMPTY, regrow due ~tick 11 (daytime)
    world.tick = 40  # skip generation-seeded backlog jitter; stay in daylight
    world.step()
    assert (world.grid.blocks == Block.BUSH_TOXIC).any()


def test_fatigue_builds_active_recovers_at_rest() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    for _ in range(200):
        world.apply_action(rid, drive_action())
        world.step()
    peak = robot.fatigue
    expected = 200 * (CFG.economy.fatigue_rise_base + CFG.economy.fatigue_rise_active)
    assert abs(peak - expected) < 1e-6
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32)))
    for _ in range(200):
        world.step()
    assert robot.fatigue < peak
    assert abs(robot.fatigue - max(0.0, peak - 200 * CFG.economy.fatigue_recover)) < 1e-6


def test_resting_discounts_basal_drain() -> None:
    def drained(mult: float) -> float:
        eco = EconomyConfig(rest_basal_mult=mult)
        cfg = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000, economy=eco)
        world = World.new(cfg)
        rid = world.spawn_robot("bot_000", "test").id
        robot = world.robots[rid]
        robot.energy = 50.0  # under repair_threshold: repair can't touch the meter
        for _ in range(200):
            world.step()  # no drive: resting throughout
        return 50.0 - robot.energy

    sleeping = drained(CFG.economy.rest_basal_mult)
    flat = drained(1.0)
    assert abs(sleeping / flat - CFG.economy.rest_basal_mult) < 1e-6


def test_night_rest_recovers_fatigue_faster() -> None:
    def recovered(start_tick: int) -> float:
        world = make_world()
        rid = world.spawn_robot("bot_000", "test").id
        robot = world.robots[rid]
        robot.fatigue = 0.5
        world.tick = start_tick
        for _ in range(100):
            world.step()
        return 0.5 - robot.fatigue

    by_day = recovered(500)  # sun high: light 1.0, no bonus
    by_night = recovered(1500)  # deep night: light 0.0, full bonus
    assert abs(by_night / by_day - (1.0 + CFG.economy.night_rest_bonus)) < 1e-6


def test_night_repair_outpaces_day_repair() -> None:
    def repaired(start_tick: int) -> float:
        world = make_world()
        rid = world.spawn_robot("bot_000", "test").id
        robot = world.robots[rid]
        robot.integrity = 50.0
        world.tick = start_tick
        for _ in range(100):
            world.step()
        return robot.ledger["repaired"]

    ratio = repaired(1500) / repaired(500)
    assert abs(ratio - (1.0 + CFG.economy.night_rest_bonus)) < 1e-6


def test_exhaustion_bleeds_integrity_and_multiplies_drain() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.energy = 50.0  # below repair_threshold so repair can't mask the bleed
    robot.fatigue = CFG.economy.exhaustion_threshold - 1e-5
    start_integrity = robot.integrity
    for _ in range(100):
        world.apply_action(rid, drive_action())
        world.step()
    assert robot.fatigue > CFG.economy.exhaustion_threshold
    assert robot.integrity <= start_integrity - 99 * CFG.economy.exhaustion_integrity_drain
    assert any(e["kind"] == "exhausted" for e in world.consume_events())


def _yaw_travel(world: World, rid: str, ticks: int = 10) -> float:
    """Total yaw swept at full turn command — actuation without terrain in the way."""
    robot = world.robots[rid]
    travel = 0.0
    for _ in range(ticks):
        before = robot.yaw
        world.apply_action(rid, Action(drive=np.array([0.0, 1.0], dtype=np.float32)))
        world.step()
        travel += abs((robot.yaw - before + np.pi) % (2 * np.pi) - np.pi)
    return travel


def test_brownout_scales_actuation_with_depletion() -> None:
    def travel_at(energy: float) -> float:
        world = make_world()
        rid = world.spawn_robot("bot_000", "test").id
        world.robots[rid].energy = energy
        return _yaw_travel(world, rid)

    eco = CFG.economy
    full = travel_at(eco.energy_max)
    starving = travel_at(10.0)
    expected = eco.brownout_floor + (1.0 - eco.brownout_floor) * (10.0 / eco.brownout_threshold)
    assert starving < full
    assert abs(starving / full - expected) < 0.05
    # Above the threshold (with margin for drain) the body is at full strength.
    assert abs(travel_at(eco.brownout_threshold + 5.0) - full) < 1e-9


def test_brownout_threshold_zero_disables() -> None:
    eco = EconomyConfig(brownout_threshold=0.0)
    cfg = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000, economy=eco)

    def travel_at(energy: float) -> float:
        world = World.new(cfg)
        rid = world.spawn_robot("bot_000", "test").id
        world.robots[rid].energy = energy
        return _yaw_travel(world, rid)

    assert abs(travel_at(10.0) - travel_at(eco.energy_max)) < 1e-9


def test_eat_reaches_bush_below_eye_level() -> None:
    """A bush one block downhill sits below the eye row; the mouth dips to it."""
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    x = int(robot.pos[0] + np.cos(robot.yaw) * 1.2)
    y = int(robot.pos[1] + np.sin(robot.yaw) * 1.2)
    world.grid.set_block(x, y, int(robot.eye[2]), Block.AIR)  # clear the eye row
    world.grid.set_block(x, y, int(robot.eye[2]) - 1, Block.BUSH_RIPE)
    robot.energy = 40.0
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()
    assert robot.events[EV_ATE] == 1.0
    assert abs(robot.energy - (40.0 + CFG.economy.eat_energy)) < 0.01


def test_eat_blocked_by_wall_at_eye_level() -> None:
    """A solid non-edible block in the gaze still blocks the bite."""
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    x = int(robot.pos[0] + np.cos(robot.yaw) * 1.2)
    y = int(robot.pos[1] + np.sin(robot.yaw) * 1.2)
    world.grid.set_block(x, y, int(robot.eye[2]), Block.ROCK)
    bx = int(robot.pos[0] + np.cos(robot.yaw) * 1.5)
    by = int(robot.pos[1] + np.sin(robot.yaw) * 1.5)
    if (bx, by) != (x, y):  # a bush behind the wall must be unreachable
        world.grid.set_block(bx, by, int(robot.eye[2]), Block.BUSH_RIPE)
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()
    assert robot.events[EV_ATE] == 0.0


def test_repair_from_energy_surplus_while_resting() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.integrity = 50.0
    start_energy = robot.energy
    world.tick = 400  # full daylight throughout: the circadian rest bonus stays 1x
    for _ in range(200):
        world.step()
    eco = CFG.economy
    expected_rate = eco.repair_rate * eco.rest_repair_mult  # young, resting
    gained = robot.integrity - 50.0
    assert abs(gained - 200 * (expected_rate - eco.awake_wear)) < 0.01
    # Repair is paid for in energy, on top of basal drain.
    spent = start_energy - robot.energy
    assert spent > 200 * eco.basal_drain
    assert abs(robot.ledger["repaired"] - 200 * expected_rate) < 0.01
    assert abs(robot.ledger["wear"] - 200 * eco.awake_wear) < 1e-9


def test_no_repair_below_energy_threshold() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.integrity = 50.0
    robot.energy = CFG.economy.repair_threshold - 5.0
    for _ in range(200):
        world.step()
    # Only chronic wear moves integrity; nothing is repaired.
    assert robot.ledger["repaired"] == 0.0
    assert abs((50.0 - robot.integrity) - 200 * CFG.economy.awake_wear) < 1e-9


def test_repair_never_spends_energy_below_threshold() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.integrity = 10.0
    robot.energy = CFG.economy.repair_threshold + 1.0
    for _ in range(500):
        world.step()
    # Repair may spend the 1.0 surplus at most; basal drain took its share too.
    assert 0.5 < robot.ledger["repaired"] <= 1.0
    frozen = robot.ledger["repaired"]
    for _ in range(200):  # energy is under the threshold now: repair has stopped
        world.step()
    assert robot.ledger["repaired"] == frozen


def test_senescence_halves_repair_efficiency() -> None:
    def repaired_at(age: int) -> float:
        world = make_world()
        rid = world.spawn_robot("bot_000", "test").id
        robot = world.robots[rid]
        robot.integrity = 50.0
        robot.age_ticks = age
        for _ in range(100):
            world.step()
        return robot.ledger["repaired"]

    young = repaired_at(0)
    old = repaired_at(int(CFG.economy.senescence_halflife))
    assert abs(old / young - 0.5) < 0.01


def test_rest_repairs_faster_than_activity() -> None:
    def repaired(drive: Action | None) -> float:
        world = make_world()
        rid = world.spawn_robot("bot_000", "test").id
        robot = world.robots[rid]
        robot.integrity = 50.0
        world.tick = 400  # full daylight: no circadian bonus muddying the ratio
        if drive is not None:
            world.apply_action(rid, drive)  # drive persists between act-steps
        for _ in range(100):
            world.step()
        return robot.ledger["repaired"]

    resting = repaired(None)
    turning = repaired(Action(drive=np.array([0.0, 1.0], dtype=np.float32)))
    assert abs(resting / turning - CFG.economy.rest_repair_mult) < 0.05


def test_repair_rate_zero_disables() -> None:
    eco = EconomyConfig(repair_rate=0.0)
    cfg = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000, economy=eco)
    world = World.new(cfg)
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.integrity = 50.0
    for _ in range(100):
        world.step()
    assert robot.ledger["repaired"] == 0.0
    assert robot.integrity < 50.0


def test_ledger_attributes_damage_and_rides_the_death_event() -> None:
    world = make_world()
    rid = place_robot_before_bush(world)
    robot = world.robots[rid]
    bx = int(robot.pos[0] + np.cos(robot.yaw) * 1.2)
    by = int(robot.pos[1] + np.sin(robot.yaw) * 1.2)
    world.grid.set_block(bx, by, int(robot.eye[2]), Block.BUSH_TOXIC)
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()
    assert abs(robot.ledger["poison"] - CFG.economy.toxic_integrity_damage) < 1e-9
    robot.integrity = 0.05  # ~170 dormant ticks from death
    robot.energy = 0.0
    robot.dormant = True
    world.tick = CFG.day_length_ticks * 3 // 4  # midnight: no solar rescue
    for _ in range(300):
        world.step()
    death = next(e for e in world.consume_events() if e["kind"] == "death")
    assert death["ledger"]["poison"] == CFG.economy.toxic_integrity_damage
    assert death["ledger"]["hibernation"] > 0.0


def test_hand_eaten_bush_schedules_a_replacement_sprout() -> None:
    """Eating from the gripper must not permanently delete a food site."""
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.held = int(Block.BUSH_RIPE)
    queued = len(world.sprout_heap)
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()
    assert robot.held is None
    assert len(world.sprout_heap) == queued + 1


def test_feeding_a_sleeper_schedules_a_replacement_sprout() -> None:
    world = make_world()
    feeder = world.robots[world.spawn_robot("bot_000", "test").id]
    feeder.held = int(Block.BUSH_RIPE)
    pos = feeder.pos + np.array([np.cos(feeder.yaw), np.sin(feeder.yaw), 0.0]) * 1.2
    sleeper = world.spawn_robot("bot_001", "test")
    sleeper.pos[:] = pos
    sleeper.energy = 0.0
    sleeper.dormant = True
    queued = len(world.sprout_heap)
    world.apply_action(
        feeder.id, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_PLACE)
    )
    world.step()
    assert len(world.sprout_heap) == queued + 1


def test_carried_bush_spoils_and_returns_its_slot() -> None:
    eco = EcologyConfig(held_spoil_ticks=5)
    cfg = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000, ecology=eco)
    world = World.new(cfg)
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.held = int(Block.BUSH_RIPE)
    queued = len(world.sprout_heap)
    for _ in range(6):
        world.step()
    assert robot.held is None
    assert len(world.sprout_heap) == queued + 1
    assert any(e["kind"] == "spoil" for e in world.consume_events())


def test_held_spoil_zero_disables_spoilage() -> None:
    eco = EcologyConfig(held_spoil_ticks=0)
    cfg = WorldConfig(seed=5, size=(48, 48, 40), day_length_ticks=2000, ecology=eco)
    world = World.new(cfg)
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.held = int(Block.BUSH_RIPE)
    for _ in range(200):
        world.step()
    assert robot.held == Block.BUSH_RIPE


def test_dying_with_a_bush_returns_its_slot() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.held = int(Block.BUSH_TOXIC)
    robot.integrity = 0.0001
    robot.dormant = True
    queued = len(world.sprout_heap)
    world.step()
    assert rid not in world.robots
    assert len(world.sprout_heap) == queued + 1


def test_death_leaves_a_cry_that_expires() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.integrity = 0.0001
    robot.dormant = True  # drain finishes it off this tick
    world.step()
    assert rid not in world.robots
    sounds = world.active_sounds()
    assert len(sounds) == 1
    x, y, s0, s1 = sounds[0]
    assert (s0, s1) == (-1.0, -1.0)
    for _ in range(CFG.sounds.death_cry_ticks + 1):
        world.step()
    assert world.active_sounds() == []
    assert world.transient_sounds == []  # pruned, not just filtered


def test_energy_ledger_balances_against_energy_delta() -> None:
    """The energy ledger's spend − income must equal the actual energy change
    (away from the 0/max clamps), so calibration reads are trustworthy."""
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.energy = 60.0  # headroom on both sides: no clamp during the walk
    start = robot.energy
    for _ in range(200):
        world.apply_action(rid, drive_action())
        world.step()
    spend = sum(v for k, v in robot.energy_ledger.items() if k not in ("eaten", "solar"))
    income = robot.energy_ledger["eaten"] + robot.energy_ledger["solar"]
    assert spend > 0.0
    assert abs((start - robot.energy) - (spend - income)) < 1e-6
    # movement shows up under its own cause
    assert robot.energy_ledger["move"] > 0.0
    assert robot.energy_ledger["basal"] > 0.0


def test_turn_cost_is_charged_separately_from_movement() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    robot.energy = 60.0
    turn = Action(drive=np.array([0.0, 1.0], dtype=np.float32))
    for _ in range(20):
        world.apply_action(rid, turn)
        world.step()
    assert robot.energy_ledger["turn"] == pytest.approx(20 * CFG.economy.turn_cost)
    assert robot.energy_ledger["move"] == 0.0


def test_energy_ledger_records_banked_meal_not_nominal() -> None:
    world = make_world()
    rid = place_robot_before_bush(world)
    robot = world.robots[rid]
    robot.energy = 90.0  # meal overflows the 100 cap: only 10 banked
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_EAT))
    world.step()
    assert abs(robot.energy_ledger["eaten"] - 10.0) < 0.01


def test_energy_ledger_charges_dig_and_solar() -> None:
    world = make_world()
    rid = world.spawn_robot("bot_000", "test").id
    robot = world.robots[rid]
    x, y, z = (
        int(robot.pos[0] + np.cos(robot.yaw) * 1.2),
        int(robot.pos[1] + np.sin(robot.yaw) * 1.2),
        int(robot.eye[2]),
    )
    world.grid.set_block(x, y, z, Block.SOIL)
    world.apply_action(rid, Action(drive=np.zeros(2, dtype=np.float32), gripper=GRIP_DIG))
    world.step()
    assert abs(robot.energy_ledger["dig"] - CFG.economy.dig_cost) < 1e-9
    # dormant solar income lands under "solar"
    robot.energy = 0.0
    robot.dormant = True
    world.tick = CFG.day_length_ticks // 4  # noon: light_level == 1.0
    world.step()
    assert robot.energy_ledger["solar"] > 0.0
