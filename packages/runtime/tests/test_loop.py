import dataclasses
import threading
from pathlib import Path
from typing import cast

from gol_runtime.config import DormancyAccelerationConfig, PopulationConfig, RunConfig
from gol_runtime.governor import GovernorDecision, VirtualTimeGovernor
from gol_runtime.loop import SimLoop
from gol_runtime.scheduler import Population
from gol_world import persistence
from gol_world.config import WorldConfig
from gol_world.world import World

CFG = WorldConfig(seed=3, size=(32, 32, 40), day_length_ticks=500)
RUN = RunConfig(checkpoint_interval_ticks=200)


def _make(tmp_path: Path) -> tuple[Path, SimLoop]:
    save = tmp_path / "save"
    persistence.create_save(save, CFG, run_config=dataclasses.asdict(RUN))
    world = World.new(CFG)
    return save, SimLoop(world, save, RUN)


def test_headless_run_checkpoints_and_resumes(tmp_path: Path) -> None:
    save, loop = _make(tmp_path)
    loop.run(max_ticks=500, paced=False)
    assert loop.world.tick == 500

    # Interval checkpoints at 200 and 400, final one at 500.
    ckpts = sorted(p.name for p in (save / "checkpoints").glob("ckpt_*"))
    assert ckpts == ["ckpt_000000000200", "ckpt_000000000400", "ckpt_000000000500"]

    resumed = persistence.load_world(save)
    assert resumed.tick == 500
    loop2 = SimLoop(resumed, save, RUN)
    loop2.run(max_ticks=100, paced=False)
    assert resumed.tick == 600


def test_frame_logger_called(tmp_path: Path) -> None:
    save, loop = _make(tmp_path)
    frames: list[int] = []
    loop.log_frame = lambda world: frames.append(world.tick)
    loop.run(max_ticks=50, paced=False)
    assert frames, "log_frame should be called at least once"
    assert frames[-1] == 50, "a final frame is logged on stop"


def test_stop_requested_halts_and_checkpoints(tmp_path: Path) -> None:
    save, loop = _make(tmp_path)
    loop.stop_requested = True
    loop.run(max_ticks=1000, paced=False)
    assert loop.world.tick == 0
    assert persistence.latest_checkpoint(save) is not None


def test_backpressure_services_checkpoint_metrics_and_frames(tmp_path: Path) -> None:
    class ReleasingGovernor:
        def __init__(self) -> None:
            self.decisions = 0

        def decision(self, all_dormant: bool) -> GovernorDecision:
            del all_dormant
            self.decisions += 1
            blocked = self.decisions <= 2
            return GovernorDecision(
                tick_rate=20.0,
                limiting_subsystem="learner_debt" if blocked else "configured",
                backpressured=blocked,
                reason="causal_lag" if blocked else "none",
            )

        def observe_advance(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    save, loop = _make(tmp_path)
    holds: list[int] = []
    frames: list[int] = []
    loop.governor = cast(VirtualTimeGovernor, ReleasingGovernor())
    loop.on_backpressure = lambda world: holds.append(world.tick)
    loop.log_frame = lambda world: frames.append(world.tick)
    loop.checkpoint_requested = True

    loop.run(max_ticks=1, paced=False)

    checkpoints = sorted(path.name for path in (save / "checkpoints").glob("ckpt_*"))
    assert "ckpt_000000000000" in checkpoints
    assert holds == [0]
    assert frames[0] == 0
    assert frames[-1] == 1


def test_status_snapshot_waits_for_simulation_mutation(tmp_path: Path) -> None:
    _, loop = _make(tmp_path)
    started = threading.Event()
    completed = threading.Event()

    def read_status() -> None:
        started.set()
        loop.status()
        completed.set()

    with loop._state_lock:
        reader = threading.Thread(target=read_status)
        reader.start()
        assert started.wait(1.0)
        assert not completed.wait(0.05)
    assert completed.wait(1.0)
    reader.join(timeout=1.0)
    assert not reader.is_alive()


def test_event_fast_forward_preserves_checkpoint_and_act_boundaries(tmp_path: Path) -> None:
    run = RunConfig(
        checkpoint_interval_ticks=100,
        population=PopulationConfig(
            target=1,
            mix=({"brain": {"kind": "random_walker"}, "count": 1},),
        ),
        dormancy_acceleration=DormancyAccelerationConfig(
            exact_unpaced=True,
            event_fast_forward=True,
            max_jump_ticks=1000,
        ),
    )
    save = tmp_path / "fast"
    persistence.create_save(save, CFG, run_config=dataclasses.asdict(run))
    world = World.new(CFG)
    population = Population(world, run)
    robot = next(iter(world.robots.values()))
    for _ in range(30):
        world.step()
    world.consume_events()
    robot.dormant = True
    robot.energy = 0.0
    world.step()
    world.consume_events()
    start_tick = world.tick

    loop = SimLoop(
        world,
        save,
        run,
        brain_states=population.brain_states,
        act_step=population.act_step,
        fast_forward_ready=population.can_fast_forward_dormant,
        fast_forward_opportunities=population.advance_dormant_opportunities,
        next_lifecycle_tick=population.next_lifecycle_tick,
    )
    loop.run(max_ticks=250, paced=True)

    assert world.tick == start_tick + 250
    expected_opportunities = world.tick // run.act_every - start_tick // run.act_every
    assert population._dormant_steps[robot.id] == expected_opportunities
    checkpoints = sorted(path.name for path in (save / "checkpoints").glob("ckpt_*"))
    assert f"ckpt_{world.tick:012d}" in checkpoints
