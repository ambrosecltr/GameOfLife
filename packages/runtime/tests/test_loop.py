import dataclasses
from pathlib import Path

from gol_runtime.config import RunConfig
from gol_runtime.loop import SimLoop
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
