from pathlib import Path

import numpy as np
from gol_world import persistence
from gol_world.config import WorldConfig
from gol_world.world import World

CFG = WorldConfig(seed=9, size=(48, 48, 40), day_length_ticks=1000)


def _fresh_save(tmp_path: Path) -> tuple[Path, World]:
    save = tmp_path / "save"
    persistence.create_save(save, CFG, run_config={"note": "test"})
    return save, World.new(CFG)


def test_checkpoint_roundtrip_is_exact(tmp_path: Path) -> None:
    save, world = _fresh_save(tmp_path)
    for _ in range(500):
        world.step()
    persistence.save_checkpoint(save, world, brain_states={})

    loaded = persistence.load_world(save)
    assert loaded.tick == world.tick
    assert np.array_equal(loaded.grid.blocks, world.grid.blocks)
    assert sorted(loaded.regrow_heap) == sorted(world.regrow_heap)
    assert loaded.rng.bit_generator.state == world.rng.bit_generator.state

    # And the resumed world evolves identically to the original.
    for _ in range(1000):
        world.step()
        loaded.step()
    assert np.array_equal(loaded.grid.blocks, world.grid.blocks)
    assert loaded.tick == world.tick


def test_latest_and_pruning(tmp_path: Path) -> None:
    save, world = _fresh_save(tmp_path)
    for _ in range(5):
        for _ in range(10):
            world.step()
        persistence.save_checkpoint(save, world, brain_states={})
    ckpts = sorted((save / "checkpoints").glob("ckpt_*"))
    assert len(ckpts) == persistence.KEEP_CHECKPOINTS
    latest = persistence.latest_checkpoint(save)
    assert latest is not None and latest.name == f"ckpt_{world.tick:012d}"


def test_transient_sounds_roundtrip(tmp_path: Path) -> None:
    save, world = _fresh_save(tmp_path)
    world.spawn_robot("bot_000", "test")  # entities.json must exist for sounds to load
    world.transient_sounds = [(8.5, 9.5, -1.0, -1.0, world.tick + 40)]
    persistence.save_checkpoint(save, world, brain_states={})
    loaded = persistence.load_world(save)
    assert loaded.transient_sounds == world.transient_sounds


def test_brain_states_roundtrip(tmp_path: Path) -> None:
    save, world = _fresh_save(tmp_path)
    blobs = {"agent_001": b"weights-1", "agent_002": b"weights-2"}
    ckpt = persistence.save_checkpoint(save, world, brain_states=blobs)
    assert persistence.load_brain_states(ckpt) == blobs


def test_create_save_refuses_overwrite(tmp_path: Path) -> None:
    save, _ = _fresh_save(tmp_path)
    try:
        persistence.create_save(save, CFG, run_config={})
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
