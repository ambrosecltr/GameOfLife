from pathlib import Path

import pytest
from gol_runtime.config import apply_overrides, load_run_config

REPO = Path(__file__).resolve().parents[3]


def test_load_local_m1_config() -> None:
    run_cfg, world_cfg = load_run_config(REPO / "configs/run/local_m1.yaml")
    assert run_cfg.tick_rate == 20
    assert run_cfg.population.target == 8
    assert len(run_cfg.population.mix) == 2
    assert world_cfg.size == (256, 256, 64)
    assert world_cfg.economy.eat_energy == 40.0


def test_overrides_reach_both_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
    run_cfg, world_cfg = load_run_config(
        "configs/run/local_m1.yaml",
        sets=["tick_rate=40", "population.target=2", "world.seed=99", "world.terrain.octaves=2"],
    )
    assert run_cfg.tick_rate == 40
    assert run_cfg.population.target == 2
    assert world_cfg.seed == 99
    assert world_cfg.terrain.octaves == 2


def test_unknown_key_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        load_run_config(REPO / "configs/run/local_m1.yaml", sets=["tick_rte=40"])


def test_apply_overrides_parses_yaml_values() -> None:
    data = apply_overrides({}, ["a.b=true", "a.c=1.5", "d=[1, 2]"])
    assert data == {"a": {"b": True, "c": 1.5}, "d": [1, 2]}
