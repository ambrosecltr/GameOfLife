from pathlib import Path

import pytest
import yaml
from gol_runtime.config import PopulationConfig, apply_overrides, load_run_config

REPO = Path(__file__).resolve().parents[3]


def test_load_local_m1_config() -> None:
    run_cfg, world_cfg = load_run_config(REPO / "configs/run/local_m1.yaml")
    assert run_cfg.tick_rate == 20
    assert run_cfg.population.target == 8
    assert len(run_cfg.population.mix) == 3
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


def test_learning_devices_accept_explicit_multi_gpu_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO)
    run_cfg, _ = load_run_config(
        "configs/run/local_m1.yaml",
        sets=["devices.learning=[cuda:0, cuda:1]"],
    )
    assert run_cfg.devices.learning_devices() == ("cuda:0", "cuda:1")


def test_aion_two_gpu_round_has_one_learning_brain_per_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO)
    run_cfg, _ = load_run_config("configs/run/aion_01_2gpu.yaml")

    assert run_cfg.devices.learning_devices() == ("cuda:0", "cuda:1")
    assert run_cfg.population.target == 6
    assert run_cfg.population.mix[0]["count"] == 2
    assert run_cfg.population.mix[1]["count"] == 4


@pytest.mark.parametrize("round_number", ["02", "03"])
def test_felt_economy_round_starts_fresh_organisms_with_descendant_inheritance(
    monkeypatch: pytest.MonkeyPatch, round_number: str
) -> None:
    monkeypatch.chdir(REPO)
    run_cfg, _ = load_run_config(f"configs/run/aion_{round_number}_economy.yaml")

    assert run_cfg.population.inherit_weights == "descendant"
    assert (
        run_cfg.population.mix[0]["brain"]
        == f"configs/brain/aion_{round_number}_economy.yaml"
    )


def test_aion_03_preserves_aion_02_scientific_configuration() -> None:
    with (REPO / "configs/brain/aion_02_economy.yaml").open() as file:
        aion_02 = yaml.safe_load(file)
    with (REPO / "configs/brain/aion_03_economy.yaml").open() as file:
        aion_03 = yaml.safe_load(file)

    assert aion_03 == aion_02


def test_unknown_inheritance_mode_fails_loudly() -> None:
    with pytest.raises(ValueError, match="inherit_weights"):
        PopulationConfig(inherit_weights="reincarnate")
