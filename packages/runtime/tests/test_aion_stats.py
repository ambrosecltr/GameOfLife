import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "aion_stats.py"
TEMPERAMENT_KEYS_FOR_TEST = (
    "w_curiosity",
    "w_homeostasis",
    "drive_energy",
    "drive_integrity",
    "drive_rest",
    "boredom",
    "entropy",
)


def _write_ndjson(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_aion_report_is_human_readable_and_semantically_honest(tmp_path: Path) -> None:
    save = tmp_path / "aion_test"
    checkpoint = save / "checkpoints" / "ckpt_000000001000"
    checkpoint.mkdir(parents=True)
    (save / "checkpoints" / "LATEST").write_text(checkpoint.name)
    (save / "manifest.json").write_text(
        json.dumps(
            {
                "git_commit": "abc123",
                "run_config": {
                    "tick_rate": 20,
                    "checkpoint_interval_ticks": 500,
                    "pacing": {"max_debt_updates": 4.0},
                    "population": {"inherit_weights": "lineage"},
                    "reproduction": {"mode": "respawn"},
                },
            }
        )
    )
    brain_base = {
        "pred_error_kind": 0.5,
        "curiosity": 0.2,
        "updates": 1,
        "buffer": 100,
        "learn_seconds": 0.4,
        "action_seconds": 0.02,
        "pending_update_credit": 0.5,
        "inference_lag_updates": 1,
        "act_latched_frac": 0.0,
        "policy_cont_std_mean": 0.5,
        "policy_cont_std_max": 1.1,
        "policy_action_abs_mean": 0.3,
        "policy_action_saturation_frac": 0.6,
        "policy_rest_sample_frac": 0.2,
        "wellbeing": 0.1,
        "affect_viability": 0.0,
        **{f"temperament_{key}": 1.0 for key in TEMPERAMENT_KEYS_FOR_TEST},
    }
    metrics = []
    for index in range(8):
        tick = (index + 1) * 200
        brain = {
            **brain_base,
            "loss_model": 80.0 - index * 8,
            "pred_error_depth": 0.08 - index * 0.008,
            "updates": index + 1,
        }
        metrics.append(
            {
                "tick": tick,
                "population": 2,
                "ripe_bushes": 10,
                "toxic_bushes": 2,
                "empty_bushes": 3,
                "robots": {
                    "aion_000": {
                        "brain": "aion",
                        "energy": 10.0,
                        "integrity": 80.0,
                        "age": tick,
                        "dormant": index % 2 == 0,
                        "resting": index % 2 == 0,
                        "signal": [(-1.0) ** index * 0.5, index / 10.0],
                        "signal_magnitude": max(0.5, index / 10.0),
                        "near_robots": 0,
                        "near_bushes": 1,
                    }
                },
                "brains": {"aion_000": brain},
                "runtime": {
                    "precision": "amp_bf16",
                    "actual_virtual_ticks_per_second": 20.0,
                    "safe_ticks_per_second": 30.0,
                    "total_learner_updates_per_second": 2.0,
                    "max_learner_debt": 0.5,
                    "dropped_update_credit": 0.0,
                    "inference_deadline_misses": 0.0,
                    "limiting_subsystem": "learner",
                },
            }
        )
    _write_ndjson(save / "metrics.ndjson", metrics)
    _write_ndjson(
        save / "events.ndjson",
        [
            {"tick": 0, "kind": "spawn", "robot": "aion_000", "brain": "aion"},
            {"tick": 200, "kind": "hibernate", "robot": "aion_000"},
            {"tick": 400, "kind": "wake", "robot": "aion_000"},
        ],
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(save), "--window", "1000"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "AION CHECK-IN" in result.stdout
    assert "LEARNING" in result.stdout
    assert "76.0000 -> 28.0000 (down)" in result.stdout
    assert "SURVIVAL & BODIES" in result.stdout
    assert "fixed founder" in result.stdout
    assert "not genetic selection" in result.stdout
    assert "WATCH: low current vitality for aion_000" in result.stdout
    assert "ALERT: continuous policy standard deviation exceeds 1.0" in result.stdout
    assert "WATCH: more than half of imagined continuous actions are saturated" in result.stdout
    assert "ALERT: wellbeing is logged but absent from imagined viability affect" in result.stdout
    assert "policy std=0.500/1.100 mean/max" in result.stdout
    assert "signal mean=" in result.stdout
    assert "entropy=(" in result.stdout


def test_aion_report_rejects_non_save_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)], capture_output=True, text=True
    )

    assert result.returncode == 1
    assert "is not a save directory" in result.stderr
