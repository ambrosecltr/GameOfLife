"""gol-stats interest profiles: individuality and stability from the logs."""

import json
from pathlib import Path
from typing import Any

from gol_runtime.inspect import circadian, interests


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_interest_profiles_divergence_and_stability(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    # Two agents with steady, opposite habits: one social, one foraging.
    metrics = [
        {
            "tick": tick,
            "robots": {
                "social_0": {
                    "resting": False,
                    "near_robots": 2,
                    "near_bushes": 0,
                    "dormant": False,
                },
                "forager_1": {
                    "resting": True,
                    "near_robots": 0,
                    "near_bushes": 3,
                    "dormant": False,
                },
            },
        }
        for tick in range(0, 2000, 100)
    ]
    _write(tmp_path / "metrics.ndjson", metrics)
    _write(
        tmp_path / "events.ndjson",
        [
            {"tick": 0, "kind": "spawn", "robot": "social_0", "brain": "dreamer"},
            {"tick": 0, "kind": "spawn", "robot": "forager_1", "brain": "dreamer"},
            {"tick": 150, "kind": "eat", "robot": "forager_1"},
            {"tick": 1150, "kind": "eat", "robot": "forager_1"},
        ],
    )

    out = interests(tmp_path, window_ticks=1000)
    pa = out["per_agent"]
    assert pa["social_0"]["profile"]["social"] == 1.0
    assert pa["social_0"]["profile"]["forage"] == 0.0
    assert pa["forager_1"]["profile"]["forage"] == 1.0
    assert pa["forager_1"]["profile"]["eat"] == 0.1  # 1 eat per 10 samples
    assert pa["forager_1"]["brain"] == "dreamer"
    # Habits repeat across windows: perfect within-agent stability...
    assert pa["social_0"]["stability"] == 1.0
    assert pa["forager_1"]["stability"] == 1.0
    # ...while the agents differ from each other in every window.
    assert len(out["divergence"]) == 2
    assert all(d["divergence"] > 0 for d in out["divergence"])
    assert all(d["agents"] == 2 for d in out["divergence"])


def test_interests_tolerates_old_saves(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    # A save from before interest logging: robots lack the near_* fields.
    _write(
        tmp_path / "metrics.ndjson",
        [{"tick": 0, "robots": {"walker_0": {"energy": 50.0, "dormant": False}}}],
    )
    _write(tmp_path / "events.ndjson", [])
    out = interests(tmp_path, window_ticks=1000)
    assert out["per_agent"] == {} and out["divergence"] == []


def test_circadian_splits_chosen_rest_from_forced_dormancy(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    # Alternating day/night samples: one agent rests only at night while
    # awake; another hibernates through the day.
    metrics = []
    for i, tick in enumerate(range(0, 4000, 100)):
        light = 1.0 if i % 2 == 0 else 0.0
        metrics.append(
            {
                "tick": tick,
                "light": light,
                "robots": {
                    "night_rester": {
                        "brain": "dreamer",
                        "resting": light <= 0.5,
                        "dormant": False,
                    },
                    "day_sleeper": {
                        "brain": "scripted_forager",
                        "resting": False,
                        "dormant": light > 0.5,
                    },
                },
            }
        )
    _write(tmp_path / "metrics.ndjson", metrics)

    out = circadian(tmp_path)
    rester = out["awake_resting"]["dreamer"]
    assert rester["corr_with_light"] < -0.99  # sleeps at night: the H2 signature
    assert rester["night_frac"] == 1.0
    assert rester["day_frac"] == 0.0
    sleeper = out["dormant"]["scripted_forager"]
    assert sleeper["corr_with_light"] > 0.99
    # --since trims the early samples.
    trimmed = circadian(tmp_path, since_tick=2000)
    assert trimmed["dormant"]["scripted_forager"]["samples"] == 20


def test_circadian_tolerates_old_saves(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    # Pre-rest-logging save: robots carry no resting field.
    _write(
        tmp_path / "metrics.ndjson",
        [{"tick": 0, "light": 1.0, "robots": {"walker_0": {"energy": 50.0}}}],
    )
    out = circadian(tmp_path)
    assert out["awake_resting"] == {} and out["dormant"] == {}
