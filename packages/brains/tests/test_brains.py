"""Contract test: every registered brain honors the Brain interface."""

import numpy as np
import pytest
from gol_brains.base import Brain
from gol_brains.registry import build_brain
from gol_world.interface import (
    EVENTS_DIM,
    NUM_RAY_CLASSES,
    PROPRIO_DIM,
    SOUND_DIM,
    Action,
    BodySpec,
    Observation,
)

AVAILABLE_KINDS = ["random_walker"]  # extended at M2 (forager) and M3 (dreamer)


def fake_obs(rng: np.random.Generator) -> Observation:
    body = BodySpec()
    rays = np.zeros((body.num_rays, 1 + NUM_RAY_CLASSES), dtype=np.float32)
    rays[:, 0] = rng.random(body.num_rays)
    rays[np.arange(body.num_rays), 1 + rng.integers(0, NUM_RAY_CLASSES, body.num_rays)] = 1.0
    return Observation(
        rays=rays,
        proprio=rng.random(PROPRIO_DIM).astype(np.float32),
        sound=rng.random(SOUND_DIM).astype(np.float32),
        events=np.zeros(EVENTS_DIM, dtype=np.float32),
    )


@pytest.mark.parametrize("kind", AVAILABLE_KINDS)
def test_brain_contract(kind: str) -> None:
    brain = build_brain({"kind": kind}, seed=123)
    assert isinstance(brain, Brain)
    rng = np.random.default_rng(0)
    for _ in range(20):
        action = brain.act(fake_obs(rng))
        assert isinstance(action, Action)
        assert action.drive.shape == (2,)
        assert np.abs(action.drive).max() <= 1.0
        assert 0 <= action.gripper <= 3
    assert brain.learn() is None or isinstance(brain.learn(), dict)
    assert isinstance(brain.introspect(), dict)


@pytest.mark.parametrize("kind", AVAILABLE_KINDS)
def test_brain_state_roundtrip_is_deterministic(kind: str) -> None:
    rng = np.random.default_rng(1)
    obs_stream = [fake_obs(rng) for _ in range(30)]

    a = build_brain({"kind": kind}, seed=7)
    for o in obs_stream[:10]:
        a.act(o)
    saved = a.state_dict()

    b = build_brain({"kind": kind}, seed=7)
    b.load_state_dict(saved)
    for o in obs_stream[10:]:
        act_a = a.act(o)
        act_b = b.act(o)
        np.testing.assert_array_equal(act_a.drive, act_b.drive)
        assert act_a.gripper == act_b.gripper


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError, match="unknown brain kind"):
        build_brain({"kind": "psychic"}, seed=0)
