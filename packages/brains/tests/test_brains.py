"""Contract test: every registered brain honors the Brain interface."""

import numpy as np
import pytest
from gol_brains.base import Brain
from gol_brains.registry import build_brain
from gol_world.interface import (
    EVENTS_DIM,
    NUM_RAY_KINDS,
    PROPRIO_DIM,
    RAY_DIM,
    RAY_KIND_BLOCK,
    RAY_KIND_NOTHING,
    SOUND_DIM,
    Action,
    BodySpec,
    Observation,
)

AVAILABLE_KINDS = ["random_walker", "scripted_forager", "dreamer"]


def fake_obs(rng: np.random.Generator) -> Observation:
    body = BodySpec()
    rays = np.zeros((body.num_rays, RAY_DIM), dtype=np.float32)
    rays[:, 0] = rng.random(body.num_rays)
    rays[:, 1:4] = rng.random((body.num_rays, 3)).astype(np.float32)
    rays[np.arange(body.num_rays), 4 + rng.integers(0, NUM_RAY_KINDS, body.num_rays)] = 1.0
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


def test_body_from_config_overrides() -> None:
    brain = build_brain(
        {"kind": "scripted_forager", "body": {"rays_per_row": 8, "ray_pitches_deg": [0.0]}},
        seed=0,
    )
    assert brain.body.num_rays == 8  # type: ignore[attr-defined]


def _shaded(block_color: np.ndarray, factor: float) -> np.ndarray:
    """A block color as sensing would deliver it: luminance-scaled."""
    return (block_color.astype(np.float32) / 255.0) * factor


def bush_in_reach_obs(body: BodySpec) -> Observation:
    """Daylight, low energy, a ripe bush dead ahead within reach — eat bait.

    Colors arrive shaded and grained like the real sensor: the forager must
    recognize the berry by chroma, not by exact palette value.
    """
    from gol_world.blocks import COLOR, Block

    rays = np.zeros((body.num_rays, RAY_DIM), dtype=np.float32)
    rays[:, 0] = 1.0
    rays[:, 4 + RAY_KIND_NOTHING] = 1.0  # everything else: no hit (sky)
    center = body.rays_per_row // 2  # azimuth a few degrees off-axis: inside the eat cone
    rays[center] = 0.0
    rays[center, 0] = 1.0 / body.ray_range  # 1 block away, inside reach
    rays[center, 1:4] = _shaded(COLOR[Block.BUSH_RIPE], 0.8 * 1.07)
    rays[center, 4 + RAY_KIND_BLOCK] = 1.0
    proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
    proprio[5] = 0.3  # hungry
    proprio[13] = 1.0  # daylight
    return Observation(
        rays=rays,
        proprio=proprio,
        sound=np.zeros(SOUND_DIM, dtype=np.float32),
        events=np.zeros(EVENTS_DIM, dtype=np.float32),
    )


def test_forager_breaks_out_of_a_failing_eat_loop() -> None:
    """If GRIP_EAT never lands (no ate event), the probe must not wedge."""
    brain = build_brain({"kind": "scripted_forager"}, seed=3)
    obs = bush_in_reach_obs(BodySpec())
    actions = [brain.act(obs) for _ in range(12)]
    assert actions[0].gripper == 3  # it does try to eat first
    assert any(a.drive[0] < 0.0 for a in actions)  # ...then backs off the bush
