"""anima (plastic-valence) brain: contract, plasticity, and inheritance."""

from __future__ import annotations

from typing import Any

import numpy as np
from gol_brains.plastic.brain import PlasticBrain
from gol_world.interface import (
    EVENTS_DIM,
    NUM_GRIP_MODES,
    NUM_RAY_KINDS,
    PROPRIO_DIM,
    RAY_DIM,
    SOUND_DIM,
    BodySpec,
    Observation,
)

CFG: dict[str, Any] = {
    "kind": "plastic",
    "core": {"hidden": 64},
    "plasticity": {"alpha": 0.2, "tau": 10.0, "decay": 1.0e-3, "w_clip": 2.0},
    "restlessness": 0.2,
    "genome": {"enabled": True, "sigma": 0.25, "mutation_sigma": 0.1},
}


def fake_obs(rng: np.random.Generator) -> Observation:
    body = BodySpec()
    rays = np.zeros((body.num_rays, RAY_DIM), dtype=np.float32)
    rays[:, 0] = rng.random(body.num_rays).astype(np.float32)
    rays[:, 1:4] = rng.random((body.num_rays, 3)).astype(np.float32)
    rays[np.arange(body.num_rays), 4 + rng.integers(0, NUM_RAY_KINDS, body.num_rays)] = 1.0
    return Observation(
        rays=rays,
        proprio=rng.random(PROPRIO_DIM).astype(np.float32),
        sound=rng.random(SOUND_DIM).astype(np.float32),
        events=np.zeros(EVENTS_DIM, dtype=np.float32),
    )


def test_action_contract() -> None:
    rng = np.random.default_rng(0)
    brain = PlasticBrain(CFG, seed=0)
    for _ in range(50):
        a = brain.act(fake_obs(rng))
        assert a.drive.shape == (2,)
        assert np.all(np.abs(a.drive) <= 1.0 + 1e-5)
        assert 0 <= a.gripper < NUM_GRIP_MODES
        assert a.signal is not None and a.gaze is not None
        assert np.all(np.isfinite(a.drive)) and np.all(np.isfinite(a.signal))


def test_plasticity_changes_fast_weights() -> None:
    rng = np.random.default_rng(1)
    brain = PlasticBrain(CFG, seed=1)
    assert brain.net.fast_norm() == 0.0  # W_fast starts at zero
    for _ in range(200):
        brain.act(fake_obs(rng))
    assert brain.net.fast_norm() > 0.0  # nonzero M under drive deltas moved them
    assert np.isfinite(brain.net.fast_norm())


def test_frozen_control_never_learns() -> None:
    rng = np.random.default_rng(2)
    frozen_cfg = {**CFG, "plasticity": {**CFG["plasticity"], "enabled": False}}
    brain = PlasticBrain(frozen_cfg, seed=2)
    for _ in range(200):
        brain.act(fake_obs(rng))
    assert brain.net.fast_norm() == 0.0  # alpha:0 ⇒ pure evolved reflex, W_fast frozen


def test_checkpoint_roundtrip() -> None:
    rng = np.random.default_rng(3)
    a = PlasticBrain(CFG, seed=3)
    for _ in range(100):
        a.act(fake_obs(rng))
    state = a.state_dict()

    b = PlasticBrain(CFG, seed=99)  # different seed, then restore
    b.load_state_dict(state)
    # identical obs stream ⇒ identical actions after restore
    probe = np.random.default_rng(7)
    obs_seq = [fake_obs(probe) for _ in range(20)]
    acts_a = [a.act(o) for o in obs_seq]
    acts_b = [b.act(o) for o in obs_seq]
    for xa, xb in zip(acts_a, acts_b, strict=True):
        assert np.allclose(xa.drive, xb.drive)
        assert xa.gripper == xb.gripper


def obs_with_body(rng: np.random.Generator, energy: float, integrity: float = 1.0) -> Observation:
    """A controlled-interoception obs: proprio zeros except the body channels."""
    obs = fake_obs(rng)
    proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
    proprio[5] = energy  # feeling.ENERGY_IDX
    proprio[6] = integrity  # feeling.INTEGRITY_IDX
    obs["proprio"] = proprio
    return obs


def test_appetite_raises_arousal_when_hungry() -> None:
    rng = np.random.default_rng(6)
    cfg = {**CFG, "appetite_gain": 2.0, "genome": {**CFG["genome"], "enabled": False}}
    brain = PlasticBrain(cfg, seed=6)
    brain.act(obs_with_body(rng, energy=0.9))
    sated = brain.introspect()["arousal"]
    brain.act(obs_with_body(rng, energy=0.3))
    hungry = brain.introspect()["arousal"]
    assert hungry > sated  # hunger raises search effort
    assert abs(sated - brain.restlessness) < 0.05  # near-setpoint ≈ base restlessness
    assert hungry <= 0.6  # arousal stays clipped

    # appetite_gain 0 (anima_01/02 configs) leaves arousal == restlessness exactly.
    off = PlasticBrain({**CFG, "genome": {**CFG["genome"], "enabled": False}}, seed=6)
    off.act(obs_with_body(rng, energy=0.3))
    assert off.introspect()["arousal"] == off.restlessness


def test_centered_valence_prediction_error() -> None:
    """anima_07: M is a PREDICTION ERROR — the felt level minus a running
    baseline. The first step of a stream seeds the baseline (so M ≈ 0), and
    afterwards M is signed by whether the body is better or worse than its recent
    normal, not by an absolute setpoint: getting hungrier than baseline → −,
    recovering above baseline → +."""

    def cfg(**valence: Any) -> dict[str, Any]:
        return {
            **CFG,
            "valence": {"viability": {"standing_gain": 0.5}, **valence},
            "genome": {**CFG["genome"], "enabled": False},
        }

    # First step of a stream seeds the baseline to itself → M ≈ 0 (nothing to
    # compare against yet), regardless of how hungry the body is.
    b0 = PlasticBrain(cfg(), seed=8)
    b0.act(obs_with_body(rng := np.random.default_rng(8), energy=0.05))
    assert abs(b0.introspect()["m_comfort"]) < 1e-6
    assert abs(b0.introspect()["m_viability"]) < 1e-6

    # Fed baseline, then a drop into hunger/danger → worse than normal → M < 0.
    fell = PlasticBrain(cfg(), seed=8)
    fell.act(obs_with_body(rng, energy=0.95))  # seed baseline high/safe
    fell.act(obs_with_body(rng, energy=0.05))
    assert fell.introspect()["m_comfort"] < 0.0
    assert fell.introspect()["m_viability"] < 0.0  # more danger than baseline

    # Hungry/endangered baseline, then RECOVERY → better than normal → M > 0.
    # (The anima_06 standing tax would have stayed negative here; centering lets
    # escaping danger read positive.)
    rose = PlasticBrain(cfg(), seed=8)
    rose.act(obs_with_body(rng, energy=0.05))  # seed baseline low/in-danger
    rose.act(obs_with_body(rng, energy=0.95))
    assert rose.introspect()["m_comfort"] > 0.0
    assert rose.introspect()["m_viability"] > 0.0  # less danger than baseline

    # standing_gain 0 (default) leaves viability out of M entirely.
    off = PlasticBrain({**CFG, "genome": {**CFG["genome"], "enabled": False}}, seed=8)
    off.act(obs_with_body(rng, energy=0.05))
    off.act(obs_with_body(rng, energy=0.95))
    assert off.introspect()["m_viability"] == 0.0


def test_load_fills_genes_added_after_checkpoint() -> None:
    rng = np.random.default_rng(9)
    a = PlasticBrain(CFG, seed=9)
    for _ in range(20):
        a.act(fake_obs(rng))
    state = a.state_dict()
    del state["genes"]["appetite_gain"]  # simulate a pre-appetite checkpoint

    b = PlasticBrain(CFG, seed=10)
    b.load_state_dict(state)
    assert b.genes["appetite_gain"] == 1.0  # neutral multiplier
    for _ in range(5):
        b.act(fake_obs(rng))


def test_inherit_mutates_and_resets_fast() -> None:
    rng = np.random.default_rng(4)
    parent = PlasticBrain(CFG, seed=4)
    for _ in range(100):
        parent.act(fake_obs(rng))
    donor = parent.state_dict()

    child = PlasticBrain(CFG, seed=5)
    child.inherit(donor)  # Darwinian default: reset W_fast, mutate genome + wiring
    assert child.net.fast_norm() == 0.0
    # genome drifted off the parent's
    assert any(
        abs(child.genes[k] - parent.genes[k]) > 1e-9 for k in child.genes
    )
    # and the child still runs
    for _ in range(20):
        child.act(fake_obs(rng))
