"""DreamerV3 component tests: shapes, invariants, and a tiny convergence probe."""

import numpy as np
import pytest
import torch
from gol_brains.dreamer.brain import ACTION_DIM, DreamerBrain
from gol_brains.dreamer.networks import TwoHot, symexp, symlog
from gol_brains.dreamer.rssm import RSSM, RSSMConfig
from gol_world.interface import (
    EVENTS_DIM,
    NUM_RAY_CLASSES,
    PROPRIO_DIM,
    SOUND_DIM,
    Action,
    BodySpec,
    Observation,
)

TINY = {
    "kind": "dreamer",
    "preset": "nano",
    "replay": {"capacity": 3000, "batch_size": 4, "seq_len": 16, "warmup_steps": 64},
    "training": {"imag_starts": 32},
    "actor_critic": {"imagination_horizon": 5},
}


def fake_obs(rng: np.random.Generator, bias: float = 0.0) -> Observation:
    body = BodySpec()
    rays = np.zeros((body.num_rays, 1 + NUM_RAY_CLASSES), dtype=np.float32)
    rays[:, 0] = np.clip(rng.random(body.num_rays) + bias, 0, 1)
    rays[np.arange(body.num_rays), 1 + rng.integers(0, NUM_RAY_CLASSES, body.num_rays)] = 1.0
    proprio = rng.random(PROPRIO_DIM).astype(np.float32)
    return Observation(
        rays=rays,
        proprio=proprio,
        sound=rng.random(SOUND_DIM).astype(np.float32),
        events=np.zeros(EVENTS_DIM, dtype=np.float32),
    )


def test_symlog_symexp_inverse() -> None:
    x = torch.linspace(-100, 100, 41)
    torch.testing.assert_close(symexp(symlog(x)), x, atol=1e-3, rtol=1e-4)


def test_twohot_roundtrip() -> None:
    th = TwoHot()
    values = torch.tensor([-50.0, -1.0, 0.0, 0.5, 3.0, 100.0])
    target = th.encode(values)
    assert target.shape == (6, 41)
    torch.testing.assert_close(target.sum(-1), torch.ones(6))
    # softmax(log(p)) == p, so decoding the target's log recovers the value
    # (up to symlog interpolation error, which grows with magnitude).
    recovered = th.decode(target.clamp(min=1e-30).log())
    torch.testing.assert_close(recovered, values, atol=0.05, rtol=0.15)


def test_rssm_shapes_and_kl() -> None:
    cfg = RSSMConfig(deter=32, stoch_groups=4, stoch_classes=4, hidden=32)
    rssm = RSSM(cfg, embed_dim=16, action_dim=ACTION_DIM)
    h, z = rssm.initial(3, torch.device("cpu"))
    a = torch.zeros(3, ACTION_DIM)
    e = torch.randn(3, 16)
    h, z, post, prior = rssm.obs_step(h, z, a, e)
    assert h.shape == (3, 32) and z.shape == (3, 16)
    assert post.shape == (3, 4, 4) and prior.shape == (3, 4, 4)
    # One-hot-ish samples: each group sums to ~1.
    torch.testing.assert_close(z.view(3, 4, 4).sum(-1), torch.ones(3, 4))
    kl = rssm.kl_loss(post, prior)
    assert kl.shape == (3,)
    assert (kl >= cfg.free_bits * (cfg.dyn_scale + cfg.rep_scale) - 1e-5).all()
    h2, z2, pp = rssm.img_step(h, z, a)
    assert h2.shape == h.shape and z2.shape == z.shape and pp.shape == post.shape


def test_ensemble_disagreement_positive_on_random() -> None:
    brain = DreamerBrain(TINY, seed=0)
    feat = torch.randn(10, brain.wm.rssm_cfg.feat_dim)
    action = torch.randn(10, ACTION_DIM)
    d = brain.wm.disagreement(feat, action)
    assert d.shape == (10,)
    assert (d > 0).all(), "freshly initialized ensemble must disagree"


def test_act_contract_and_warmup() -> None:
    brain = DreamerBrain(TINY, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = brain.act(fake_obs(rng))
        assert isinstance(action, Action)
        assert action.drive.shape == (2,) and np.abs(action.drive).max() <= 1.0
        assert 0 <= action.gripper < 4
    assert len(brain.buffer) == 10
    assert brain.learn() is None, "no learning before warmup"


def test_learn_returns_metrics_and_updates() -> None:
    brain = DreamerBrain(TINY, seed=2)
    rng = np.random.default_rng(1)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None
    for key in ("loss_model", "kl", "curiosity", "loss_critic", "loss_actor"):
        assert key in metrics and np.isfinite(metrics[key])


@pytest.mark.slow
def test_overfits_a_repeating_sequence() -> None:
    """The world model must be able to memorize a tiny deterministic stream."""
    brain = DreamerBrain(TINY, seed=3)
    rng = np.random.default_rng(2)
    pattern = [fake_obs(rng, bias=b) for b in (0.0, 0.3, 0.6, 0.9)] * 60
    for obs in pattern:
        brain.act(obs)
    first = 0.0
    last = 0.0
    for i in range(60):
        metrics = brain.learn()
        assert metrics is not None
        last = metrics["loss_model"]
        if i == 0:
            first = last
    assert last < first * 0.7, f"model loss should drop while overfitting: {first} -> {last}"


def test_state_dict_roundtrip() -> None:
    brain = DreamerBrain(TINY, seed=4)
    rng = np.random.default_rng(3)
    for _ in range(80):
        brain.act(fake_obs(rng))
    brain.learn()
    state = brain.state_dict()

    fresh = DreamerBrain(TINY, seed=99)
    fresh.load_state_dict(state)
    assert len(fresh.buffer) == len(brain.buffer)
    for p1, p2 in zip(brain.wm.parameters(), fresh.wm.parameters(), strict=True):
        torch.testing.assert_close(p1, p2)
    obs = fake_obs(np.random.default_rng(5))
    brain.act(obs)
    fresh.act(obs)
    # Same weights + same restored recurrent state: posteriors evolve identically.
    torch.testing.assert_close(brain.h, fresh.h)


def test_obs_version_mismatch_rejected() -> None:
    brain = DreamerBrain(TINY, seed=5)
    state = brain.state_dict()
    state["obs_version"] = 999
    with pytest.raises(ValueError, match="obs_version"):
        brain.load_state_dict(state)
