"""DreamerV3 component tests: shapes, invariants, and a tiny convergence probe."""

import numpy as np
import pytest
import torch
from gol_brains.dreamer.brain import ACTION_DIM, CONT_DIM, DreamerBrain
from gol_brains.dreamer.networks import TwoHot, symexp, symlog
from gol_brains.dreamer.rssm import RSSM, RSSMConfig
from gol_world.interface import (
    EVENTS_DIM,
    NUM_RAY_KINDS,
    PROPRIO_DIM,
    RAY_DIM,
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
    rays = np.zeros((body.num_rays, RAY_DIM), dtype=np.float32)
    rays[:, 0] = np.clip(rng.random(body.num_rays) + bias, 0, 1)
    rays[:, 1:4] = rng.random((body.num_rays, 3)).astype(np.float32)
    rays[np.arange(body.num_rays), 4 + rng.integers(0, NUM_RAY_KINDS, body.num_rays)] = 1.0
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
    assert ACTION_DIM == CONT_DIM + 4  # drive+signal+gaze, then gripper one-hot
    brain = DreamerBrain(TINY, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = brain.act(fake_obs(rng))
        assert isinstance(action, Action)
        assert action.drive.shape == (2,) and np.abs(action.drive).max() <= 1.0
        assert action.gaze is not None and action.gaze.shape == (2,)
        assert np.abs(action.gaze).max() <= 1.0
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


def test_curiosity_masking_erases_agents() -> None:
    cfg = dict(TINY, reward={"curiosity_mask_agents": True})
    brain = DreamerBrain(cfg, seed=6)
    from gol_world.blocks import SKY_DAY
    from gol_world.interface import RAY_KIND_BLOCK, RAY_KIND_NOTHING, RAY_KIND_ROBOT

    body = BodySpec()
    n = body.num_rays
    depth = torch.rand(2, 3, n)
    rgb = torch.rand(2, 3, n, 3)
    onehot = torch.zeros(2, 3, n, NUM_RAY_KINDS)
    onehot[..., RAY_KIND_ROBOT] = 1.0  # every ray sees a robot
    proprio = torch.rand(2, 3, PROPRIO_DIM)
    proprio[..., 13] = 1.0  # full daylight
    obs = {
        "depth": depth,
        "rgb": rgb,
        "kind_onehot": onehot,
        "proprio": proprio,
        "sound": torch.rand(2, 3, SOUND_DIM),
        "events": torch.zeros(2, 3, EVENTS_DIM),
    }
    masked = brain._mask_agents(obs)
    assert (masked["kind_onehot"][..., RAY_KIND_ROBOT] == 0).all()
    assert (masked["kind_onehot"][..., RAY_KIND_NOTHING] == 1).all()
    assert (masked["depth"] == 1.0).all()
    # Masked rays wear the daytime sky, like a real miss would.
    sky = torch.as_tensor(SKY_DAY)
    torch.testing.assert_close(masked["rgb"], sky.expand_as(masked["rgb"]), atol=1e-5, rtol=0)
    # Non-agent rays pass through untouched.
    onehot2 = torch.zeros(2, 3, n, NUM_RAY_KINDS)
    onehot2[..., RAY_KIND_BLOCK] = 1.0
    obs2 = dict(obs, kind_onehot=onehot2)
    masked2 = brain._mask_agents(obs2)
    torch.testing.assert_close(masked2["depth"], depth)
    torch.testing.assert_close(masked2["rgb"], rgb)


def test_masked_brain_learns() -> None:
    cfg = dict(TINY, reward={"curiosity_mask_agents": True})
    brain = DreamerBrain(cfg, seed=7)
    rng = np.random.default_rng(4)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None and np.isfinite(metrics["loss_model"])
