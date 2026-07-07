"""Swift speed-core tests: the optimized paths must be the same math.

Every rewrite here (batched ensemble, closed-form distributions, burn-in,
recent-slot replay, Muon, L2-init) is either bit-parity with the module it
replaced or a flagged behavior with its semantics pinned by a test.
"""

import math
from typing import Any, cast

import numpy as np
import pytest
import torch
from gol_brains.dreamer.brain import ACTION_DIM, DreamerBrain, _migrate_ensemble_state
from gol_brains.dreamer.buffer import ReplayBuffer
from gol_brains.dreamer.networks import (
    DiscreteDist,
    EnsembleMLP,
    TanhNormal,
    mlp,
    sample_categorical,
)
from gol_brains.dreamer.optim import Muon, newton_schulz
from gol_world.interface import (
    EVENTS_DIM,
    NUM_RAY_KINDS,
    PROPRIO_DIM,
    RAY_DIM,
    SOUND_DIM,
    BodySpec,
    Observation,
)

# Mirrors test_dreamer.py's TINY/fake_obs (tests are standalone modules, not a
# package, so helpers are duplicated rather than imported across files).
TINY: dict[str, Any] = {
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


def _reference_members(ens: EnsembleMLP, in_dim: int) -> list[torch.nn.Sequential]:
    """Rebuild the pre-swift ModuleList members from the stacked params."""
    members = []
    for j in range(ens.k):
        m = mlp(in_dim, ens.hidden, ens.w2.shape[-1], layers=1)
        lin1 = cast(torch.nn.Linear, m[0])
        norm = cast(torch.nn.LayerNorm, m[1])
        lin2 = cast(torch.nn.Linear, m[3])
        with torch.no_grad():
            lin1.weight.copy_(ens.w1[j].T)
            lin1.bias.copy_(ens.b1[j])
            norm.weight.copy_(ens.ln_w[j])
            norm.bias.copy_(ens.ln_b[j])
            lin2.weight.copy_(ens.w2[j].T)
            lin2.bias.copy_(ens.b2[j])
        members.append(m)
    return members


def test_ensemble_mlp_matches_module_list() -> None:
    """The batched ensemble is the ModuleList computation in 2 kernels."""
    torch.manual_seed(0)
    ens = EnsembleMLP(k=5, in_dim=12, hidden=24, out_dim=8)
    members = _reference_members(ens, in_dim=12)
    x = torch.randn(3, 7, 12)
    batched = ens(x)
    assert batched.shape == (5, 3, 7, 8)
    for j, m in enumerate(members):
        torch.testing.assert_close(batched[j], m(x), atol=1e-5, rtol=1e-5)


def test_ensemble_migration_from_module_list_checkpoint() -> None:
    """Pre-swift checkpoints (ensemble.{k}.{0,1,3}.*) load into the batched layout."""
    torch.manual_seed(1)
    ens = EnsembleMLP(k=3, in_dim=6, hidden=10, out_dim=4)
    members = _reference_members(ens, in_dim=6)
    # Build an old-style wm state: stacked keys removed, per-member keys added.
    state = {f"ensemble.{j}.{k}": v for j, m in enumerate(members)
             for k, v in m.state_dict().items()}
    _migrate_ensemble_state(state)
    assert "ensemble.0.0.weight" not in state
    fresh = EnsembleMLP(k=3, in_dim=6, hidden=10, out_dim=4)
    fresh.load_state_dict({k.removeprefix("ensemble."): v for k, v in state.items()})
    x = torch.randn(4, 6)
    torch.testing.assert_close(fresh(x), ens(x), atol=1e-6, rtol=1e-6)
    # Idempotent on already-migrated state.
    before = dict(state)
    _migrate_ensemble_state(state)
    assert state.keys() == before.keys()


def test_tanh_normal_matches_torch_distributions() -> None:
    torch.manual_seed(2)
    mean, std = torch.randn(64, 6), torch.rand(64, 6) + 0.1
    ours = TanhNormal(mean, std)
    base = torch.distributions.Normal(mean, std)
    action = torch.tanh(base.sample())  # type: ignore[no-untyped-call]
    a = action.clamp(-0.999, 0.999)
    pre = torch.atanh(a)
    ref_logp = (base.log_prob(pre) - torch.log1p(-a.pow(2) + 1e-6)).sum(-1)  # type: ignore[no-untyped-call]
    torch.testing.assert_close(ours.log_prob(action), ref_logp, atol=1e-5, rtol=1e-5)
    ref_ent = base.entropy().sum(-1)  # type: ignore[no-untyped-call]
    torch.testing.assert_close(ours.entropy(), ref_ent, atol=1e-5, rtol=1e-5)
    assert ours.sample().abs().max() <= 1.0


def test_discrete_dist_matches_torch_categorical() -> None:
    torch.manual_seed(3)
    probs = torch.softmax(torch.randn(32, 4), dim=-1) * 0.99 + 0.01 / 4
    ours = DiscreteDist(probs)
    ref = torch.distributions.Categorical(probs=probs)
    idx = ref.sample()  # type: ignore[no-untyped-call]
    ref_logp = ref.log_prob(idx)  # type: ignore[no-untyped-call]
    ref_ent = ref.entropy()  # type: ignore[no-untyped-call]
    torch.testing.assert_close(ours.log_prob(idx), ref_logp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ours.entropy(), ref_ent, atol=1e-5, rtol=1e-5)


def test_sample_categorical_frequencies_and_bounds() -> None:
    torch.manual_seed(4)
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4]).expand(20000, 4)
    idx = sample_categorical(probs)
    assert idx.dtype == torch.int64
    assert idx.min() >= 0 and idx.max() <= 3
    freq = torch.bincount(idx, minlength=4).float() / idx.numel()
    torch.testing.assert_close(freq, probs[0], atol=0.02, rtol=0.0)
    # Extreme mass never samples out of range (cumsum round-off clamp).
    hot = torch.zeros(1000, 4)
    hot[:, 3] = 1.0
    assert (sample_categorical(hot) == 3).all()


def _filled_buffer(capacity: int, fill: int) -> ReplayBuffer:
    """Buffer whose proprio[0] is the global step index (time made visible)."""
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=capacity, num_rays=4, action_dim=ACTION_DIM, seed=0)
    for i in range(fill):
        obs = fake_obs(rng)
        obs["rays"] = obs["rays"][:4]
        obs["proprio"][0] = i % 512  # exact in float16
        buf.add(obs, np.zeros(ACTION_DIM, dtype=np.float32))
    return buf


def test_recent_slot_pins_newest_experience() -> None:
    buf = _filled_buffer(capacity=1000, fill=300)
    batch = buf.sample_sequences(4, 32, recent=2)
    assert batch is not None
    t = batch["proprio"][..., 0]
    # Row 0 ends at the newest step; row 1 staggers one window back.
    assert t[0, -1] == 299 and t[0, 0] == 299 - 31
    assert t[1, -1] == 299 - 32
    np.testing.assert_array_equal(np.diff(t[:2], axis=1), 1)


def test_recent_slot_wraps_ring_seam() -> None:
    """When full, the newest window crosses the array end but stays contiguous."""
    buf = _filled_buffer(capacity=100, fill=120)  # pos=20 < window, so raw indices wrap
    batch = buf.sample_sequences(2, 32, recent=1)
    assert batch is not None
    t = batch["proprio"][0, :, 0]
    assert t[-1] == 119 and t[0] == 88
    np.testing.assert_array_equal(np.diff(t), 1)


def test_uniform_rows_never_cross_seam() -> None:
    buf = _filled_buffer(capacity=100, fill=150)
    for _ in range(20):
        batch = buf.sample_sequences(8, 16, recent=1)
        assert batch is not None
        np.testing.assert_array_equal(np.diff(batch["proprio"][..., 0], axis=1) % 512, 1)


def test_burn_in_brain_learns() -> None:
    cfg = dict(TINY, replay=dict(TINY["replay"], seq_len=8, burn_in=8, recent=1))
    brain = DreamerBrain(cfg, seed=40)
    rng = np.random.default_rng(40)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None
    assert np.isfinite(metrics["loss_model"]) and np.isfinite(metrics["loss_actor"])
    # Checkpoint roundtrip unaffected by the new replay knobs.
    fresh = DreamerBrain(cfg, seed=41)
    fresh.load_state_dict(brain.state_dict())
    assert fresh.experience_count() == brain.experience_count()


def test_l2_init_penalty_tracks_drift_from_init() -> None:
    cfg = dict(TINY, training=dict(TINY["training"], l2_init=1e-6))
    brain = DreamerBrain(cfg, seed=42)
    with torch.no_grad():
        for p, p0 in zip(brain.wm.parameters(), brain._wm_init, strict=True):
            torch.testing.assert_close(p, p0)
    rng = np.random.default_rng(42)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None and metrics["l2_init_dist"] >= 0.0
    metrics2 = brain.learn()
    assert metrics2 is not None and metrics2["l2_init_dist"] > 0.0, "training drifts from init"


def test_newton_schulz_orthogonalizes() -> None:
    torch.manual_seed(5)
    for shape in ((16, 16), (8, 24), (24, 8)):
        g = torch.randn(*shape)
        o = newton_schulz(g)
        s = torch.linalg.svdvals(o)
        # The quintic iteration lands singular values in ~[0.7, 1.2], by design.
        assert s.min() > 0.3 and s.max() < 1.5, (shape, s.min(), s.max())


def test_muon_rejects_non_2d_params() -> None:
    with pytest.raises(ValueError, match="2D"):
        Muon([torch.nn.Parameter(torch.zeros(3))])


def test_muon_brain_learns_and_checkpoints() -> None:
    cfg = dict(TINY, training=dict(TINY["training"], optimizer="muon"))
    brain = DreamerBrain(cfg, seed=43)
    assert brain.opt_model_muon is not None
    rng = np.random.default_rng(43)
    for _ in range(80):
        brain.act(fake_obs(rng))
    m1 = brain.learn()
    assert m1 is not None and np.isfinite(m1["loss_model"])
    state = brain.state_dict()
    assert state["opt_model_muon"] is not None
    fresh = DreamerBrain(cfg, seed=44)
    fresh.load_state_dict(state)
    for p1, p2 in zip(brain.wm.parameters(), fresh.wm.parameters(), strict=True):
        torch.testing.assert_close(p1, p2)
    m2 = fresh.learn()
    assert m2 is not None and np.isfinite(m2["loss_model"])


def test_unknown_optimizer_rejected() -> None:
    with pytest.raises(ValueError, match="optimizer"):
        DreamerBrain(dict(TINY, training=dict(TINY["training"], optimizer="sgd")), seed=45)


def test_entropy_constant_matches_math() -> None:
    std = torch.full((1, 3), 2.0)
    ent = TanhNormal(torch.zeros(1, 3), std).entropy()
    expected = 3 * (0.5 * math.log(2 * math.pi * math.e * 4.0))
    torch.testing.assert_close(ent, torch.tensor([expected]), atol=1e-5, rtol=1e-5)


@pytest.mark.slow
def test_compiled_brain_acts_and_learns() -> None:
    cfg = dict(TINY, training=dict(TINY["training"], compile=True))
    brain = DreamerBrain(cfg, seed=46)
    rng = np.random.default_rng(46)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None and np.isfinite(metrics["loss_model"])
