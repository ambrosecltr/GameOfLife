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
    POLICY_MAX_STD,
    POLICY_MIN_STD,
    DiscreteDist,
    EnsembleMLP,
    TanhNormal,
    bounded_policy_std,
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


def test_policy_standard_deviation_is_smoothly_bounded() -> None:
    raw = torch.tensor([-100.0, 0.0, 100.0])
    std = bounded_policy_std(raw)
    assert float(std.min()) >= POLICY_MIN_STD
    assert float(std.max()) <= POLICY_MAX_STD
    assert std[1] == pytest.approx((POLICY_MIN_STD + POLICY_MAX_STD) / 2)


def test_reinforce_sample_is_detached_but_log_prob_still_trains_policy() -> None:
    torch.manual_seed(21)
    mean = torch.tensor([[0.2]], requires_grad=True)
    std = torch.tensor([[0.5]], requires_grad=True)
    dist = TanhNormal(mean, std)
    action = dist.sample_for_reinforce()
    assert not action.requires_grad

    (-dist.log_prob(action).sum()).backward()  # type: ignore[no-untyped-call]

    assert mean.grad is not None and float(mean.grad.abs().sum()) > 0.0
    assert std.grad is not None and float(std.grad.abs().sum()) > 0.0


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


def _spiked_buffer(capacity: int, fill: int, spike_steps: set[int]) -> ReplayBuffer:
    rng = np.random.default_rng(1)
    buf = ReplayBuffer(capacity=capacity, num_rays=4, action_dim=ACTION_DIM, seed=1)
    for i in range(fill):
        obs = fake_obs(rng)
        obs["rays"] = obs["rays"][:4]
        obs["proprio"][0] = i % 512
        obs["events"][:] = 0.0
        if i in spike_steps:
            obs["events"][0] = 1.0  # ate
        buf.add(obs, np.zeros(ACTION_DIM, dtype=np.float32),
                salience=1.0 if i in spike_steps else 0.0)
    return buf


def test_prioritized_rows_contain_spikes() -> None:
    buf = _spiked_buffer(capacity=1000, fill=300, spike_steps={137})
    for _ in range(10):
        batch = buf.sample_sequences(4, 32, prioritized=2, spike_offset=8)
        assert batch is not None
        for row in range(2):
            hits = np.flatnonzero(batch["events"][row, :, 0] > 0)
            assert hits.size, "prioritized row must contain the spike"
            assert batch["proprio"][row, hits[0], 0] == 137
            # burn-in offset: the spike lands in the graded region.
            assert hits[0] >= 8
            np.testing.assert_array_equal(np.diff(batch["proprio"][row, :, 0]), 1)


def test_prioritized_rows_wrap_ring() -> None:
    buf = _spiked_buffer(capacity=100, fill=150, spike_steps={130})
    for _ in range(10):
        batch = buf.sample_sequences(2, 32, prioritized=1)
        assert batch is not None
        assert (batch["events"][0, :, 0] > 0).any()
        np.testing.assert_array_equal(np.diff(batch["proprio"][0, :, 0]) % 512, 1)


def test_prioritized_falls_back_uniform_without_spikes() -> None:
    buf = _filled_buffer(capacity=1000, fill=300)  # no events anywhere
    batch = buf.sample_sequences(4, 32, prioritized=2)
    assert batch is not None and batch["depth"].shape == (4, 32, 4)


def test_prioritize_brain_learns_and_reports() -> None:
    cfg = dict(TINY, replay=dict(TINY["replay"], prioritize="reward", prioritize_rows=1))
    brain = DreamerBrain(cfg, seed=50)
    rng = np.random.default_rng(50)
    for i in range(80):
        obs = fake_obs(rng)
        obs["events"][:] = 0.0
        if i % 20 == 5:
            obs["events"][0] = 1.0  # occasional meal
        brain.act(obs)
    metrics = brain.learn()
    assert metrics is not None
    assert np.isfinite(metrics["loss_reward"])
    assert "spike_row_frac" in metrics and metrics["spike_row_frac"] > 0.0
    assert "reward_head_spike_err" in metrics and np.isfinite(metrics["reward_head_spike_err"])


def test_unknown_prioritize_rejected() -> None:
    cfg = dict(TINY, replay=dict(TINY["replay"], prioritize="vibes"))
    with pytest.raises(ValueError, match="prioritize"):
        DreamerBrain(cfg, seed=51)


def _drive_obs(rng: np.random.Generator, energy: float) -> Observation:
    obs = fake_obs(rng)
    obs["proprio"][5] = energy  # energy
    obs["proprio"][6] = 1.0  # full integrity
    obs["proprio"][14] = 0.0  # fully rested
    return obs


def test_salience_is_drive_reduction_not_event_flag() -> None:
    """A meal at satiety is worth zero salience; a starving meal is loud.

    The swift_01 screen finding: all 4 recorded meals happened at energy
    >= 0.96 (above the 0.85 setpoint), so event-flag priority would have fed
    the reward head windows carrying exactly zero reward information.
    """
    cfg = dict(TINY, reward={"homeostasis": "drive"},
               replay=dict(TINY["replay"], prioritize="reward"))
    brain = DreamerBrain(cfg, seed=52)
    rng = np.random.default_rng(52)
    brain.act(_drive_obs(rng, 0.95))
    brain.act(_drive_obs(rng, 0.99))  # sated nibble: no deficit moved
    brain.act(_drive_obs(rng, 0.30))  # collapse into hunger
    brain.act(_drive_obs(rng, 0.30))
    brain.act(_drive_obs(rng, 0.80))  # the meal that matters
    sal = brain.buffer.salience[:5].astype(np.float32)
    assert sal[0] == 0.0, "first step has no predecessor"
    assert sal[1] < 1e-3, "sated meal carries ~no salience"
    assert sal[2] > 0.1, "crashing into hunger is salient"
    assert sal[4] > 0.1, "a hungry meal is salient"
    assert sal[4] > sal[1]
    # A stream break severs the delta: no fake spike across the gap.
    brain.reset_stream()
    brain.act(_drive_obs(rng, 0.99))
    assert brain.buffer.salience[5] == 0.0


def test_salience_recomputed_for_old_checkpoints() -> None:
    cfg = dict(TINY, reward={"homeostasis": "drive"},
               replay=dict(TINY["replay"], prioritize="reward"))
    brain = DreamerBrain(cfg, seed=53)
    rng = np.random.default_rng(53)
    for energy in (0.9, 0.9, 0.25, 0.25, 0.75, 0.9):
        brain.act(_drive_obs(rng, energy))
    state = brain.state_dict()
    del state["buffer"]["salience"]  # pre-salience checkpoint
    fresh = DreamerBrain(cfg, seed=54)
    fresh.load_state_dict(state)
    got = fresh.buffer.salience[:6].astype(np.float32)
    want = brain.buffer.salience[:6].astype(np.float32)
    # fp16 storage of the live path vs fp32 recompute: loose tolerance.
    np.testing.assert_allclose(got, want, atol=2e-3)
    assert got[2] > 0.1 and got[4] > 0.1


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
