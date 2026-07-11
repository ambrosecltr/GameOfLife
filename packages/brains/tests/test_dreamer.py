"""DreamerV3 component tests: shapes, invariants, and a tiny convergence probe."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from gol_brains import feeling
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


@pytest.mark.parametrize(
    ("replay", "message"),
    [
        (
            {
                "capacity": 100,
                "batch_size": 1,
                "seq_len": 64,
                "burn_in": 16,
                "warmup_steps": 32,
            },
            "warmup_steps must be at least",
        ),
        (
            {"capacity": 64, "batch_size": 1, "seq_len": 64, "warmup_steps": 66},
            "capacity must be at least replay.burn_in",
        ),
        (
            {"capacity": 32, "batch_size": 1, "seq_len": 4, "warmup_steps": 64},
            "capacity must be at least replay.warmup_steps",
        ),
    ],
)
def test_replay_contract_rejects_permanently_unlearnable_configs(
    replay: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DreamerBrain({**TINY, "replay": replay}, seed=1)


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


def test_cuda_learning_sync_uses_the_brain_device(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = DreamerBrain.__new__(DreamerBrain)
    brain.device = torch.device("cuda:1")
    requested: list[torch.device] = []
    synchronized: list[bool] = []

    class FakeStream:
        def synchronize(self) -> None:
            synchronized.append(True)

    def current_stream(device: torch.device) -> FakeStream:
        requested.append(device)
        return FakeStream()

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    brain._synchronize_learning_stream()

    assert requested == [torch.device("cuda:1")]
    assert synchronized == [True]


def body_obs(rng: np.random.Generator, energy: float, integrity: float) -> Observation:
    """fake_obs with a controlled internal state (drive reads proprio 5/6/14)."""
    obs = fake_obs(rng)
    obs["proprio"][5] = energy
    obs["proprio"][6] = integrity
    obs["proprio"][14] = 0.0  # rested
    return obs


DRIVE = {"homeostasis": "drive"}


def test_priced_blackout_wake_carries_salience() -> None:
    brain = DreamerBrain({**TINY, "reward": {**DRIVE, "blackout": "priced"}}, seed=3)
    rng = np.random.default_rng(2)
    brain.act(body_obs(rng, energy=0.9, integrity=1.0))
    brain.act(body_obs(rng, energy=0.02, integrity=1.0))  # pre-collapse step
    brain.wake()
    # The mind was off: live recurrent state resets even in priced mode...
    assert float(brain.h.abs().sum()) == 0.0 and float(brain.z.abs().sum()) == 0.0
    # ...but the salience chain survives the gap.
    assert brain._prev_drive is not None
    brain.act(body_obs(rng, energy=0.4, integrity=0.6))  # wake observation
    assert float(brain.buffer.salience[2]) > 0.1, "the gap's drive delta must be a real spike"


def test_cut_blackout_wake_severs_salience() -> None:
    brain = DreamerBrain({**TINY, "reward": dict(DRIVE)}, seed=3)  # blackout defaults to cut
    rng = np.random.default_rng(2)
    brain.act(body_obs(rng, energy=0.9, integrity=1.0))
    brain.act(body_obs(rng, energy=0.02, integrity=1.0))
    brain.wake()
    assert brain._prev_drive is None
    brain.act(body_obs(rng, energy=0.4, integrity=0.6))
    assert float(brain.buffer.salience[2]) == 0.0, "legacy wake must not fake a spike"


def test_respawn_severs_salience_even_when_priced() -> None:
    brain = DreamerBrain({**TINY, "reward": {**DRIVE, "blackout": "priced"}}, seed=3)
    rng = np.random.default_rng(2)
    brain.act(body_obs(rng, energy=0.02, integrity=1.0))
    brain.reset_stream()  # new body: nobody lived this gap
    assert brain._prev_drive is None


def test_stream_break_zeroes_replayed_reduction() -> None:
    """A window spanning a respawn must not pay the newborn's full tank as
    drive reduction (beta_09 census: +3.9 vs +0.5 for a real meal)."""
    brain = DreamerBrain({**TINY, "reward": dict(DRIVE)}, seed=7)
    proprio = torch.zeros(1, 2, PROPRIO_DIM)
    proprio[0, 0, 5], proprio[0, 0, 6] = 0.01, 0.02  # dying
    proprio[0, 1, 5], proprio[0, 1, 6] = 1.0, 1.0  # newborn, full tank
    events = torch.zeros(1, 2, EVENTS_DIM)
    unmasked = brain._homeostasis(events, proprio)
    assert float(unmasked[0, 1]) > 1.0, "sanity: the fictional jackpot exists unmasked"
    masked = brain._homeostasis(events, proprio, torch.tensor([[0.0, 1.0]]))
    assert float(masked[0, 1]) <= 0.0, "marked break: only the level penalty may remain"


def test_wake_marks_stream_break_only_when_cut() -> None:
    for mode, expect_break in (("cut", 1), ("priced", 0)):
        brain = DreamerBrain({**TINY, "reward": {**DRIVE, "blackout": mode}}, seed=3)
        rng = np.random.default_rng(2)
        brain.act(body_obs(rng, energy=0.9, integrity=1.0))
        brain.act(body_obs(rng, energy=0.02, integrity=1.0))
        brain.wake()
        brain.act(body_obs(rng, energy=0.4, integrity=0.6))
        assert brain.buffer.first[0] == 1, "a fresh mind's first step is a break"
        assert brain.buffer.first[1] == 0
        assert brain.buffer.first[2] == expect_break, mode


def test_priced_blackout_requires_drive_homeostasis() -> None:
    with pytest.raises(ValueError, match="blackout"):
        DreamerBrain({**TINY, "reward": {"blackout": "priced"}}, seed=0)
    with pytest.raises(ValueError, match="blackout"):
        DreamerBrain({**TINY, "reward": {**DRIVE, "blackout": "banana"}}, seed=0)


def test_prev_drive_rides_checkpoint() -> None:
    cfg = {**TINY, "reward": {**DRIVE, "blackout": "priced"}}
    brain = DreamerBrain(cfg, seed=3)
    rng = np.random.default_rng(2)
    brain.act(body_obs(rng, energy=0.3, integrity=0.9))
    assert brain._prev_drive is not None
    twin = DreamerBrain(cfg, seed=9)
    twin.load_state_dict(brain.state_dict())
    assert twin._prev_drive == pytest.approx(brain._prev_drive)


def test_spike_weighted_reward_loss() -> None:
    base_cfg = {**TINY, "reward": dict(DRIVE)}
    weighted_cfg = {**TINY, "reward": {**DRIVE, "spike_loss_weight": 4.0}}
    losses = []
    for cfg in (base_cfg, weighted_cfg):
        brain = DreamerBrain(cfg, seed=5)
        rng = np.random.default_rng(4)
        for _ in range(80):
            brain.act(fake_obs(rng))
        metrics = brain.learn()
        assert metrics is not None and np.isfinite(metrics["loss_reward"])
        losses.append(metrics["loss_reward"])
    # Identical seed and data: weights >= 1 elementwise, so the weighted
    # batch's reward loss can only match or exceed the unweighted one.
    assert losses[1] >= losses[0]


def test_priced_blackout_brain_learns() -> None:
    brain = DreamerBrain({**TINY, "reward": {**DRIVE, "blackout": "priced"}}, seed=6)
    rng = np.random.default_rng(5)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None
    assert np.isfinite(metrics["loss_model"])


# ------------------------------------------------------------ viability drive

VIA = {"scale": 1.0, "energy_safe": 0.25, "integrity_safe": 0.5, "barrier_cap": 4.0}
# The STAGED round-012 operating point: reduction off, the standing tax alone
# carries the mortality gradient (offline calibration, proposal 003).
VIA_STAGED = {**VIA, "scale": 0.0, "floor": 1.0}


def test_wellbeing_is_bounded_and_tracks_regulation() -> None:
    viability = torch.tensor([0.0, 1.0, 4.0])
    drive = torch.tensor([0.0, 0.5, 1.0])
    value = feeling.wellbeing(
        viability,
        drive,
        weight=0.25,
        barrier_cap=4.0,
        comfort_decay=1.0,
    )
    assert value[0] == pytest.approx(0.25)
    assert 0.0 < float(value[1]) < 0.25
    assert value[2] == pytest.approx(0.0)


def test_acute_pain_requires_damage_and_an_experienced_predecessor() -> None:
    proprio = torch.zeros(1, 4, PROPRIO_DIM)
    proprio[..., 6] = torch.tensor([1.0, 0.88, 0.76, 0.64])
    damage = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    discontinuity = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    loss = feeling.acute_integrity_loss(proprio, damage, discontinuity)
    torch.testing.assert_close(loss, torch.tensor([[0.0, 0.12, 0.0, 0.0]]))


def test_suspended_blackout_discounts_future_by_elapsed_time() -> None:
    cfg = {
        **TINY,
        "reward": {
            **DRIVE,
            "blackout": "suspended",
            "death_terminal": True,
            "viability": dict(VIA_STAGED),
        },
    }
    brain = DreamerBrain(cfg, seed=17)
    proprio = torch.zeros(1, 3, PROPRIO_DIM)
    proprio[..., 6] = torch.tensor([1.0, 1.0, 0.0])
    scale = torch.tensor([[1.0, 101.0, 1.0]])
    target, death = brain._continuation_target(proprio, scale)
    assert target[0, 0] == pytest.approx(1.0)
    assert float(target[0, 1]) == pytest.approx(brain.gamma**100, rel=1e-5)
    assert target[0, 2] == pytest.approx(0.0)
    assert death.tolist() == [[False, False, True]]


def test_pain_enabled_brain_has_separate_affect_and_damage_head() -> None:
    cfg = {
        **TINY,
        "reward": {
            **DRIVE,
            "imagined_homeostasis": "proprio",
            "pain": {"weight": 5.0, "event_loss_weight": 8.0},
        },
        "actor_critic": {"imagination_horizon": 5, "vector_critic": True},
    }
    brain = DreamerBrain(cfg, seed=18)
    assert brain.affect_names[-1] == "pain"
    assert brain.wm.head_damage is not None
    rng = np.random.default_rng(18)
    for index in range(80):
        obs = fake_obs(rng)
        if index % 10 == 0:
            obs["events"][1] = 1.0
        brain.act(obs)
    metrics = brain.learn()
    assert metrics is not None
    assert np.isfinite(metrics["loss_damage"])
    assert np.isfinite(metrics["affect_pain"])


def test_viability_barrier_zero_when_safe_rises_toward_floor() -> None:
    brain = DreamerBrain({**TINY, "reward": {**DRIVE, "viability": dict(VIA)}}, seed=0)
    rng = np.random.default_rng(0)
    # Above both safety margins: no viability cost at all.
    safe = brain._viability(brain._obs_to_tensors(body_obs(rng, 0.9, 0.9))["proprio"])
    assert float(safe[0]) == pytest.approx(0.0, abs=1e-6)
    # Approaching the energy floor: strictly rising, capped at barrier_cap.
    mild = brain._viability(brain._obs_to_tensors(body_obs(rng, 0.10, 0.9))["proprio"])
    dire = brain._viability(brain._obs_to_tensors(body_obs(rng, 0.02, 0.9))["proprio"])
    assert 0.0 < float(mild[0]) < float(dire[0]) <= brain.via_barrier_cap + 1e-5


def test_viability_total_cap_applies_after_component_sum() -> None:
    cfg = {
        **TINY,
        "reward": {
            **DRIVE,
            "viability": {**VIA, "total_cap": 4.0},
        },
    }
    brain = DreamerBrain(cfg, seed=39)
    rng = np.random.default_rng(39)
    both_dire = brain._viability(
        brain._obs_to_tensors(body_obs(rng, energy=0.0, integrity=0.0))["proprio"]
    )
    assert float(both_dire[0]) == pytest.approx(4.0)


def test_viability_reward_rewards_escaping_the_boundary() -> None:
    brain = DreamerBrain({**TINY, "reward": {**DRIVE, "viability": dict(VIA)}}, seed=0)
    # proprio[t]: step 0 near the floor, step 1 recovered — moving away from
    # death must pay a positive viability reward.
    proprio = torch.zeros(1, 2, PROPRIO_DIM)
    proprio[0, 0, 5], proprio[0, 0, 6] = 0.03, 0.9
    proprio[0, 1, 5], proprio[0, 1, 6] = 0.5, 0.9
    r = brain._viability_reward(proprio)
    assert float(r[0, 1]) > 0.0, "escaping the barrier is rewarded"
    # And sliding toward it is punished (reverse the pair).
    r_rev = brain._viability_reward(proprio.flip(1))
    assert float(r_rev[0, 1]) < 0.0


def test_viability_off_is_beta10_exactly() -> None:
    """No viability block (default) must leave the reward-head target untouched."""
    off = DreamerBrain({**TINY, "reward": dict(DRIVE)}, seed=1)
    assert not off.via_on, "no scale and no floor: the drive is off"
    proprio = torch.rand(2, 4, PROPRIO_DIM)
    via = off._viability_reward(proprio)  # both terms zero-scaled
    assert float(via.abs().sum()) == pytest.approx(0.0)


def test_viability_requires_drive_homeostasis() -> None:
    with pytest.raises(ValueError, match="viability"):
        DreamerBrain({**TINY, "reward": {"viability": {"scale": 1.0}}}, seed=0)


def test_life_return_goes_negative_and_rides_checkpoint() -> None:
    """The reframe as a live measurement: a lived stream of decaying energy
    integrates to a negative homeostatic return."""
    cfg = {**TINY, "reward": {**DRIVE, "viability": dict(VIA)}}
    brain = DreamerBrain(cfg, seed=3)
    rng = np.random.default_rng(1)
    for e in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4):  # a body draining toward the floor
        brain.act(body_obs(rng, energy=e, integrity=0.9))
    assert brain._life_return_homeo < 0.0, "draining life earns negative homeostatic return"
    twin = DreamerBrain(cfg, seed=9)
    twin.load_state_dict(brain.state_dict())
    assert twin._life_return_homeo == pytest.approx(brain._life_return_homeo)
    assert twin._life_return_via == pytest.approx(brain._life_return_via)
    # A stream break zeroes the per-life integral (a new body starts fresh).
    brain.reset_stream()
    assert brain._life_return_homeo == 0.0 and brain._prev_via is None


@pytest.mark.parametrize("via", [VIA, VIA_STAGED], ids=["reduction", "floor-tax"])
def test_viability_salience_spikes_near_death(via: dict[str, float]) -> None:
    """A plunge toward the floor adds salience BEYOND the comfort drive's — in
    both barrier forms. The floor-tax case is the staged round-012 config and
    is the regression: with only the |scale·ΔV| term, scale 0 contributed
    exactly zero replay priority and near-death was never oversampled."""

    def plunge_salience(cfg: dict[str, object]) -> float:
        brain = DreamerBrain({**TINY, "reward": cfg}, seed=4)
        rng = np.random.default_rng(2)
        brain.act(body_obs(rng, energy=0.20, integrity=0.9))
        brain.act(body_obs(rng, energy=0.03, integrity=0.9))  # deep into the barrier
        return float(brain.buffer.salience[1])

    with_barrier = plunge_salience({**DRIVE, "viability": dict(via)})
    without = plunge_salience(dict(DRIVE))
    assert with_barrier > without + 0.1, "the barrier must add its own priority"


def test_viability_brain_learns_with_full_stack() -> None:
    cfg = {
        **TINY,
        "reward": {
            **DRIVE,
            "blackout": "priced",
            "viability": dict(VIA),
            "death_terminal": True,
            "boredom": {"weight": 0.02, "gate": "viability", "pressure": True},
            "curiosity": "lp",
        },
    }
    brain = DreamerBrain(cfg, seed=6)
    rng = np.random.default_rng(5)
    for _ in range(80):
        brain.act(body_obs(rng, energy=0.5, integrity=0.7))
    metrics = brain.learn()
    assert metrics is not None
    assert np.isfinite(metrics["loss_model"])
    assert np.isfinite(metrics["reward_viability"])
    assert "viability_level" in metrics


def test_record_death_writes_terminal_sample() -> None:
    """The runtime-delivered death lands in the buffer AT the lethal floor, so
    the cont head has real terminal targets (integrity <= lethal) to learn
    from — without it death_terminal trains 'continue' everywhere, because a
    dying body is never observable from inside (round-012 review)."""
    cfg = {
        **TINY,
        "reward": {
            **DRIVE,
            "blackout": "priced",
            "viability": dict(VIA_STAGED),
            "death_terminal": True,
        },
    }
    brain = DreamerBrain(cfg, seed=5)
    rng = np.random.default_rng(3)
    brain.act(body_obs(rng, energy=0.30, integrity=0.4))
    last = body_obs(rng, energy=0.05, integrity=0.2)
    brain.act(last)
    n = len(brain.buffer)
    brain.record_death(last, dormant=True)
    assert len(brain.buffer) == n + 1
    dead = brain.buffer.proprio[n]
    assert float(dead[6]) == 0.0, "integrity at the lethal floor — the terminal target"
    assert float(dead[5]) == 0.0, "a hibernation death's energy had already collapsed"
    assert not np.any(brain.buffer.action[n]), "dead bodies don't act"
    assert brain.buffer.first[n] == 0, "priced: the plunge is a real transition"
    assert float(brain.buffer.salience[n]) > 1.0, "the terminal plunge is maximally salient"


def test_record_death_is_gated_on_death_terminal() -> None:
    """Without the terminal knob the buffer stays byte-identical to beta_10 —
    the A/B against beta_10 must not gain samples it never had."""
    brain = DreamerBrain({**TINY, "reward": {**DRIVE, "blackout": "priced"}}, seed=5)
    rng = np.random.default_rng(3)
    obs = body_obs(rng, energy=0.05, integrity=0.2)
    brain.act(obs)
    n = len(brain.buffer)
    brain.record_death(obs, dormant=True)
    assert len(brain.buffer) == n


def test_record_death_under_cut_blackout_is_a_stream_break() -> None:
    """cut severs the dormant gap: a hibernation death is recorded, but no
    fictional drive delta is read across the unlived blackout."""
    cfg = {
        **TINY,
        "reward": {
            **DRIVE,
            "blackout": "cut",
            "viability": dict(VIA_STAGED),
            "death_terminal": True,
        },
    }
    brain = DreamerBrain(cfg, seed=5)
    rng = np.random.default_rng(3)
    obs = body_obs(rng, energy=0.9, integrity=0.9)
    brain.act(obs)
    n = len(brain.buffer)
    brain.record_death(obs, dormant=True)
    assert brain.buffer.first[n] == 1
    assert float(brain.buffer.salience[n]) == 0.0


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
    for key in (
        "loss_model",
        "kl",
        "curiosity",
        "loss_critic",
        "loss_actor",
        "homeo_max",
        "homeo_spike_frac",
        "learn_seconds",
        "policy_cont_std_mean",
        "policy_cont_std_max",
        "policy_action_abs_mean",
        "policy_action_saturation_frac",
        "policy_rest_sample_frac",
    ):
        assert key in metrics and np.isfinite(metrics[key])
    assert 0.1 <= metrics["policy_cont_std_mean"] <= metrics["policy_cont_std_max"] <= 1.0
    assert 0.0 <= metrics["policy_action_saturation_frac"] <= 1.0
    assert 0.0 <= metrics["policy_rest_sample_frac"] <= 1.0
    # Pacing counters: 80 acts recorded, 1 update done.
    assert brain.experience_count() == 80
    assert metrics["act_steps"] == 80.0
    assert metrics["updates"] == 1.0
    assert metrics["train_ratio_eff"] == 1.0 / 80
    assert brain.target_train_ratio() == 0.25  # config default


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
    assert fresh.experience_count() == brain.experience_count()
    # Pre-pacing checkpoints (no act_steps key) seed the counter from the
    # stored buffer so the update/act-step pair stays coherent.
    legacy = {k: v for k, v in state.items() if k != "act_steps"}
    older = DreamerBrain(TINY, seed=98)
    older.load_state_dict(legacy)
    assert older.experience_count() == len(brain.buffer)
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


def test_drive_reward_semantics() -> None:
    """HRRL drive-reduction: valence is need-relative, satiation is automatic."""
    cfg = dict(TINY, reward={"homeostasis": "drive"})
    brain = DreamerBrain(cfg, seed=8)
    proprio = torch.zeros(1, 7, PROPRIO_DIM)
    proprio[..., 6] = 1.0  # full integrity
    proprio[..., 14] = 0.0  # fully rested
    # Energy: hungry, hungry, +0.2 meal, +0.2 meal, hold, +0.2 near-sated meal,
    # then topping up past the setpoint.
    proprio[0, :, 5] = torch.tensor([0.3, 0.3, 0.5, 0.7, 0.7, 0.9, 1.0])
    events = torch.zeros(1, 7, EVENTS_DIM)
    r = brain._homeostasis(events, proprio)[0]
    assert r[2] > 0.1, "eating while starving is strongly rewarded"
    assert r[2] > r[5], "the same 0.2 energy is worth more the hungrier you are"
    assert abs(r[6]) < 1e-4, "eating past the setpoint earns nothing (satiation)"
    assert r[1] < 0.0, "a standing deficit stings (level penalty)"

    # Fatigue is a drive too: growing tiredness feels bad, winding down feels good.
    proprio2 = torch.zeros(1, 3, PROPRIO_DIM)
    proprio2[..., 5] = 1.0
    proprio2[..., 6] = 1.0
    proprio2[0, :, 14] = torch.tensor([0.2, 0.6, 0.3])
    r2 = brain._homeostasis(torch.zeros(1, 3, EVENTS_DIM), proprio2)[0]
    assert r2[1] < 0.0 and r2[2] > 0.0


def test_drive_mode_brain_learns() -> None:
    cfg = dict(TINY, reward={"homeostasis": "drive"})
    brain = DreamerBrain(cfg, seed=9)
    rng = np.random.default_rng(6)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None and np.isfinite(metrics["loss_model"])
    assert "drive_level" in metrics and metrics["drive_level"] >= 0.0


def test_unknown_homeostasis_mode_rejected() -> None:
    with pytest.raises(ValueError, match="homeostasis"):
        DreamerBrain(dict(TINY, reward={"homeostasis": "vibes"}), seed=10)


def test_online_regions_separate_clusters() -> None:
    from gol_brains.dreamer.interest import OnlineRegions

    torch.manual_seed(0)
    regions = OnlineRegions(2, 4, lr=0.5, device=torch.device("cpu"))
    a = torch.randn(64, 4) * 0.1 + torch.tensor([5.0, 0, 0, 0])
    b = torch.randn(64, 4) * 0.1 + torch.tensor([-5.0, 0, 0, 0])
    for _ in range(5):
        regions.adapt(torch.cat([a, b]))
    ia, ib = regions.assign(a), regions.assign(b)
    assert (ia == ia[0]).all() and (ib == ib[0]).all(), "each cluster maps to one region"
    assert ia[0] != ib[0], "distinct clusters land in distinct regions"


def test_learning_progress_rewards_falling_error() -> None:
    from gol_brains.dreamer.interest import LearningProgress

    lp = LearningProgress(2, fast=0.5, slow=0.05, relative=False)
    idx = torch.tensor([0] * 8 + [1] * 8)
    # First sight seeds both EMAs: no progress from mere novelty.
    lp.update(idx, torch.cat([torch.full((8,), 1.0), torch.full((8,), 1.0)]))
    assert lp.lp().abs().max() < 1e-6
    # Region 0's error falls (learnable frontier); region 1 stays flat (noise).
    for err0 in (0.7, 0.5, 0.3, 0.2):
        lp.update(idx, torch.cat([torch.full((8,), err0), torch.full((8,), 1.0)]))
    assert lp.lp()[0] > 0.1
    assert lp.lp()[1] < 1e-6
    assert (lp.reward(torch.tensor([0])) > lp.reward(torch.tensor([1]))).all()
    # Checkpoint roundtrip.
    clone = LearningProgress(2, fast=0.5, slow=0.05, relative=False)
    clone.load_state_dict(lp.state_dict())
    torch.testing.assert_close(clone.lp(), lp.lp())


def test_lp_brain_learns_with_boredom() -> None:
    cfg = dict(
        TINY,
        reward={
            "curiosity": "lp",
            "homeostasis": "drive",
            "boredom": {"weight": 0.05},
        },
    )
    brain = DreamerBrain(cfg, seed=11)
    rng = np.random.default_rng(7)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None
    for key in (
        "lp_reward",
        "lp_regions",
        "boredom",
        "stimulation",
        "drive_level",
        "lp_p50",
        "lp_p90",
        "lp_stale_frac",
        "lp_occ_entropy",
        "boredom_calm_gate",
        "boredom_dull_gate",
    ):
        assert key in metrics and np.isfinite(metrics[key]), key
    resumed = DreamerBrain(cfg, seed=44)
    resumed.load_state_dict(brain.state_dict())
    assert resumed._active_skill == brain._active_skill
    assert resumed.critic[-1].weight.shape == brain.critic[-1].weight.shape
    assert metrics["lp_regions"] >= 1
    assert 0.0 <= metrics["lp_occ_entropy"] <= 1.0
    assert 0.0 <= metrics["lp_stale_frac"] <= 1.0
    # LP machinery must survive a checkpoint.
    fresh = DreamerBrain(cfg, seed=12)
    fresh.load_state_dict(brain.state_dict())
    torch.testing.assert_close(fresh.regions.centroids, brain.regions.centroids)
    torch.testing.assert_close(fresh.lp.fast, brain.lp.fast)


def test_kind_partition_brain_learns() -> None:
    cfg = dict(TINY, reward={"curiosity": "lp", "lp": {"partition": "kind"}})
    brain = DreamerBrain(cfg, seed=13)
    rng = np.random.default_rng(8)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None and np.isfinite(metrics["lp_reward"])


def test_boredom_bites_only_when_enabled() -> None:
    cfg = dict(
        TINY,
        reward={
            "homeostasis": "drive",
            "boredom": {"weight": 1.0, "stim_threshold": 100.0, "drive_threshold": 100.0},
        },
    )
    brain = DreamerBrain(cfg, seed=14)
    feat = torch.randn(6, brain.wm.rssm_cfg.feat_dim)
    action = torch.randn(6, ACTION_DIM)
    with torch.no_grad():
        reward_bored, _, bored = brain._imagination_reward(feat, action)
        brain.boredom_weight = 0.0
        reward_free, _, none = brain._imagination_reward(feat, action)
    assert (bored > 0).all(), "wide-open thresholds: everything is boring"
    assert (none == 0).all()
    assert (reward_bored < reward_free).all()


def test_temperament_sampling_and_inheritance() -> None:
    cfg = dict(TINY, temperament={"enabled": True, "sigma": 0.5, "mutation_sigma": 0.2})
    parent = DreamerBrain(cfg, seed=20)
    sibling = DreamerBrain(cfg, seed=21)
    assert parent.temperament != sibling.temperament, "birth diversity"
    assert parent.w_curiosity != sibling.w_curiosity
    plain = DreamerBrain(TINY, seed=22)
    assert all(v == 1.0 for v in plain.temperament.values()), "disabled -> neutral"

    # Checkpoint restore is exact; inheritance mutates.
    resumed = DreamerBrain(cfg, seed=23)
    resumed.load_state_dict(parent.state_dict())
    assert resumed.temperament == parent.temperament
    assert abs(resumed.w_curiosity - parent.w_curiosity) < 1e-9
    child = DreamerBrain(cfg, seed=24)
    child.inherit(parent.state_dict())
    assert child.temperament != parent.temperament, "mutation on inheritance"
    ratios = [child.temperament[k] / parent.temperament[k] for k in child.temperament]
    assert all(0.4 < r < 2.5 for r in ratios), "children resemble their parents"
    # Effective knobs follow the mutated temperament.
    assert abs(child.w_curiosity - parent.w_curiosity * ratios[0]) < 1e-9


def test_copied_newborn_starts_without_donor_update_credit() -> None:
    parent = DreamerBrain(TINY, seed=25)
    parent._act_steps = 80
    parent._updates = 2
    parent._dropped_update_credit = 1.0
    assert parent.pending_update_credit() == pytest.approx(1.25)

    child = DreamerBrain(TINY, seed=26)
    child.inherit(parent.state_dict())

    assert child.pending_update_credit() == 0.0
    assert child._dropped_update_credit == 0.0
    child._act_steps += 4
    assert child.pending_update_credit() == pytest.approx(1.0)


def test_masked_brain_learns() -> None:
    cfg = dict(TINY, reward={"curiosity_mask_agents": True})
    brain = DreamerBrain(cfg, seed=7)
    rng = np.random.default_rng(4)
    for _ in range(80):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None and np.isfinite(metrics["loss_model"])


def test_norm_anchor_freezes_scale() -> None:
    """Anchored normalization: a decaying signal reads as decayed (008 fix)."""
    from gol_brains.dreamer.networks import RunningMeanStd

    legacy = RunningMeanStd()
    anchored = RunningMeanStd(anchor=1000)
    rng = np.random.default_rng(0)
    # Calibration era: both see the same loud signal.
    for _ in range(10):
        x = torch.as_tensor(rng.normal(0, 1.0, 200), dtype=torch.float32)
        legacy.update(x)
        anchored.update(x)
    frozen_var = anchored.var
    # Decay era: the signal shrinks 10x.
    for _ in range(50):
        x = torch.as_tensor(rng.normal(0, 0.1, 200), dtype=torch.float32)
        legacy.update(x)
        anchored.update(x)
    assert anchored.var == frozen_var, "anchored stats must freeze after calibration"
    small = torch.as_tensor(rng.normal(0, 0.1, 500), dtype=torch.float32)
    # Anchored: normalized magnitude tracks the true decay. Legacy: the
    # shrinking yardstick re-inflates it (the hedonic treadmill).
    assert anchored.normalize(small).abs().mean() < legacy.normalize(small).abs().mean()
    assert anchored.normalize(small).abs().mean() < 0.3


def test_lp_mix_anneals_with_age() -> None:
    cfg = dict(TINY, reward={"curiosity": "lp", "lp": {"mix_anneal_steps": 100}})
    brain = DreamerBrain(cfg, seed=30)
    assert brain._lp_mix() == brain.lp_mix_disagreement, "newborn gets the full trickle"
    brain._act_steps = 50
    assert abs(brain._lp_mix() - 0.5 * brain.lp_mix_disagreement) < 1e-9
    brain._act_steps = 200
    assert brain._lp_mix() == 0.0, "adults get no disagreement subsidy"
    # Legacy: no anneal configured, the trickle is constant.
    legacy = DreamerBrain(dict(TINY, reward={"curiosity": "lp"}), seed=31)
    legacy._act_steps = 10**6
    assert legacy._lp_mix() == legacy.lp_mix_disagreement


def test_boredom_pressure_follows_lived_chronology_and_resets() -> None:
    cfg = dict(
        TINY,
        reward={
            "homeostasis": "drive",
            "boredom": {
                # Wide-open gates: once learning publishes a low-stimulation
                # estimate, subsequent lived steps are calm and dull.
                "weight": 1.0,
                "stim_threshold": 100.0,
                "drive_threshold": 100.0,
                "pressure": True,
                "pressure_rise": 0.05,
                "pressure_decay": 0.001,
            },
        },
    )
    brain = DreamerBrain(cfg, seed=32)
    feat = torch.randn(6, brain.wm.rssm_cfg.feat_dim)
    action = torch.randn(6, ACTION_DIM)
    with torch.no_grad():
        _, _, bored = brain._imagination_reward(feat, action)
    assert (bored == 0).all(), "a fresh mind has no accumulated boredom"

    rng = np.random.default_rng(9)
    for _ in range(80):
        brain.act(fake_obs(rng))
    m1 = brain.learn()
    assert m1 is not None and m1["boredom_pressure"] == 0.0, (
        "replaying past experience must not advance a chronological mood"
    )
    for _ in range(10):
        brain.act(fake_obs(rng))
    lived_pressure = brain._boredom_pressure
    assert lived_pressure > 0.0
    m2 = brain.learn()
    assert m2 is not None and m2["boredom_pressure"] == lived_pressure
    with torch.no_grad():
        _, _, bored = brain._imagination_reward(feat, action)
    assert (bored > 0).all(), "charged pressure makes dull imagined states cost"

    # Pressure survives a checkpoint; a warm-started newborn is not born jaded.
    resumed = DreamerBrain(cfg, seed=33)
    resumed.load_state_dict(brain.state_dict())
    assert resumed._boredom_pressure == brain._boredom_pressure
    child = DreamerBrain(cfg, seed=34)
    child.inherit(brain.state_dict())
    assert child._boredom_pressure == 0.0


class _ProprioFromFeature(nn.Module):
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return feat[..., :PROPRIO_DIM]


class _ConstantContinuation(nn.Module):
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.full((*feat.shape[:-1], 1), 20.0, device=feat.device)


def test_imagination_reads_predicted_body_state_directly() -> None:
    cfg = dict(
        TINY,
        reward={
            "homeostasis": "drive",
            "imagined_homeostasis": "proprio",
            "w_curiosity": 0.0,
        },
    )
    brain = DreamerBrain(cfg, seed=40)
    brain.wm.head_proprio = nn.Sequential(_ProprioFromFeature())
    before = torch.zeros(1, brain.wm.rssm_cfg.feat_dim)
    after = torch.zeros_like(before)
    before[..., 5], after[..., 5] = 0.2, 0.6
    before[..., 6] = after[..., 6] = 1.0
    action = torch.zeros(1, ACTION_DIM)
    with torch.no_grad():
        affect, _, _ = brain._imagination_affect(before, after, action)
    comfort = affect[..., brain.affect_names.index("comfort")]
    assert float(comfort) > 0.0, "a predicted reduction in hunger must feel beneficial"


def test_imagination_includes_regulated_wellbeing_in_viability_affect() -> None:
    cfg = dict(
        TINY,
        reward={
            "homeostasis": "drive",
            "imagined_homeostasis": "proprio",
            "w_curiosity": 0.0,
            "viability": {"scale": 0.0, "floor": 0.0, "barrier_cap": 4.0},
            "wellbeing": {"weight": 0.25, "comfort_decay": 1.0},
        },
        actor_critic={"imagination_horizon": 5, "vector_critic": True},
    )
    brain = DreamerBrain(cfg, seed=47)
    brain.wm.head_proprio = nn.Sequential(_ProprioFromFeature())
    regulated = torch.zeros(1, brain.wm.rssm_cfg.feat_dim)
    regulated[..., 5] = 0.85
    regulated[..., 6] = 1.0
    action = torch.zeros(1, ACTION_DIM)

    with torch.no_grad():
        affect, _, _ = brain._imagination_affect(regulated, regulated, action)

    viability = affect[..., brain.affect_names.index("viability")]
    assert viability == pytest.approx(torch.tensor([0.25]))


def test_imagination_continuation_obeys_predicted_coma_and_death_boundaries() -> None:
    cfg = dict(
        TINY,
        reward={
            "homeostasis": "drive",
            "imagined_homeostasis": "proprio",
            "blackout": "suspended",
            "death_terminal": True,
        },
    )
    brain = DreamerBrain(cfg, seed=48)
    brain.wm.head_proprio = nn.Sequential(_ProprioFromFeature())
    brain.wm.head_cont = nn.Sequential(_ConstantContinuation())
    states = torch.zeros(3, brain.wm.rssm_cfg.feat_dim)
    states[:, 5] = torch.tensor([0.8, 0.0, 0.8])
    states[:, 6] = torch.tensor([1.0, 1.0, 0.0])

    continuation = brain._imagination_continuation(states)

    assert continuation[0] > 0.999
    assert continuation[1:].tolist() == [0.0, 0.0]


def test_terminal_examples_receive_configured_weight() -> None:
    cfg = dict(TINY, reward={"terminal_loss_weight": 8.0})
    brain = DreamerBrain(cfg, seed=41)
    raw, weighted = brain._continuation_loss(torch.zeros(2), torch.tensor([1.0, 0.0]))
    assert weighted[0] == raw[0]
    assert weighted[1] == 8.0 * raw[1]


def test_temporal_skills_and_vector_affect_learn_end_to_end() -> None:
    cfg = {
        **TINY,
        "replay": {"capacity": 3000, "batch_size": 4, "seq_len": 16, "warmup_steps": 18},
        "reward": {
            "homeostasis": "drive",
            "imagined_homeostasis": "proprio",
            "fear_weight": 0.05,
        },
        "actor_critic": {
            "imagination_horizon": 8,
            "vector_critic": True,
        },
        "temporal_skills": {
            "enabled": True,
            "num_skills": 4,
            "duration": 4,
            "intrinsic_weight": 0.1,
        },
    }
    brain = DreamerBrain(cfg, seed=42)
    rng = np.random.default_rng(42)
    for _ in range(80):
        brain.act(fake_obs(rng))
    assert (brain.buffer.skill[: brain.warmup_steps] == -1).all(), (
        "motor babbling has no learned intent"
    )
    assert (brain.buffer.skill[brain.warmup_steps : 80] >= 0).all()
    for start in range(brain.warmup_steps, 80 - brain.skill_duration + 1, brain.skill_duration):
        assert len(set(brain.buffer.skill[start : start + brain.skill_duration])) == 1
    metrics = brain.learn()
    assert metrics is not None
    assert brain.critic(torch.randn(2, brain.wm.rssm_cfg.feat_dim)).shape == (2, 41 * 6)
    for key in (
        "value_comfort",
        "value_viability",
        "value_curiosity",
        "value_boredom",
        "value_fear",
        "value_skill",
        "skill_discriminator_loss",
        "skill_manager_entropy",
    ):
        assert key in metrics and np.isfinite(metrics[key]), key


def test_async_inference_publishes_immutable_controller_snapshots() -> None:
    cfg = {
        **TINY,
        "training": {"imag_starts": 16, "async_inference": True, "publish_every": 1},
    }
    brain = DreamerBrain(cfg, seed=43)
    assert brain.allows_concurrent_learning()
    initial = brain._inference
    assert initial is not None
    inference_parameter = next(initial.encoder.parameters())
    training_parameter = next(brain.wm.encoder.parameters())
    assert inference_parameter.data_ptr() != training_parameter.data_ptr()
    rng = np.random.default_rng(43)
    for _ in range(80):
        brain.act(fake_obs(rng))
    assert brain.learn() is not None
    assert brain._inference is not initial
