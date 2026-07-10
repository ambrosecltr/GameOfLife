"""Aion S5 architecture, lifecycle, and integration tests."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from gol_brains.aion.brain import AionBrain
from gol_brains.aion.s5 import S5Dynamics, S5DynamicsConfig, S5Stack
from gol_brains.dreamer.brain import ACTION_DIM, DreamerBrain
from gol_brains.dreamer.buffer import ReplayBuffer
from gol_brains.registry import build_brain, is_learning_kind
from gol_world.interface import (
    EVENTS_DIM,
    NUM_RAY_KINDS,
    PROPRIO_DIM,
    RAY_DIM,
    SOUND_DIM,
    BodySpec,
    Observation,
)

TINY_AION = {
    "kind": "aion",
    "preset": "nano",
    "world_model": {
        "s5": {
            "model_dim": 32,
            "state_dim": 8,
            "blocks": 2,
            "slow_fraction": 0.5,
        }
    },
    "replay": {
        "capacity": 256,
        "batch_size": 2,
        "seq_len": 8,
        "burn_in": 4,
        "warmup_steps": 16,
    },
    "training": {"imag_starts": 8},
    "actor_critic": {"imagination_horizon": 3},
}


def fake_obs(rng: np.random.Generator) -> Observation:
    body = BodySpec()
    rays = np.zeros((body.num_rays, RAY_DIM), dtype=np.float32)
    rays[:, 0] = rng.random(body.num_rays)
    rays[:, 1:4] = rng.random((body.num_rays, 3)).astype(np.float32)
    kinds = rng.integers(0, NUM_RAY_KINDS, body.num_rays)
    rays[np.arange(body.num_rays), 4 + kinds] = 1.0
    return Observation(
        rays=rays,
        proprio=rng.random(PROPRIO_DIM).astype(np.float32),
        sound=rng.random(SOUND_DIM).astype(np.float32),
        events=np.zeros(EVENTS_DIM, dtype=np.float32),
    )


def tiny_s5_config() -> S5DynamicsConfig:
    return S5DynamicsConfig(
        model_dim=12,
        state_dim=4,
        blocks=2,
        stoch_groups=2,
        stoch_classes=2,
        hidden=8,
    )


def test_parallel_s5_sequence_matches_recurrent_steps() -> None:
    torch.manual_seed(7)
    cfg = tiny_s5_config()
    stack = S5Stack(cfg)
    signal = torch.randn(3, 17, cfg.model_dim)
    initial = torch.randn(3, cfg.deter)
    first = torch.zeros(3, 17)
    wake = torch.zeros(3, 17)
    step_scale = torch.ones(3, 17)
    first[0, 7] = 1.0
    wake[1, 11] = 1.0
    step_scale[1, 11] = 6.0

    parallel = stack.sequence(signal, initial, first, wake, step_scale)
    state = initial
    recurrent = []
    for index in range(signal.shape[1]):
        state = torch.where(first[:, index, None] > 0.5, 0.0, state)
        fast_reset = stack.reset_fast(state)
        state = torch.where(wake[:, index, None] > 0.5, fast_reset, state)
        state = stack.step(signal[:, index], state, step_scale[:, index])
        recurrent.append(state)

    torch.testing.assert_close(parallel, torch.stack(recurrent, dim=1), atol=2e-6, rtol=2e-5)


def test_wake_scan_resets_fast_modes_and_advances_slow_modes() -> None:
    torch.manual_seed(8)
    cfg = tiny_s5_config()
    stack = S5Stack(cfg)
    initial = torch.randn(2, cfg.deter)
    signal = torch.zeros(2, 1, cfg.model_dim)
    first = torch.zeros(2, 1)
    wake = torch.ones(2, 1)
    step_scale = torch.full((2, 1), 9.0)

    scanned = stack.sequence(signal, initial, first, wake, step_scale)[:, 0]
    persistent = stack.reset_fast(initial)
    recurrent = stack.step(signal[:, 0], persistent, step_scale=9.0)
    torch.testing.assert_close(scanned, recurrent, atol=2e-6, rtol=2e-5)
    assert 0.0 < float(persistent.abs().sum()) < float(initial.abs().sum())


def test_parallel_observation_dynamics_match_online_recurrence() -> None:
    torch.manual_seed(81)
    cfg = tiny_s5_config()
    dynamics = S5Dynamics(cfg, embed_dim=7, action_dim=5)
    embed = torch.randn(1, 13, 7)
    action = torch.randn(1, 13, 5)
    markers = torch.zeros(1, 13)

    torch.manual_seed(82)
    parallel = dynamics.observe_sequence(
        embed, action, markers, markers, torch.ones_like(markers), burn_in=0
    )

    torch.manual_seed(82)
    h, z = dynamics.initial(1, torch.device("cpu"))
    zero_action = torch.zeros(1, 5)
    states_h = []
    states_z = []
    posts = []
    priors = []
    for index in range(embed.shape[1]):
        previous_action = action[:, index - 1] if index else zero_action
        h, z, post, prior = dynamics.obs_step(h, z, previous_action, embed[:, index])
        states_h.append(h)
        states_z.append(z)
        posts.append(post)
        priors.append(prior)

    torch.testing.assert_close(parallel.h, torch.stack(states_h, dim=1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(parallel.z, torch.stack(states_z, dim=1))
    torch.testing.assert_close(parallel.post, torch.stack(posts, dim=1))
    torch.testing.assert_close(parallel.prior, torch.stack(priors, dim=1), atol=2e-6, rtol=2e-5)


def test_s5_backpropagates_through_1024_step_context() -> None:
    torch.manual_seed(9)
    cfg = S5DynamicsConfig(
        model_dim=8,
        state_dim=4,
        blocks=2,
        stoch_groups=2,
        stoch_classes=2,
        hidden=8,
    )
    stack = S5Stack(cfg)
    signal = torch.randn(2, 1024, cfg.model_dim, requires_grad=True)
    initial = stack.initial(2, torch.device("cpu"))
    zeros = torch.zeros(2, 1024)
    state = stack.sequence(signal, initial, zeros, zeros, torch.ones_like(zeros))
    loss = state.square().mean()
    # PyTorch's runtime method is not annotated in the installed stubs.
    loss.backward()  # type: ignore[no-untyped-call]
    assert signal.grad is not None and bool(torch.isfinite(signal.grad).all())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in stack.parameters()
    )


def test_aion_wake_preserves_context_and_records_elapsed_transition() -> None:
    brain = AionBrain(TINY_AION, seed=10)
    rng = np.random.default_rng(10)
    brain.act(fake_obs(rng))
    brain.act(fake_obs(rng))
    before = brain.h.clone()
    expected = brain.s5.reset_fast(before)

    brain.wake(dormant_steps=7)
    torch.testing.assert_close(brain.h, expected)
    assert float(brain.z.abs().sum()) == 0.0
    assert brain._stream_wake and brain._step_scale == 8.0

    brain.act(fake_obs(rng))
    assert brain.buffer.first[2] == 0
    assert brain.buffer.wake[2] == 1
    assert brain.buffer.step_scale[2] == pytest.approx(8.0)
    assert not brain._stream_wake and brain._step_scale == 1.0


def test_aion_new_body_clears_all_live_state() -> None:
    brain = AionBrain(TINY_AION, seed=11)
    rng = np.random.default_rng(11)
    brain.act(fake_obs(rng))
    brain.act(fake_obs(rng))
    assert float(brain.h.abs().sum()) > 0.0
    brain.reset_stream()
    assert float(brain.h.abs().sum()) == 0.0
    assert float(brain.z.abs().sum()) == 0.0
    assert brain._stream_first and not brain._stream_wake


def test_dormant_death_replay_carries_blackout_duration() -> None:
    cfg = {**TINY_AION, "reward": {"death_terminal": True}}
    brain = AionBrain(cfg, seed=17)
    rng = np.random.default_rng(17)
    last = fake_obs(rng)
    brain.act(last)
    brain.record_death(last, dormant=True, dormant_steps=7)
    assert brain.buffer.wake[1] == 1
    assert brain.buffer.step_scale[1] == pytest.approx(8.0)


def test_aion_learns_and_roundtrips_checkpoint() -> None:
    brain = AionBrain(TINY_AION, seed=12)
    rng = np.random.default_rng(12)
    for _ in range(24):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None and np.isfinite(metrics["loss_model"])
    assert metrics["context_steps"] == 8.0
    assert metrics["graded_timepoints_per_second"] > 0.0

    state = brain.state_dict()
    assert state["brain_family"] == "aion"
    fresh = AionBrain(TINY_AION, seed=13)
    fresh.load_state_dict(state)
    assert len(fresh.buffer) == len(brain.buffer)
    torch.testing.assert_close(fresh.h, brain.h)
    for expected, actual in zip(brain.wm.parameters(), fresh.wm.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_lineage_checkpoint_identity_is_enforced() -> None:
    aion = AionBrain(TINY_AION, seed=14)
    dreamer = DreamerBrain(
        {
            "kind": "dreamer",
            "preset": "nano",
            "replay": {"capacity": 64, "batch_size": 2, "seq_len": 4},
        },
        seed=14,
    )
    with pytest.raises(ValueError, match="checkpoint belongs"):
        dreamer.load_state_dict(aion.state_dict())
    with pytest.raises(ValueError, match="checkpoint belongs"):
        aion.load_state_dict(dreamer.state_dict())


def test_pre_aion_beta_checkpoint_remains_compatible() -> None:
    cfg = {
        "kind": "dreamer",
        "preset": "nano",
        "replay": {"capacity": 64, "batch_size": 2, "seq_len": 4},
    }
    brain = DreamerBrain(cfg, seed=18)
    brain.act(fake_obs(np.random.default_rng(18)))
    state = brain.state_dict()
    for key in ("brain_family", "stream_wake", "step_scale"):
        state.pop(key)
    state["buffer"].pop("wake")
    state["buffer"].pop("step_scale")

    restored = DreamerBrain(cfg, seed=19)
    restored.load_state_dict(state)
    assert len(restored.buffer) == 1
    assert restored.buffer.wake[0] == 0
    assert restored.buffer.step_scale[0] == 1.0


def test_replay_defaults_legacy_checkpoints_to_no_wake_and_unit_time() -> None:
    rng = np.random.default_rng(15)
    buffer = ReplayBuffer(capacity=16, num_rays=BodySpec().num_rays, action_dim=ACTION_DIM, seed=15)
    buffer.add(fake_obs(rng), np.zeros(ACTION_DIM, dtype=np.float32), wake=True, step_scale=6)
    state = buffer.state_dict()
    del state["wake"]
    del state["step_scale"]

    restored = ReplayBuffer(
        capacity=16, num_rays=BodySpec().num_rays, action_dim=ACTION_DIM, seed=16
    )
    restored.load_state_dict(state)
    assert restored.wake[0] == 0
    assert restored.step_scale[0] == 1.0


def test_aion_is_registered_as_a_scheduled_learner() -> None:
    brain = build_brain(TINY_AION, seed=16)
    assert isinstance(brain, AionBrain)
    assert is_learning_kind(TINY_AION)


def test_aion_01_uses_full_1024_step_research_context() -> None:
    root = Path(__file__).parents[3]
    config = yaml.safe_load((root / "configs/brain/aion_01_s5.yaml").read_text())
    assert config["kind"] == "aion"
    assert config["replay"]["seq_len"] == 1024
    assert config["replay"]["burn_in"] == 256
    assert config["world_model"]["s5"]["blocks"] == 4
