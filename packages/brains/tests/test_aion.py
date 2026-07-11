"""Aion S5 architecture, lifecycle, and integration tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from gol_brains.aion.brain import AionBrain
from gol_brains.aion.s5 import (
    S5_CHECKPOINT_FORMAT,
    S5SSM,
    S5Block,
    S5Dynamics,
    S5DynamicsConfig,
    S5Stack,
)
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


@pytest.mark.parametrize("sequence_length", [1, 2, 17, 64])
def test_parallel_s5_sequence_matches_recurrent_steps(sequence_length: int) -> None:
    torch.manual_seed(7)
    cfg = tiny_s5_config()
    stack = S5Stack(cfg)
    signal = torch.randn(3, sequence_length, cfg.model_dim)
    initial = torch.randn(3, cfg.deter)
    first = torch.zeros(3, sequence_length)
    wake = torch.zeros(3, sequence_length)
    step_scale = torch.ones(3, sequence_length)
    first[0, sequence_length // 3] = 1.0
    wake_index = 2 * sequence_length // 3
    wake[1, wake_index] = 1.0
    step_scale[1, wake_index] = 6.0

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


def test_delayed_cue_survives_1024_repeated_fp32_steps() -> None:
    ssm = S5SSM(width=1, state_dim=2, slow_fraction=1.0, dt_min=1e-4, dt_max=1e-3)
    with torch.no_grad():
        ssm.raw_decay.fill_(float(torch.log(torch.expm1(torch.tensor(0.5)))))
        ssm.frequency.zero_()
        ssm.log_step.fill_(float(torch.log(torch.tensor(1e-4))))
        ssm.input_matrix_real.zero_()
        ssm.input_matrix_real[0, 0] = 1.0
        ssm.input_matrix_imag.zero_()
        ssm.output_matrix_real.zero_()
        ssm.output_matrix_real[0, 0] = 1.0
        ssm.output_matrix_imag.zero_()
        ssm.feedthrough.zero_()

    signal = torch.zeros(1, 1024, 1)
    signal[:, 0] = 1.0
    initial = torch.zeros(1, 2, 2)
    markers = torch.zeros(1, 1024)
    step_scale = torch.ones(1, 1024)

    _, parallel_states = ssm.sequence(signal, initial, markers, markers, step_scale)
    recurrent_state = initial
    recurrent_states = []
    for index in range(signal.shape[1]):
        _, recurrent_state = ssm.step(signal[:, index], recurrent_state)
        recurrent_states.append(recurrent_state)
    recurrent = torch.stack(recurrent_states, dim=1)

    torch.testing.assert_close(parallel_states, recurrent, atol=2e-7, rtol=2e-5)
    retained_ratio = parallel_states[0, -1, 0, 0] / parallel_states[0, 0, 0, 0]
    expected = torch.exp(torch.tensor(-0.5001 * 1e-4 * 1023))
    torch.testing.assert_close(retained_ratio, expected, atol=2e-6, rtol=2e-6)
    assert float(retained_ratio.detach()) < 1.0


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


def _complex_reference_step(
    ssm: S5SSM,
    signal: torch.Tensor,
    state: torch.Tensor,
    step_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decay = torch.nn.functional.softplus(ssm.raw_decay) + 1e-4
    eigenvalues = torch.complex(-decay, ssm.frequency)
    delta = torch.exp(ssm.log_step)
    base_transition = torch.exp(eigenvalues * delta)
    transition = torch.pow(base_transition, step_scale[..., None])
    input_matrix = torch.complex(ssm.input_matrix_real, ssm.input_matrix_imag)
    input_factor = (base_transition - 1.0) / eigenvalues
    discretized_input = input_factor[:, None] * input_matrix
    complex_state = torch.view_as_complex(state.contiguous())
    drive = signal.to(discretized_input.dtype) @ discretized_input.T
    next_state = transition * complex_state + drive
    output_matrix = torch.complex(ssm.output_matrix_real, ssm.output_matrix_imag)
    output = 2.0 * (next_state @ output_matrix.T).real + ssm.feedthrough * signal
    return output, torch.view_as_real(next_state)


def test_paired_real_step_matches_complex64_outputs_states_and_gradients() -> None:
    torch.manual_seed(91)
    ssm = S5SSM(width=7, state_dim=5, slow_fraction=0.6, dt_min=1e-4, dt_max=0.1)
    signal = torch.randn(3, 7, requires_grad=True)
    state = torch.randn(3, 5, 2, requires_grad=True)
    scale = torch.tensor([1.0, 7.0, 100.0])
    parameters = list(ssm.parameters())

    paired_output, paired_state = ssm.step(signal, state, scale)
    paired_grads = torch.autograd.grad(
        paired_output.square().mean() + paired_state.square().mean(),
        [signal, state, *parameters],
    )
    reference_output, reference_state = _complex_reference_step(ssm, signal, state, scale)
    reference_grads = torch.autograd.grad(
        reference_output.square().mean() + reference_state.square().mean(),
        [signal, state, *parameters],
    )

    torch.testing.assert_close(paired_output, reference_output, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(paired_state, reference_state, atol=2e-5, rtol=1e-4)
    for paired, reference in zip(paired_grads, reference_grads, strict=True):
        torch.testing.assert_close(paired, reference, atol=3e-5, rtol=1e-4)


def test_slow_transition_retains_fp32_decay_across_research_timepoints() -> None:
    ssm = S5SSM(width=1, state_dim=2, slow_fraction=1.0, dt_min=1e-4, dt_max=1e-3)
    with torch.no_grad():
        ssm.raw_decay.fill_(float(torch.log(torch.expm1(torch.tensor(0.5)))))
        ssm.frequency.zero_()
        ssm.log_step.fill_(float(torch.log(torch.tensor(1e-4))))
        ssm.input_matrix_real.zero_()
        ssm.input_matrix_imag.zero_()
        ssm.output_matrix_real.fill_(1.0)
        ssm.output_matrix_imag.zero_()
        ssm.feedthrough.zero_()
    signal = torch.zeros(5, 1)
    state = torch.zeros(5, 2, 2)
    state[..., 0] = 1.0
    timepoints = torch.tensor([100.0, 250.0, 500.0, 1024.0, 100_000.0])

    _, retained = ssm.step(signal, state, timepoints)
    expected = torch.exp(-0.5001 * 1e-4 * timepoints)

    torch.testing.assert_close(retained[:, 0, 0], expected, atol=2e-6, rtol=2e-6)
    rounded_bf16 = torch.tensor(0.9999500012, dtype=torch.bfloat16)
    assert float(rounded_bf16) == 1.0
    assert float(retained[3, 0, 0].detach()) == pytest.approx(0.9500789, rel=2e-5)
    assert float(retained[-1, 0, 0].detach()) < 0.01


def test_protected_recurrence_stays_fp32_under_bf16_autocast() -> None:
    torch.manual_seed(92)
    ssm = S5SSM(width=8, state_dim=4, slow_fraction=0.5, dt_min=1e-4, dt_max=0.1)
    signal = torch.randn(2, 8)
    state = torch.randn(2, 4, 2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, next_state = ssm.step(signal, state)
    assert output.dtype is torch.float32
    assert next_state.dtype is torch.float32
    assert all(parameter.dtype is torch.float32 for parameter in ssm.parameters())


def test_s5_dense_gate_autocasts_while_recurrent_state_stays_fp32() -> None:
    cfg = tiny_s5_config()
    stack = S5Stack(cfg)
    signal = torch.randn(2, cfg.model_dim)
    state = stack.initial(2, torch.device("cpu"))
    gate_dtypes: list[torch.dtype] = []
    block = stack.blocks[0]
    assert isinstance(block, S5Block)
    handle = block.gate.register_forward_hook(
        lambda _module, _inputs, output: gate_dtypes.append(output.dtype)
    )
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            next_state = stack.step(signal, state)
    finally:
        handle.remove()

    assert gate_dtypes == [torch.bfloat16]
    assert next_state.dtype is torch.float32


def test_1024_graded_steps_after_burn_in_have_finite_gradients() -> None:
    torch.manual_seed(93)
    cfg = S5DynamicsConfig(
        model_dim=4,
        state_dim=2,
        blocks=1,
        stoch_groups=1,
        stoch_classes=2,
        hidden=4,
        dt_min=1e-4,
        dt_max=0.1,
    )
    dynamics = S5Dynamics(cfg, embed_dim=3, action_dim=2)
    embed = torch.randn(1, 1280, 3, requires_grad=True)
    action = torch.randn(1, 1280, 2)
    markers = torch.zeros(1, 1280)
    sequence = dynamics.observe_sequence(
        embed, action, markers, markers, torch.ones_like(markers), burn_in=256
    )
    loss = sequence.h.square().mean() + sequence.prior.square().mean()
    loss.backward()  # type: ignore[no-untyped-call]
    assert embed.grad is not None and bool(torch.isfinite(embed.grad).all())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in dynamics.parameters()
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


def test_suspended_aion_wake_preserves_slow_context_but_severs_affect() -> None:
    cfg = {**TINY_AION, "reward": {"blackout": "suspended"}}
    brain = AionBrain(cfg, seed=19)
    rng = np.random.default_rng(19)
    brain.act(fake_obs(rng))
    assert brain._prev_drive is not None
    before = brain.h.clone()

    brain.wake(dormant_steps=100)

    torch.testing.assert_close(brain.h, brain.s5.reset_fast(before))
    assert brain._prev_drive is None
    assert brain._prev_via is None
    assert brain._prev_integrity is None
    assert brain._stream_wake and brain._step_scale == 101.0


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


def test_tiny_aion_02_stack_learns_and_roundtrips() -> None:
    cfg = {
        **TINY_AION,
        "reward": {
            "homeostasis": "drive",
            "imagined_homeostasis": "proprio",
            "blackout": "suspended",
            "death_terminal": True,
            "viability": {
                "scale": 0.0,
                "floor": 0.0,
                "barrier_cap": 4.0,
                "total_cap": 4.0,
            },
            "wellbeing": {"weight": 0.25, "comfort_decay": 1.0},
            "pain": {"weight": 5.0, "event_loss_weight": 8.0},
        },
        "actor_critic": {"imagination_horizon": 3, "vector_critic": True},
        "temporal_skills": {"enabled": False},
    }
    brain = AionBrain(cfg, seed=20)
    rng = np.random.default_rng(20)
    for index in range(24):
        obs = fake_obs(rng)
        if index % 6 == 0:
            obs["events"][1] = 1.0
        brain.act(obs)
    metrics = brain.learn()
    assert metrics is not None
    assert np.isfinite(metrics["wellbeing"])
    assert np.isfinite(metrics["affect_pain"])
    assert np.isfinite(metrics["loss_damage"])

    restored = AionBrain(cfg, seed=21)
    restored.load_state_dict(brain.state_dict())
    assert restored.affect_names == brain.affect_names
    assert restored.experience_count() == brain.experience_count()


def test_aion_learns_and_roundtrips_checkpoint() -> None:
    brain = AionBrain(TINY_AION, seed=12)
    rng = np.random.default_rng(12)
    for _ in range(24):
        brain.act(fake_obs(rng))
    metrics = brain.learn()
    assert metrics is not None and np.isfinite(metrics["loss_model"])
    assert metrics["context_steps"] == 8.0
    assert metrics["graded_timepoints_per_second"] > 0.0
    modules = [brain.wm, brain.actor, brain.critic, brain.critic_ema]
    assert all(
        parameter.dtype is torch.float32 for module in modules for parameter in module.parameters()
    )
    optimizers: list[torch.optim.Optimizer] = [
        brain.opt_model,
        brain.opt_actor,
        brain.opt_critic,
    ]
    if brain.opt_model_muon is not None:
        optimizers.append(brain.opt_model_muon)
    if brain.opt_skill is not None:
        optimizers.append(brain.opt_skill)
    for optimizer in optimizers:
        for optimizer_state in optimizer.state.values():
            for value in optimizer_state.values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    assert value.dtype is torch.float32

    state = brain.state_dict()
    assert state["brain_family"] == "aion"
    fresh = AionBrain(TINY_AION, seed=13)
    fresh.load_state_dict(state)
    assert len(fresh.buffer) == len(brain.buffer)
    torch.testing.assert_close(fresh.h, brain.h)
    assert fresh.pending_update_credit() == pytest.approx(brain.pending_update_credit())
    for expected, actual in zip(brain.wm.parameters(), fresh.wm.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def _legacy_complex_aion_state(brain: AionBrain) -> dict[str, Any]:
    state = brain.state_dict()
    wm = state["wm"]
    assert isinstance(wm, dict)
    prefixes = [
        key.removesuffix("_real")
        for key in list(wm)
        if key.endswith(".input_matrix_real") or key.endswith(".output_matrix_real")
    ]
    for prefix in prefixes:
        real = wm.pop(f"{prefix}_real")
        imag = wm.pop(f"{prefix}_imag")
        assert isinstance(real, torch.Tensor) and isinstance(imag, torch.Tensor)
        wm[prefix] = torch.complex(real, imag)
    state.pop("aion_s5_format")
    return state


def test_native_complex_aion_checkpoint_migrates_without_state_ambiguity() -> None:
    brain = AionBrain(TINY_AION, seed=121)
    rng = np.random.default_rng(121)
    for _ in range(24):
        brain.act(fake_obs(rng))
    legacy = _legacy_complex_aion_state(brain)

    restored = AionBrain(TINY_AION, seed=122)
    restored.load_state_dict(legacy)

    torch.testing.assert_close(restored.h, brain.h)
    for expected, actual in zip(brain.wm.parameters(), restored.wm.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert restored.state_dict()["aion_s5_format"] == S5_CHECKPOINT_FORMAT

    torch.manual_seed(991)
    expected_metrics = brain.learn()
    torch.manual_seed(991)
    actual_metrics = restored.learn()
    assert expected_metrics is not None and actual_metrics is not None
    assert actual_metrics["loss_model"] == pytest.approx(expected_metrics["loss_model"], rel=2e-5)
    for expected, actual in zip(brain.wm.parameters(), restored.wm.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_paired_real_checkpoint_requires_explicit_format_marker() -> None:
    brain = AionBrain(TINY_AION, seed=123)
    state = brain.state_dict()
    state.pop("aion_s5_format")
    with pytest.raises(ValueError, match="missing aion_s5_format"):
        AionBrain(TINY_AION, seed=124).load_state_dict(state)


def test_async_inference_snapshot_roundtrips_independently_of_training_weights() -> None:
    cfg = {**TINY_AION, "training": {"async_inference": True, "publish_every": 4}}
    brain = AionBrain(cfg, seed=125)
    assert brain._inference is not None
    with torch.no_grad():
        next(brain.wm.encoder.parameters()).add_(1.0)
    state = brain.state_dict()

    restored = AionBrain(cfg, seed=126)
    restored.load_state_dict(state)

    assert restored._inference is not None
    for expected, actual in zip(
        brain._inference.parameters(), restored._inference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    training_parameter = next(restored.wm.encoder.parameters())
    inference_parameter = next(restored._inference.encoder.parameters())
    assert not torch.equal(training_parameter, inference_parameter)


def test_lineage_checkpoint_identity_is_enforced() -> None:
    aion = AionBrain(TINY_AION, seed=14)
    dreamer = DreamerBrain(
        {
            "kind": "dreamer",
            "preset": "nano",
            "replay": {"capacity": 64, "batch_size": 2, "seq_len": 4, "warmup_steps": 8},
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
        "replay": {"capacity": 64, "batch_size": 2, "seq_len": 4, "warmup_steps": 8},
    }
    brain = DreamerBrain(cfg, seed=18)
    brain.act(fake_obs(np.random.default_rng(18)))
    state = brain.state_dict()
    for key in (
        "brain_family",
        "stream_wake",
        "step_scale",
        "precision",
        "learning_contract",
        "published_updates",
        "inference",
        "schedule_credit_origin",
        "dropped_update_credit",
    ):
        state.pop(key)
    state["buffer"].pop("wake")
    state["buffer"].pop("step_scale")

    restored = DreamerBrain(cfg, seed=19)
    restored.load_state_dict(state)
    assert len(restored.buffer) == 1
    assert restored.buffer.wake[0] == 0
    assert restored.buffer.step_scale[0] == 1.0
    assert restored.pending_update_credit() == 0.0


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
