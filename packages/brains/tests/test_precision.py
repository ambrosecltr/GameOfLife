"""Precision modes are explicit, observable, and fail closed."""

import gc
import weakref

import gol_brains.precision as precision
import pytest
import torch
from gol_brains.aion.brain import AionBrain
from gol_brains.precision import (
    PrecisionMode,
    PrecisionPolicy,
    configure_process_precision,
    register_process_precision,
)

TINY_AION = {
    "kind": "aion",
    "preset": "nano",
    "world_model": {"s5": {"model_dim": 16, "state_dim": 4, "blocks": 1}},
    "replay": {"capacity": 32, "batch_size": 1, "seq_len": 4, "warmup_steps": 8},
    "actor_critic": {"imagination_horizon": 2},
}


def test_reference_fp32_is_the_backward_compatible_default() -> None:
    brain = AionBrain(TINY_AION, seed=201)
    assert brain.precision.mode is PrecisionMode.IEEE_FP32
    state = brain.state_dict()
    assert state["precision"] == "ieee_fp32"


@pytest.mark.parametrize("mode", ["tf32", "amp_bf16"])
def test_cuda_precision_modes_fail_on_cpu(mode: str) -> None:
    cfg = {**TINY_AION, "training": {"precision": mode}}
    with pytest.raises(ValueError, match="requires a CUDA device"):
        AionBrain(cfg, seed=202, device="cpu")


def test_unknown_precision_mode_fails_explicitly() -> None:
    cfg = {**TINY_AION, "training": {"precision": "automatic"}}
    with pytest.raises(ValueError, match="unknown training.precision"):
        AionBrain(cfg, seed=203)


def test_process_runtime_rejects_conflicting_tf32_postures() -> None:
    policies = [
        PrecisionPolicy(PrecisionMode.IEEE_FP32, torch.device("cuda")),
        PrecisionPolicy(PrecisionMode.AMP_BF16, torch.device("cuda")),
    ]
    with pytest.raises(ValueError, match="conflicting process-global TF32"):
        configure_process_precision(policies)


def test_live_cuda_brain_prevents_process_precision_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Owner:
        pass

    monkeypatch.setattr(precision, "_LIVE_CUDA_TF32", weakref.WeakKeyDictionary())
    monkeypatch.setattr(PrecisionPolicy, "validate", lambda self: None)
    monkeypatch.setattr(precision, "_set_process_tf32", lambda enabled: None)
    tf32 = PrecisionPolicy(PrecisionMode.AMP_BF16, torch.device("cuda:0"))
    ieee = PrecisionPolicy(PrecisionMode.IEEE_FP32, torch.device("cuda:1"))
    first = Owner()
    second = Owner()

    register_process_precision(first, tf32)
    with pytest.raises(ValueError, match="live CUDA brains require TF32"):
        configure_process_precision([ieee])
    with pytest.raises(ValueError, match="live CUDA brains require TF32"):
        register_process_precision(second, ieee)

    del first
    gc.collect()
    register_process_precision(second, ieee)


def test_checkpoint_precision_change_requires_a_deliberate_migration() -> None:
    brain = AionBrain(TINY_AION, seed=204)
    state = brain.state_dict()
    state["precision"] = "tf32"
    with pytest.raises(ValueError, match="checkpoint precision"):
        AionBrain(TINY_AION, seed=205).load_state_dict(state)


def test_checkpoint_learning_contract_cannot_change_silently() -> None:
    brain = AionBrain(TINY_AION, seed=206)
    state = brain.state_dict()
    state["learning_contract"]["train_ratio"] = 9.0
    with pytest.raises(ValueError, match="learning contract"):
        AionBrain(TINY_AION, seed=207).load_state_dict(state)
