"""Explicit numerical precision policy for learning and embodied inference."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch


class PrecisionMode(StrEnum):
    IEEE_FP32 = "ieee_fp32"
    TF32 = "tf32"
    AMP_BF16 = "amp_bf16"


_TF32_LOCK = threading.Lock()
_LIVE_CUDA_TF32: weakref.WeakKeyDictionary[object, bool] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class PrecisionPolicy:
    """Validated per-brain compute policy.

    Parameters and optimizer state remain FP32 in every supported mode. AMP is
    applied only to forward/loss construction; protected recurrence code can
    explicitly disable it around numerically sensitive operations.
    """

    mode: PrecisionMode
    device: torch.device

    @classmethod
    def from_config(cls, training: dict[str, Any], device: torch.device) -> PrecisionPolicy:
        raw = str(training.get("precision", PrecisionMode.IEEE_FP32.value))
        try:
            mode = PrecisionMode(raw)
        except ValueError as exc:
            choices = ", ".join(item.value for item in PrecisionMode)
            raise ValueError(
                f"unknown training.precision {raw!r}; expected one of {choices}"
            ) from exc
        policy = cls(mode=mode, device=device)
        policy.validate()
        return policy

    @property
    def uses_tf32(self) -> bool:
        return self.mode in (PrecisionMode.TF32, PrecisionMode.AMP_BF16)

    @property
    def uses_autocast(self) -> bool:
        return self.mode is PrecisionMode.AMP_BF16

    def validate(self) -> None:
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {self.device} was requested, but CUDA is unavailable")
        if self.mode is PrecisionMode.IEEE_FP32:
            return
        if self.device.type != "cuda":
            raise ValueError(
                f"training.precision={self.mode.value} requires a CUDA device, got {self.device}"
            )
        major, _ = torch.cuda.get_device_capability(self.device)
        if self.mode is PrecisionMode.TF32:
            if major < 8:
                raise RuntimeError(
                    "training.precision=tf32 requires an Ampere-class CUDA GPU or newer"
                )
        else:
            with torch.cuda.device(self.device):
                bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
            if major < 8 or not bf16_supported:
                raise RuntimeError(
                    "training.precision=amp_bf16 requested, but CUDA BF16 is unsupported"
                )

    def autocast(self) -> AbstractContextManager[None]:
        if not self.uses_autocast:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _set_process_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def _ensure_live_brains_allow(requested: bool) -> None:
    live_postures = set(_LIVE_CUDA_TF32.values())
    if live_postures and live_postures != {requested}:
        existing = "TF32" if True in live_postures else "IEEE FP32"
        wanted = "TF32" if requested else "IEEE FP32"
        raise ValueError(
            f"live CUDA brains require {existing}, but the new policy requests {wanted}; "
            "use one process-global TF32 posture while those brains are alive"
        )


def configure_process_precision(policies: Iterable[PrecisionPolicy]) -> None:
    """Set the process-global TF32 posture after checking it is unambiguous.

    PyTorch's TF32 controls are process-global, while autocast is lexical. A
    runtime may therefore mix FP32 and BF16 autocast only when every CUDA brain
    agrees on whether remaining FP32 GEMMs use TF32.
    """

    cuda_policies = [policy for policy in policies if policy.device.type == "cuda"]
    if not cuda_policies:
        return
    tf32_postures = {policy.uses_tf32 for policy in cuda_policies}
    if len(tf32_postures) != 1:
        modes = ", ".join(sorted({policy.mode.value for policy in cuda_policies}))
        raise ValueError(
            "CUDA brains request conflicting process-global TF32 behavior: "
            f"{modes}. Use one TF32 posture per runtime process."
        )
    for policy in cuda_policies:
        policy.validate()
    enabled = tf32_postures.pop()
    with _TF32_LOCK:
        _ensure_live_brains_allow(enabled)
        _set_process_tf32(enabled)


def register_process_precision(owner: object, policy: PrecisionPolicy) -> None:
    """Apply and retain one CUDA brain's process-global TF32 requirement."""
    if policy.device.type != "cuda":
        return
    policy.validate()
    enabled = policy.uses_tf32
    with _TF32_LOCK:
        _ensure_live_brains_allow(enabled)
        _set_process_tf32(enabled)
        _LIVE_CUDA_TF32[owner] = enabled


__all__ = [
    "PrecisionMode",
    "PrecisionPolicy",
    "configure_process_precision",
    "register_process_precision",
]
