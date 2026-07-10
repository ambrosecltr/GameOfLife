"""Resettable S5 dynamics for the Aion lineage.

The linear state transition is evaluated recurrently for embodied action and
with an associative scan for replay consolidation. The implementation follows
the S5 formulation (Smith, Warrington, and Linderman, 2023): diagonalized
continuous-time dynamics, HiPPO-LegS eigenvalue initialization, learnable
discretization steps, and nonlinear residual blocks around one MIMO SSM.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import cache
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from gol_brains.dreamer.dynamics import CategoricalLatentDynamics, DynamicsSequence
from gol_brains.dreamer.networks import mlp


@dataclass(frozen=True)
class S5DynamicsConfig:
    model_dim: int
    state_dim: int
    blocks: int
    stoch_groups: int
    stoch_classes: int
    hidden: int
    slow_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    projection_precision: str = "fp32"
    unimix: float = 0.01
    free_bits: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim < 1 or self.state_dim < 2 or self.blocks < 1:
            raise ValueError("S5 model_dim, state_dim, and blocks must be positive")
        if not 0.0 < self.slow_fraction <= 1.0:
            raise ValueError("S5 slow_fraction must be in (0, 1]")
        if not 0.0 < self.dt_min < self.dt_max:
            raise ValueError("S5 timescales require 0 < dt_min < dt_max")
        if self.projection_precision != "fp32":
            raise ValueError(
                "Aion S5 projection_precision must remain 'fp32' until target-GPU "
                "long-memory and learning parity gates pass"
            )

    @property
    def stoch_dim(self) -> int:
        return self.stoch_groups * self.stoch_classes

    @property
    def deter(self) -> int:
        # Every complex mode is stored explicitly as a real/imaginary pair.
        return self.blocks * self.state_dim * 2

    @property
    def feat_dim(self) -> int:
        return self.deter + self.stoch_dim


@cache
def _hippo_legs_frequencies(size: int) -> torch.Tensor:
    """Imaginary eigenvalues of the normal HiPPO-LegS initialization."""
    order = torch.arange(size, dtype=torch.float64)
    scale = torch.sqrt(1.0 + 2.0 * order)
    hippo = -(torch.tril(scale[:, None] * scale[None, :]) - torch.diag(order))
    rank_one = torch.sqrt(order + 0.5)
    normal = hippo + rank_one[:, None] * rank_one[None, :]
    frequencies = torch.linalg.eigvalsh(normal.to(torch.complex128) * -1j)
    return torch.real(frequencies).to(torch.float32)


S5_CHECKPOINT_FORMAT = "paired_real_v1"


def migrate_complex_s5_state(state: dict[str, torch.Tensor]) -> bool:
    """Convert the original native-complex S5 parameters to paired real keys.

    The flat recurrent state was already stored as real/imaginary pairs, so
    only parameter storage changes. Optimizer moments must be restarted by the
    caller because the parameter list and tensor layout have changed.
    """

    complex_keys = [
        key
        for key in state
        if key.endswith(".input_matrix") or key.endswith(".output_matrix")
    ]
    paired_keys = [
        key
        for key in state
        if key.endswith(".input_matrix_real")
        or key.endswith(".input_matrix_imag")
        or key.endswith(".output_matrix_real")
        or key.endswith(".output_matrix_imag")
    ]
    if complex_keys and paired_keys:
        raise ValueError("Aion checkpoint mixes native-complex and paired-real S5 parameters")
    if not complex_keys:
        return False
    for key in complex_keys:
        value = state.pop(key)
        if not value.is_complex():
            raise ValueError(f"legacy Aion parameter {key!r} is not complex")
        state[f"{key}_real"] = value.real.clone()
        state[f"{key}_imag"] = value.imag.clone()
    return True


def _fp32_recurrence_scope(device: torch.device) -> AbstractContextManager[None]:
    if device.type in ("cpu", "cuda"):
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _complex_multiply(
    left_real: torch.Tensor,
    left_imag: torch.Tensor,
    right_real: torch.Tensor,
    right_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        left_real * right_real - left_imag * right_imag,
        left_real * right_imag + left_imag * right_real,
    )


def _parallel_affine_scan(
    transition_real: torch.Tensor,
    transition_imag: torch.Tensor,
    drive_real: torch.Tensor,
    drive_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive paired-real scan of x[t] = A[t] * x[t-1] + b[t].

    Pair composition is associative, so a length-1024 sequence needs ten
    tensor stages rather than 1024 Python-level recurrent calls. This
    Hillis-Steele form intentionally uses public PyTorch operations so it
    remains portable across the project's supported PyTorch versions.
    """
    shapes = {
        transition_real.shape,
        transition_imag.shape,
        drive_real.shape,
        drive_imag.shape,
    }
    if len(shapes) != 1:
        raise ValueError("paired-real transition and drive tensors must have identical shapes")
    length = transition_real.shape[1]
    offset = 1
    composed_real = transition_real
    composed_imag = transition_imag
    composed_drive_real = drive_real
    composed_drive_imag = drive_imag
    while offset < length:
        right_real, right_imag = _complex_multiply(
            composed_real[:, offset:],
            composed_imag[:, offset:],
            composed_real[:, :-offset],
            composed_imag[:, :-offset],
        )
        carried_real, carried_imag = _complex_multiply(
            composed_real[:, offset:],
            composed_imag[:, offset:],
            composed_drive_real[:, :-offset],
            composed_drive_imag[:, :-offset],
        )
        right_drive_real = composed_drive_real[:, offset:] + carried_real
        right_drive_imag = composed_drive_imag[:, offset:] + carried_imag
        composed_real = torch.cat([composed_real[:, :offset], right_real], dim=1)
        composed_imag = torch.cat([composed_imag[:, :offset], right_imag], dim=1)
        composed_drive_real = torch.cat(
            [composed_drive_real[:, :offset], right_drive_real], dim=1
        )
        composed_drive_imag = torch.cat(
            [composed_drive_imag[:, :offset], right_drive_imag], dim=1
        )
        offset *= 2
    return composed_drive_real, composed_drive_imag


class S5SSM(nn.Module):
    """One stable diagonalized continuous-time MIMO state-space layer."""

    def __init__(
        self,
        width: int,
        state_dim: int,
        slow_fraction: float,
        dt_min: float,
        dt_max: float,
    ) -> None:
        super().__init__()
        self.width = width
        self.state_dim = state_dim
        self.slow_modes = max(1, round(state_dim * slow_fraction))

        initial_decay = torch.full((state_dim,), 0.5)
        self.raw_decay = nn.Parameter(torch.log(torch.expm1(initial_decay)))
        # Conjugate symmetry stores one mode from each pair; paired real
        # parameters retain the exact complex diagonal algebra without native
        # complex kernels or reduced-precision complex dtypes.
        self.frequency = nn.Parameter(_hippo_legs_frequencies(2 * state_dim)[:state_dim].clone())
        self.log_step = nn.Parameter(
            torch.empty(state_dim).uniform_(math.log(dt_min), math.log(dt_max))
        )

        input_scale = (2.0 * width) ** -0.5
        output_scale = (2.0 * state_dim) ** -0.5
        self.input_matrix_real = nn.Parameter(torch.randn(state_dim, width) * input_scale)
        self.input_matrix_imag = nn.Parameter(torch.randn(state_dim, width) * input_scale)
        self.output_matrix_real = nn.Parameter(torch.randn(width, state_dim) * output_scale)
        self.output_matrix_imag = nn.Parameter(torch.randn(width, state_dim) * output_scale)
        self.feedthrough = nn.Parameter(torch.ones(width))

    def _continuous_eigenvalues(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The left-half-plane parameterization keeps arbitrarily long lived
        # streams stable even after years of online optimizer updates.
        decay = F.softplus(self.raw_decay.float()) + 1e-4
        return -decay, self.frequency.float()

    def _transition(
        self, step_scale: float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eigen_real, eigen_imag = self._continuous_eigenvalues()
        delta = torch.exp(self.log_step.float())
        scale = torch.as_tensor(step_scale, device=delta.device, dtype=torch.float32)
        elapsed = delta * scale[..., None]
        magnitude = torch.exp(eigen_real * elapsed)
        angle = eigen_imag * elapsed
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

    def _discretized_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        eigen_real, eigen_imag = self._continuous_eigenvalues()
        transition_real, transition_imag = self._transition(1.0)
        numerator_real = transition_real - 1.0
        denominator = eigen_real.square() + eigen_imag.square()
        factor_real = (numerator_real * eigen_real + transition_imag * eigen_imag) / denominator
        factor_imag = (transition_imag * eigen_real - numerator_real * eigen_imag) / denominator
        return _complex_multiply(
            factor_real[:, None],
            factor_imag[:, None],
            self.input_matrix_real.float(),
            self.input_matrix_imag.float(),
        )

    def slow_mask(self) -> torch.Tensor:
        """Select the currently longest-timescale modes."""
        with torch.no_grad():
            rate = F.softplus(self.raw_decay.float()) * torch.exp(self.log_step.float())
            indices = torch.topk(rate, self.slow_modes, largest=False).indices
            mask = torch.zeros(self.state_dim, dtype=torch.bool, device=rate.device)
            mask[indices] = True
        return mask

    def step(
        self,
        signal: torch.Tensor,
        state: torch.Tensor,
        step_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _fp32_recurrence_scope(signal.device):
            signal_fp32 = signal.float()
            state_fp32 = state.float()
            state_real, state_imag = state_fp32.unbind(-1)
            transition_real, transition_imag = self._transition(step_scale)
            input_real, input_imag = self._discretized_input()
            drive_real = signal_fp32 @ input_real.T
            drive_imag = signal_fp32 @ input_imag.T
            next_real, next_imag = _complex_multiply(
                transition_real, transition_imag, state_real, state_imag
            )
            next_real = next_real + drive_real
            next_imag = next_imag + drive_imag
            output = 2.0 * (
                next_real @ self.output_matrix_real.float().T
                - next_imag @ self.output_matrix_imag.float().T
            )
            output = output + self.feedthrough.float() * signal_fp32
            return output, torch.stack([next_real, next_imag], dim=-1)

    def sequence(
        self,
        signal: torch.Tensor,
        initial_state: torch.Tensor,
        first: torch.Tensor,
        wake: torch.Tensor,
        step_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _fp32_recurrence_scope(signal.device):
            signal_fp32 = signal.float()
            initial_real, initial_imag = initial_state.float().unbind(-1)
            transition_real, transition_imag = self._transition(step_scale.float())
            input_real, input_imag = self._discretized_input()
            with torch.profiler.record_function("s5/sequence_projection_fp32"):
                drive_real = torch.einsum("blh,ph->blp", signal_fp32, input_real)
                drive_imag = torch.einsum("blh,ph->blp", signal_fp32, input_imag)

            first_mask = first[..., None] > 0.5
            wake_fast_mask = (wake[..., None] > 0.5) & ~self.slow_mask()[None, None, :]
            reset_mask = first_mask | wake_fast_mask
            transition_real = torch.where(reset_mask, 0.0, transition_real)
            transition_imag = torch.where(reset_mask, 0.0, transition_imag)

            seeded_real, seeded_imag = _complex_multiply(
                transition_real[:, 0],
                transition_imag[:, 0],
                initial_real,
                initial_imag,
            )
            drive_real = torch.cat(
                [(drive_real[:, 0] + seeded_real)[:, None], drive_real[:, 1:]], dim=1
            )
            drive_imag = torch.cat(
                [(drive_imag[:, 0] + seeded_imag)[:, None], drive_imag[:, 1:]], dim=1
            )
            with torch.profiler.record_function("s5/associative_scan_fp32"):
                states_real, states_imag = _parallel_affine_scan(
                    transition_real, transition_imag, drive_real, drive_imag
                )
            with torch.profiler.record_function("s5/sequence_readout_fp32"):
                output = 2.0 * (
                    torch.einsum(
                        "blp,hp->blh", states_real, self.output_matrix_real.float()
                    )
                    - torch.einsum(
                        "blp,hp->blh", states_imag, self.output_matrix_imag.float()
                    )
                )
            output = output + self.feedthrough.float() * signal_fp32
            return output, torch.stack([states_real, states_imag], dim=-1)


class S5Block(nn.Module):
    """Pre-normalized nonlinear residual block around one S5 SSM."""

    def __init__(self, cfg: S5DynamicsConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.model_dim)
        self.ssm = S5SSM(
            cfg.model_dim,
            cfg.state_dim,
            cfg.slow_fraction,
            cfg.dt_min,
            cfg.dt_max,
        )
        self.gate = nn.Linear(cfg.model_dim, cfg.model_dim)

    def step(
        self,
        signal: torch.Tensor,
        state: torch.Tensor,
        step_scale: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, next_state = self.ssm.step(self.norm(signal), state, step_scale)
        update = F.gelu(update) * torch.sigmoid(self.gate(update))
        return signal + update, next_state

    def sequence(
        self,
        signal: torch.Tensor,
        initial_state: torch.Tensor,
        first: torch.Tensor,
        wake: torch.Tensor,
        step_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, states = self.ssm.sequence(
            self.norm(signal), initial_state, first, wake, step_scale
        )
        update = F.gelu(update) * torch.sigmoid(self.gate(update))
        return signal + update, states


class S5Stack(nn.Module):
    def __init__(self, cfg: S5DynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(S5Block(cfg) for _ in range(cfg.blocks))

    def initial(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.deter, device=device, dtype=torch.float32)

    def _unpack(self, state: torch.Tensor) -> torch.Tensor:
        return state.float().reshape(
            *state.shape[:-1], self.cfg.blocks, self.cfg.state_dim, 2
        )

    @staticmethod
    def _pack(state: torch.Tensor) -> torch.Tensor:
        leading = state.shape[:-3]
        return state.reshape(*leading, -1)

    def reset_fast(self, state: torch.Tensor) -> torch.Tensor:
        unpacked = self._unpack(state)
        kept = []
        for index, module in enumerate(self.blocks):
            block = cast(S5Block, module)
            kept.append(unpacked[..., index, :, :] * block.ssm.slow_mask()[..., None])
        return self._pack(torch.stack(kept, dim=-3))

    def step(
        self,
        signal: torch.Tensor,
        state: torch.Tensor,
        step_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        layer_states = self._unpack(state)
        next_states: list[torch.Tensor] = []
        for index, module in enumerate(self.blocks):
            block = cast(S5Block, module)
            signal, next_state = block.step(
                signal, layer_states[..., index, :, :], step_scale
            )
            next_states.append(next_state)
        return self._pack(torch.stack(next_states, dim=-3))

    def sequence(
        self,
        signal: torch.Tensor,
        initial_state: torch.Tensor,
        first: torch.Tensor,
        wake: torch.Tensor,
        step_scale: torch.Tensor,
    ) -> torch.Tensor:
        initial_layers = self._unpack(initial_state)
        state_sequences: list[torch.Tensor] = []
        for index, module in enumerate(self.blocks):
            block = cast(S5Block, module)
            signal, states = block.sequence(
                signal, initial_layers[..., index, :, :], first, wake, step_scale
            )
            state_sequences.append(states)
        return self._pack(torch.stack(state_sequences, dim=-3))


class S5Dynamics(CategoricalLatentDynamics):
    """S5 deterministic memory plus observation-grounded categorical latents."""

    def __init__(self, cfg: S5DynamicsConfig, embed_dim: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.stoch_dim + action_dim, cfg.model_dim, bias=False),
            nn.LayerNorm(cfg.model_dim),
            nn.SiLU(),
        )
        self.stack = S5Stack(cfg)
        self.prior_net = mlp(cfg.deter, cfg.hidden, cfg.stoch_dim, layers=1)
        # Observation-only posterior is the deliberate S5WM factorization: it
        # makes replay inference parallel while the predictive prior must use
        # the recurrent lifetime state to anticipate that posterior.
        self.post_net = mlp(embed_dim, cfg.hidden, cfg.stoch_dim, layers=1)

    def initial(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stack.initial(batch, device)
        z = torch.zeros(batch, self.cfg.stoch_dim, device=device)
        return h, z

    def reset_fast(self, h: torch.Tensor) -> torch.Tensor:
        return self.stack.reset_fast(h)

    def _project_input(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.input_projection(torch.cat([z, action], dim=-1)))

    def img_step(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stack.step(self._project_input(z, action), h)
        prior = self._logits_to_probs(self.prior_net(h))
        return h, self._sample(prior), prior

    def obs_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        embed: torch.Tensor,
        step_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stack.step(self._project_input(z, action), h, step_scale)
        prior = self._logits_to_probs(self.prior_net(h))
        post = self._logits_to_probs(self.post_net(embed))
        return h, self._sample(post), post, prior

    def _sequence_inputs(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        first: torch.Tensor,
        wake: torch.Tensor,
        initial_z: torch.Tensor,
        initial_action: torch.Tensor,
    ) -> torch.Tensor:
        previous_z = torch.cat([initial_z[:, None], z[:, :-1]], dim=1)
        previous_action = torch.cat([initial_action[:, None], action[:, :-1]], dim=1)
        boundary = (first > 0.5) | (wake > 0.5)
        previous_z = torch.where(boundary[..., None], 0.0, previous_z)
        previous_action = torch.where(boundary[..., None], 0.0, previous_action)
        return self._project_input(previous_z, previous_action)

    def observe_sequence(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        first: torch.Tensor,
        wake: torch.Tensor,
        step_scale: torch.Tensor,
        burn_in: int,
    ) -> DynamicsSequence:
        batch = embed.shape[0]
        h, initial_z = self.initial(batch, embed.device)
        initial_action = torch.zeros(batch, self.action_dim, device=embed.device)

        if burn_in > 0:
            with torch.no_grad():
                burn_post = self._logits_to_probs(self.post_net(embed[:, :burn_in]))
                burn_z = self._sample(burn_post)
                burn_input = self._sequence_inputs(
                    burn_z,
                    action[:, :burn_in],
                    first[:, :burn_in],
                    wake[:, :burn_in],
                    initial_z,
                    initial_action,
                )
                burn_h = self.stack.sequence(
                    burn_input,
                    h,
                    first[:, :burn_in],
                    wake[:, :burn_in],
                    step_scale[:, :burn_in],
                )
                h = burn_h[:, -1]
                initial_z = burn_z[:, -1]
                initial_action = action[:, burn_in - 1]

        graded_embed = embed[:, burn_in:]
        graded_action = action[:, burn_in:]
        graded_first = first[:, burn_in:]
        graded_wake = wake[:, burn_in:]
        graded_scale = step_scale[:, burn_in:]
        post = self._logits_to_probs(self.post_net(graded_embed))
        z = self._sample(post)
        projected = self._sequence_inputs(
            z,
            graded_action,
            graded_first,
            graded_wake,
            initial_z,
            initial_action,
        )
        h = self.stack.sequence(projected, h, graded_first, graded_wake, graded_scale)
        prior = self._logits_to_probs(self.prior_net(h))
        return DynamicsSequence(h=h, z=z, post=post, prior=prior)


__all__ = [
    "S5_CHECKPOINT_FORMAT",
    "S5Dynamics",
    "S5DynamicsConfig",
    "S5SSM",
    "S5Stack",
    "migrate_complex_s5_state",
]
