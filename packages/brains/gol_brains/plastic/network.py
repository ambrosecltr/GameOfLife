"""The plastic network: encoder + GRU core + action readout, every content
transform carrying an innate `W_slow` (evolved, fixed within a life) plus a
plastic `W_fast` (starts at zero, adapts online via a three-factor Hebbian
rule). "Fully plastic" (proposal 002, round A decision) is realised as *every
content transform is plastic*; the GRU's sigmoid gates stay innate control
structure, since Hebbian plasticity on a gating nonlinearity is not well-defined
and would wreck stability.

No gradients anywhere: all parameters are `requires_grad=False` and the fast
weights move only through the local rule in `consolidate`. The eligibility trace
accumulates pre⊗post each step; a scalar neuromodulator `M` gates how much of it
consolidates, with a one-step delay so `M(t)` (the outcome) credits the activity
that produced it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasticLinear(nn.Module):
    """`y = (W_slow + W_fast) x + b`, with a neuromodulated Hebbian `W_fast`.

    `W_slow`/`b` are innate (evolved across lives, fixed within one). `W_fast`
    starts at zero and follows `ΔW_fast = alpha·M·trace − decay·W_fast`, clipped
    to `w_clip` to keep a fully-plastic net from diverging. When `plastic` is
    False (the frozen-net control, `alpha: 0`) the fast weights never move.
    """

    W_fast: torch.Tensor
    trace: torch.Tensor

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        alpha: float,
        tau: float,
        decay: float,
        w_clip: float,
        plastic: bool,
        rng: np.random.Generator,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        w = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * (
            init_scale / math.sqrt(in_dim)
        )
        self.W_slow = nn.Parameter(torch.from_numpy(w), requires_grad=False)
        self.b_slow = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
        self.register_buffer("W_fast", torch.zeros(out_dim, in_dim))
        self.register_buffer("trace", torch.zeros(out_dim, in_dim))
        self.alpha = float(alpha)
        self.inv_tau = 1.0 / float(tau)
        self.decay = float(decay)
        self.w_clip = float(w_clip)
        self.plastic = bool(plastic) and self.alpha > 0.0
        self._pre: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._pre = x.detach().flatten()
        return F.linear(x, self.W_slow + self.W_fast, self.b_slow)

    def accumulate(self, post: torch.Tensor) -> None:
        """Fold this step's pre⊗post into the eligibility trace (leaky)."""
        if not self.plastic or self._pre is None:
            return
        outer = torch.outer(post.detach().flatten(), self._pre)
        self.trace.mul_(1.0 - self.inv_tau).add_(outer, alpha=self.inv_tau)

    def consolidate(self, m: float) -> None:
        """Apply the neuromodulator: consolidate the trace, decay toward zero."""
        if not self.plastic:
            return
        self.W_fast.add_(self.alpha * m * self.trace - self.decay * self.W_fast)
        if self.w_clip > 0.0:
            self.W_fast.clamp_(-self.w_clip, self.w_clip)

    def reset_fast(self) -> None:
        self.W_fast.zero_()
        self.trace.zero_()

    def reset_trace(self) -> None:
        self.trace.zero_()

    def fast_norm(self) -> float:
        return float(self.W_fast.abs().mean()) if self.plastic else 0.0


class PlasticGRUCell(nn.Module):
    """A GRU cell whose *candidate* transform is plastic and whose reset/update
    gates are innate. The candidate `n = tanh(W_in·x + r⊙(W_hn·h))` carries the
    content the agent can learn within a life; the gates are fixed control
    structure. Two plastic linears feed the candidate; the gate weights are
    innate `nn.Linear`s (no grad).
    """

    def __init__(self, dim: int, *, plastic_kw: dict[str, Any], rng: np.random.Generator) -> None:
        super().__init__()
        self.dim = dim
        # innate gates: input is concat[x, h] -> gate; kaiming-ish innate init.
        self.gate_r = nn.Linear(2 * dim, dim)
        self.gate_z = nn.Linear(2 * dim, dim)
        for lin in (self.gate_r, self.gate_z):
            lin.weight.requires_grad_(False)
            lin.bias.requires_grad_(False)
        self.in_cand = PlasticLinear(dim, dim, rng=rng, **plastic_kw)
        self.hid_cand = PlasticLinear(dim, dim, rng=rng, **plastic_kw)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xh = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.gate_r(xh))
        z = torch.sigmoid(self.gate_z(xh))
        n = torch.tanh(self.in_cand(x) + r * self.hid_cand(h))
        h_new = (1.0 - z) * n + z * h
        return h_new, n

    def accumulate(self, cand: torch.Tensor) -> None:
        self.in_cand.accumulate(cand)
        self.hid_cand.accumulate(cand)

    def consolidate(self, m: float) -> None:
        self.in_cand.consolidate(m)
        self.hid_cand.consolidate(m)

    def reset_fast(self) -> None:
        self.in_cand.reset_fast()
        self.hid_cand.reset_fast()

    def reset_trace(self) -> None:
        self.in_cand.reset_trace()
        self.hid_cand.reset_trace()

    def fast_norm(self) -> float:
        return 0.5 * (self.in_cand.fast_norm() + self.hid_cand.fast_norm())


class PlasticNet(nn.Module):
    """encoder (plastic) → GRU candidate (plastic) → readout (plastic).

    `forward` returns the raw readout, the new hidden state, and the activations
    each plastic layer needs for its eligibility trace. The caller accumulates
    the trace *after* it has chosen the discrete action (so the readout's
    discrete credit lands on the taken gripper mode), then consolidates with the
    neuromodulator on the following step.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        *,
        plastic_kw: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        self.encoder = PlasticLinear(in_dim, hidden, rng=rng, **plastic_kw)
        self.gru = PlasticGRUCell(hidden, plastic_kw=plastic_kw, rng=rng)
        self.readout = PlasticLinear(hidden, out_dim, rng=rng, **plastic_kw)
        self.hidden = hidden

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e = F.silu(F.layer_norm(self.encoder(x), (self.hidden,)))
        h_new, cand = self.gru(e, h)
        out = self.readout(h_new)
        return out, h_new, e, cand

    def accumulate(self, e: torch.Tensor, cand: torch.Tensor, readout_post: torch.Tensor) -> None:
        self.encoder.accumulate(e)
        self.gru.accumulate(cand)
        self.readout.accumulate(readout_post)

    def consolidate(self, m: float) -> None:
        self.encoder.consolidate(m)
        self.gru.consolidate(m)
        self.readout.consolidate(m)

    def reset_fast(self) -> None:
        self.encoder.reset_fast()
        self.gru.reset_fast()
        self.readout.reset_fast()

    def reset_trace(self) -> None:
        self.encoder.reset_trace()
        self.gru.reset_trace()
        self.readout.reset_trace()

    def fast_norm(self) -> float:
        return (self.encoder.fast_norm() + self.gru.fast_norm() + self.readout.fast_norm()) / 3.0
