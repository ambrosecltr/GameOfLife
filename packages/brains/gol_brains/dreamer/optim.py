"""Muon: momentum + Newton-Schulz orthogonalized updates for 2D weights.

Vendored minimal implementation (Jordan et al. 2024, github.com/KellerJordan/Muon)
so the ablation carries no external dependency. Muon conditions each weight
matrix's update to be approximately orthogonal, which on small dense nets has
repeatedly measured ~1.5-2x data-efficiency over Adam — worth an ablation for
a nano world model that must converge inside one life.

Scope: exactly-2D parameters only (Linear/GRU weight matrices). Biases,
LayerNorms, and the stacked 3D ensemble tensors stay on Adam — the standard
Muon recipe, matching the reference implementation's guidance.
"""

from __future__ import annotations

from typing import Any

import torch


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a matrix via quintic Newton-Schulz.

    Coefficients from the reference implementation: tuned to maximize slope at
    zero rather than to converge singular values exactly to 1 (they oscillate
    in ~[0.7, 1.2], which does not hurt and buys fewer iterations).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        s = x @ x.T
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Momentum SGD whose 2D updates are orthogonalized by Newton-Schulz."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov, "ns_steps": ns_steps}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError("Muon handles exactly-2D parameters; route others to Adam")

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        assert closure is None, "Muon does not support closures"
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1.0 - momentum)
                update = p.grad.lerp(buf, momentum) if group["nesterov"] else buf
                update = newton_schulz(update.float(), steps=group["ns_steps"]).to(p.dtype)
                # Scale so the RMS of the update matches Adam conventions
                # (reference implementation's max(1, rows/cols)^0.5 factor).
                update = update * max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update, alpha=-lr)
