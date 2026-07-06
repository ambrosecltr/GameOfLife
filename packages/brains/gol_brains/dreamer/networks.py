"""Building blocks from the DreamerV3 recipe.

symlog/symexp squashing, twohot distributional targets, LayerNorm+SiLU MLPs,
and a running-statistics normalizer. These are the published stabilizers that
make the recipe robust without per-domain tuning — do not simplify them away.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class TwoHot:
    """Twohot encoding over symlog-spaced bins (DreamerV3 reward/value heads)."""

    def __init__(self, low: float = -20.0, high: float = 20.0, bins: int = 41) -> None:
        self.bins = bins
        self.edges = torch.linspace(low, high, bins)

    def to(self, device: torch.device) -> TwoHot:
        self.edges = self.edges.to(device)
        return self

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """value (...,) -> twohot target (..., bins), in symlog space."""
        x = symlog(value)
        x = x.clamp(self.edges[0], self.edges[-1])
        idx = torch.searchsorted(self.edges, x.detach()).clamp(1, self.bins - 1)
        lo, hi = self.edges[idx - 1], self.edges[idx]
        w_hi = ((x - lo) / (hi - lo)).clamp(0, 1)
        target = torch.zeros(*x.shape, self.bins, device=x.device)
        target.scatter_(-1, (idx - 1).unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        target.scatter_add_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1))
        return target

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """logits (..., bins) -> expected value (...,), back through symexp."""
        probs = torch.softmax(logits, dim=-1)
        return symexp((probs * self.edges).sum(-1))

    def loss(self, logits: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        target = self.encode(value)
        return -(target * torch.log_softmax(logits, dim=-1)).sum(-1)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """DreamerV3-style MLP: Linear -> LayerNorm -> SiLU blocks, linear out."""
    mods: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers):
        mods += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        dim = hidden
    mods.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*mods)


class RunningMeanStd:
    """Numerically stable running statistics (Welford), for reward scaling.

    anchor > 0 freezes the statistics once `count` reaches it: the scale
    calibrates on early life and then holds, so a signal that genuinely
    decays reads as decayed instead of being re-normalized toward its own
    shrinkage (the round-008 hedonic-treadmill finding). 0 = legacy
    lifetime statistics.
    """

    def __init__(self, eps: float = 1e-8, anchor: float = 0.0) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.anchor = anchor

    def update(self, x: torch.Tensor) -> None:
        if self.anchor > 0 and self.count >= self.anchor:
            return
        flat = x.detach().reshape(-1)
        if flat.numel() == 0:
            return
        batch_mean = float(flat.mean())
        batch_var = float(flat.var(unbiased=False))
        n = flat.numel()
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m_a = self.var * self.count
        m_b = batch_var * n
        self.var = (m_a + m_b + delta**2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (self.var**0.5 + 1e-8)

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


class TanhNormal:
    """Tanh-squashed diagonal Gaussian with log-probs (for REINFORCE)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.base = torch.distributions.Normal(mean, std)

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.base.sample())

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        # atanh with clamping away from the asymptotes.
        a = action.clamp(-0.999, 0.999)
        pre = torch.atanh(a)
        log_p = self.base.log_prob(pre) - torch.log1p(-a.pow(2) + 1e-6)
        return log_p.sum(-1)

    def entropy(self) -> torch.Tensor:
        return self.base.entropy().sum(-1)  # pre-squash entropy (standard proxy)


def percentile_scale(returns: torch.Tensor, ema: list[float], decay: float = 0.99) -> torch.Tensor:
    """DreamerV3 return normalization: EMA of the 5th-95th percentile range."""
    lo = torch.quantile(returns.detach().float(), 0.05)
    hi = torch.quantile(returns.detach().float(), 0.95)
    rng = float(hi - lo)
    ema[0] = decay * ema[0] + (1 - decay) * rng
    return returns / max(1.0, ema[0])
