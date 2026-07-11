"""Building blocks from the DreamerV3 recipe.

symlog/symexp squashing, twohot distributional targets, LayerNorm+SiLU MLPs,
and a running-statistics normalizer. These are the published stabilizers that
make the recipe robust without per-domain tuning — do not simplify them away.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


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
        probs = torch.softmax(logits.float(), dim=-1)
        return symexp((probs * self.edges).sum(-1))

    def loss(self, logits: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        target = self.encode(value.float())
        return -(target * torch.log_softmax(logits.float(), dim=-1)).sum(-1)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """DreamerV3-style MLP: Linear -> LayerNorm -> SiLU blocks, linear out."""
    mods: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers):
        mods += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        dim = hidden
    mods.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*mods)


class EnsembleMLP(nn.Module):
    """K single-hidden-layer MLPs (Linear -> LayerNorm -> SiLU -> Linear) as
    stacked tensors: one einsum per layer instead of one module call per
    member. Nano-scale learn() is dispatch-bound, and the Plan2Explore
    ensemble was K Python-level MLP calls at three sites per update; this is
    numerically the same computation in 2 kernels.

    Forward maps (..., in_dim) -> (K, ..., out_dim): every member sees the
    same input, exactly like the ModuleList it replaces.
    """

    def __init__(self, k: int, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.k = k
        self.hidden = hidden
        self.w1 = nn.Parameter(torch.empty(k, in_dim, hidden))
        self.b1 = nn.Parameter(torch.empty(k, hidden))
        self.ln_w = nn.Parameter(torch.ones(k, hidden))
        self.ln_b = nn.Parameter(torch.zeros(k, hidden))
        self.w2 = nn.Parameter(torch.empty(k, hidden, out_dim))
        self.b2 = nn.Parameter(torch.empty(k, out_dim))
        with torch.no_grad():
            for j in range(k):
                # Per-member nn.Linear default init (kaiming uniform computes
                # fan from (out, in) layout, so init transposed then copy).
                lin1 = torch.empty(hidden, in_dim)
                nn.init.kaiming_uniform_(lin1, a=math.sqrt(5))
                self.w1[j] = lin1.T
                bound1 = 1.0 / math.sqrt(in_dim)
                nn.init.uniform_(self.b1[j], -bound1, bound1)
                lin2 = torch.empty(out_dim, hidden)
                nn.init.kaiming_uniform_(lin2, a=math.sqrt(5))
                self.w2[j] = lin2.T
                bound2 = 1.0 / math.sqrt(hidden)
                nn.init.uniform_(self.b2[j], -bound2, bound2)

    def _per_k(self, p: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Reshape a (K, D) parameter to broadcast over (K, ..., D)."""
        return p.reshape(self.k, *([1] * (like.dim() - 2)), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("...i,kih->k...h", x, self.w1)
        h = h + self._per_k(self.b1, h)
        h = F.layer_norm(h, (self.hidden,))
        h = h * self._per_k(self.ln_w, h) + self._per_k(self.ln_b, h)
        h = F.silu(h)
        out = torch.einsum("k...h,kho->k...o", h, self.w2)
        return out + self._per_k(self.b2, out)


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF categorical sample over the last dim -> (...,) long.

    Replaces torch.distributions.Categorical on the hot paths: distribution
    objects pay construction + validation overhead on every RSSM step, which
    dominates nano-scale unrolls. Callers guarantee strictly positive probs
    (unimix), so no validation is needed; the clamp absorbs cumsum round-off.
    """
    u = torch.rand(probs.shape[:-1], device=probs.device).unsqueeze(-1)
    idx = (probs.cumsum(-1) < u).sum(-1)
    return idx.clamp_(max=probs.shape[-1] - 1)


class DiscreteDist:
    """Categorical with closed-form log-probs/entropy, no distribution object.

    probs must be strictly positive (the policy's unimix guarantees it).
    """

    def __init__(self, probs: torch.Tensor) -> None:
        self.probs = probs
        self.logits = probs.log()

    def sample(self) -> torch.Tensor:
        return sample_categorical(self.probs)

    def log_prob(self, idx: torch.Tensor) -> torch.Tensor:
        return self.logits.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> torch.Tensor:
        return -(self.probs * self.logits).sum(-1)


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


POLICY_MIN_STD = 0.1
POLICY_MAX_STD = 1.0


def bounded_policy_std(raw_std: torch.Tensor) -> torch.Tensor:
    """Map policy logits smoothly into DreamerV3's bounded standard deviation."""
    return POLICY_MIN_STD + (POLICY_MAX_STD - POLICY_MIN_STD) * torch.sigmoid(raw_std)


class TanhNormal:
    """Tanh-squashed diagonal Gaussian with log-probs (for REINFORCE).

    Closed-form Normal math instead of torch.distributions.Normal: the policy
    builds one of these per imagination step and per act(), and distribution
    objects pay construction + validation overhead that dominates at nano
    scale. Same numbers, no objects.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.mean + self.std * torch.randn_like(self.mean))

    def sample_for_reinforce(self) -> torch.Tensor:
        """Sample an action held constant by the score-function gradient."""
        return self.sample().detach()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        # atanh with clamping away from the asymptotes.
        a = action.clamp(-0.999, 0.999)
        pre = torch.atanh(a)
        log_p = (
            -0.5 * ((pre - self.mean) / self.std).pow(2) - self.std.log() - _LOG_SQRT_2PI
        ) - torch.log1p(-a.pow(2) + 1e-6)
        return log_p.sum(-1)

    def entropy(self) -> torch.Tensor:
        # pre-squash entropy (standard proxy): 0.5 * log(2*pi*e*std^2)
        return (0.5 + _LOG_SQRT_2PI + self.std.log()).sum(-1)


def percentile_scale(returns: torch.Tensor, ema: list[float], decay: float = 0.99) -> torch.Tensor:
    """DreamerV3 return normalization: EMA of the 5th-95th percentile range."""
    lo = torch.quantile(returns.detach().float(), 0.05)
    hi = torch.quantile(returns.detach().float(), 0.95)
    rng = float(hi - lo)
    ema[0] = decay * ema[0] + (1 - decay) * rng
    return returns / max(1.0, ema[0])
