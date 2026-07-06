"""Learning-progress curiosity: interest as the derivative of competence.

Oudeyer-style intrinsic motivation (Intelligent Adaptive Curiosity): partition
experience into regions, track how fast the world model's prediction error is
*falling* in each, and reward visiting regions where it falls fastest. The
level of error is not interesting — mastered regions (error low and flat) and
unlearnable ones (error high and flat: noise, other minds) both yield zero
progress, so attention flows to the learnable frontier and moves on as each
niche is mastered. Which niche is learnable depends on what has already been
learned, so tiny early differences compound into individual interests.

Two pieces, both checkpointable:

- OnlineRegions: streaming k-means over RSSM feature space. What counts as
  "an activity" is carved by the agent's own representation, not a designer
  taxonomy; centroids drift slowly as the representation trains.
- LearningProgress: per-region fast/slow EMAs of prediction error. Their gap
  (optionally relative to the slow level, making progress scale-free across
  regions) is the LP reward for states assigned to that region — and it is a
  pure function of region index, so imagination can query it.
"""

from __future__ import annotations

from typing import Any

import torch


class OnlineRegions:
    """Streaming k-means in latent space: assign samples, drift centroids."""

    def __init__(self, n: int, dim: int, lr: float, device: torch.device) -> None:
        self.n = n
        self.lr = lr
        self.device = device
        self.centroids = torch.zeros(n, dim, device=device)
        self.initialized = False

    def assign(self, feat: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid region index for feat (..., dim) -> (...,) long."""
        if not self.initialized:
            return torch.zeros(feat.shape[:-1], dtype=torch.long, device=feat.device)
        flat = feat.detach().reshape(-1, feat.shape[-1])
        idx = torch.cdist(flat, self.centroids).argmin(-1)
        return idx.view(feat.shape[:-1])

    def adapt(self, feat: torch.Tensor) -> torch.Tensor:
        """Assign a batch (S, dim) and move its centroids toward it.

        First batch seeds the centroids from its own samples; after that each
        present centroid takes a small step toward the mean of its members, so
        the partition tracks the (slowly) drifting latent space.
        """
        flat = feat.detach().reshape(-1, feat.shape[-1])
        if not self.initialized:
            if flat.shape[0] < self.n:
                return torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
            pick = torch.randperm(flat.shape[0], device=flat.device)[: self.n]
            self.centroids = flat[pick].clone()
            self.initialized = True
        idx = torch.cdist(flat, self.centroids).argmin(-1)
        ones = torch.zeros(self.n, device=flat.device).index_add_(
            0, idx, torch.ones_like(idx, dtype=flat.dtype)
        )
        sums = torch.zeros_like(self.centroids).index_add_(0, idx, flat)
        present = ones > 0
        mean = sums[present] / ones[present].unsqueeze(-1)
        self.centroids[present] = (1 - self.lr) * self.centroids[present] + self.lr * mean
        return idx

    def state_dict(self) -> dict[str, Any]:
        return {"centroids": self.centroids.cpu(), "initialized": self.initialized}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.centroids = torch.as_tensor(state["centroids"], device=self.device)
        self.initialized = bool(state["initialized"])


class LearningProgress:
    """Per-region fast/slow error EMAs; LP = how fast error is falling."""

    def __init__(self, n: int, fast: float, slow: float, relative: bool) -> None:
        self.n = n
        self.alpha_fast = fast
        self.alpha_slow = slow
        self.relative = relative
        self.fast = torch.zeros(n)
        self.slow = torch.zeros(n)
        self.count = torch.zeros(n)

    def update(self, idx: torch.Tensor, err: torch.Tensor) -> None:
        """Fold a batch of per-sample errors into their regions' EMAs."""
        idx = idx.detach().reshape(-1).cpu()
        err = err.detach().reshape(-1).float().cpu()
        ones = torch.zeros(self.n).index_add_(0, idx, torch.ones_like(err))
        sums = torch.zeros(self.n).index_add_(0, idx, err)
        present = ones > 0
        mean = sums[present] / ones[present]
        fresh = present & (self.count == 0)
        # A region's first sight seeds both EMAs at its error: LP starts at
        # zero rather than mistaking initial ignorance for progress.
        self.fast[fresh] = self.slow[fresh] = mean[(self.count == 0)[present]]
        seen = present & ~fresh
        m_seen = mean[~(self.count == 0)[present]]
        self.fast[seen] += self.alpha_fast * (m_seen - self.fast[seen])
        self.slow[seen] += self.alpha_slow * (m_seen - self.slow[seen])
        self.count += ones

    def lp(self) -> torch.Tensor:
        """(n,) learning progress per region; only falling error counts."""
        raw = (self.slow - self.fast).clamp(min=0.0)
        if self.relative:
            raw = raw / self.slow.clamp(min=1e-3)
        return raw

    def reward(self, idx: torch.Tensor) -> torch.Tensor:
        """LP of each sample's region: idx (...,) long -> (...,) float."""
        return self.lp().to(idx.device)[idx]

    def regions_seen(self) -> int:
        return int((self.count > 0).sum())

    def state_dict(self) -> dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.fast = torch.as_tensor(state["fast"]).float()
        self.slow = torch.as_tensor(state["slow"]).float()
        self.count = torch.as_tensor(state["count"]).float()
