"""The RSSM: recurrent state-space model with categorical stochastic latents.

DreamerV3 recipe: GRU deterministic path, groups-of-classes one-hot latents
with straight-through gradients, 1% unimix, KL balancing with free bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gol_brains.dreamer.networks import mlp


@dataclass(frozen=True)
class RSSMConfig:
    deter: int = 256
    stoch_groups: int = 16
    stoch_classes: int = 16
    hidden: int = 256
    unimix: float = 0.01
    free_bits: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1

    @property
    def stoch_dim(self) -> int:
        return self.stoch_groups * self.stoch_classes

    @property
    def feat_dim(self) -> int:
        return self.deter + self.stoch_dim


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig, embed_dim: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.cell = nn.GRUCell(cfg.stoch_dim + action_dim, cfg.deter)
        self.prior_net = mlp(cfg.deter, cfg.hidden, cfg.stoch_dim, layers=1)
        self.post_net = mlp(cfg.deter + embed_dim, cfg.hidden, cfg.stoch_dim, layers=1)

    def initial(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.cfg.deter, device=device)
        z = torch.zeros(batch, self.cfg.stoch_dim, device=device)
        return h, z

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        g, c = self.cfg.stoch_groups, self.cfg.stoch_classes
        probs = torch.softmax(logits.view(*logits.shape[:-1], g, c), dim=-1)
        return (1 - self.cfg.unimix) * probs + self.cfg.unimix / c

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Straight-through one-hot sample; returns flat (..., stoch_dim)."""
        idx = torch.distributions.Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.cfg.stoch_classes).to(probs.dtype)
        sample = onehot + probs - probs.detach()
        return sample.flatten(-2)

    def _step_deter(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.cell(torch.cat([z, action], dim=-1), h)

    def img_step(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One prior (imagination) step: returns (h', z', prior_probs)."""
        h = self._step_deter(h, z, action)
        probs = self._logits_to_probs(self.prior_net(h))
        return h, self._sample(probs), probs

    def obs_step(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor, embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One posterior step: returns (h', z_post, post_probs, prior_probs)."""
        h = self._step_deter(h, z, action)
        prior_probs = self._logits_to_probs(self.prior_net(h))
        post_probs = self._logits_to_probs(self.post_net(torch.cat([h, embed], dim=-1)))
        return h, self._sample(post_probs), post_probs, prior_probs

    def kl_loss(self, post: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """Balanced KL with free bits; inputs are probs (..., groups, classes)."""

        def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(-1).sum(-1)

        dyn = kl(post.detach(), prior).clamp(min=self.cfg.free_bits)
        rep = kl(post, prior.detach()).clamp(min=self.cfg.free_bits)
        return self.cfg.dyn_scale * dyn + self.cfg.rep_scale * rep

    @staticmethod
    def feat(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)
