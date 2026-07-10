"""Shared categorical-latent dynamics primitives.

Dreamer's GRU RSSM and Aion's S5 dynamics use the same stochastic latent
contract. Keeping that contract here lets the organism stack consume either
backbone without duplicating its world-model, affect, or actor-critic logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from gol_brains.dreamer.networks import sample_categorical


class CategoricalDynamicsConfig(Protocol):
    @property
    def deter(self) -> int: ...

    @property
    def stoch_groups(self) -> int: ...

    @property
    def stoch_classes(self) -> int: ...

    @property
    def stoch_dim(self) -> int: ...

    @property
    def feat_dim(self) -> int: ...

    @property
    def unimix(self) -> float: ...

    @property
    def free_bits(self) -> float: ...

    @property
    def dyn_scale(self) -> float: ...

    @property
    def rep_scale(self) -> float: ...


@dataclass(frozen=True)
class DynamicsSequence:
    """Posterior dynamics over the gradient-carrying part of a replay window."""

    h: torch.Tensor
    z: torch.Tensor
    post: torch.Tensor
    prior: torch.Tensor


class CategoricalLatentDynamics(nn.Module):
    """The categorical stochastic-state contract shared by both lineages."""

    cfg: CategoricalDynamicsConfig

    def initial(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def img_step(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def obs_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        embed: torch.Tensor,
        step_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def observe_sequence(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        first: torch.Tensor,
        wake: torch.Tensor,
        step_scale: torch.Tensor,
        burn_in: int,
    ) -> DynamicsSequence:
        raise NotImplementedError

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        groups = self.cfg.stoch_groups
        classes = self.cfg.stoch_classes
        # Categorical probabilities feed sampling, KL, and persistent latent
        # state. Keep their normalization in FP32 even when dense logits were
        # produced under BF16 autocast.
        logits_fp32 = logits.float().view(*logits.shape[:-1], groups, classes)
        probs = torch.softmax(logits_fp32, dim=-1)
        return (1.0 - self.cfg.unimix) * probs + self.cfg.unimix / classes

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        indices = sample_categorical(probs)
        one_hot = F.one_hot(indices, self.cfg.stoch_classes).to(probs.dtype)
        straight_through = one_hot + probs - probs.detach()
        return straight_through.flatten(-2)

    def kl_loss(self, post: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """Balanced categorical KL with free bits."""

        def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            p = p.float()
            q = q.float()
            return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(-1).sum(-1)

        dynamics = kl(post.detach(), prior).clamp(min=self.cfg.free_bits)
        representation = kl(post, prior.detach()).clamp(min=self.cfg.free_bits)
        return self.cfg.dyn_scale * dynamics + self.cfg.rep_scale * representation

    @staticmethod
    def feat(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)
