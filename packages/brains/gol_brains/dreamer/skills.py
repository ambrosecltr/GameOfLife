"""Learned temporal skills for hierarchical control.

The manager selects an unnamed discrete latent intent every ``duration``
actions. The worker turns that intent and the current world-model state into
motor commands. A discriminator rewards skills whose *temporal displacement*
is distinguishable, so reusable control patterns can emerge without action
labels, demonstrations, or designer-specified behaviours.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from gol_brains.dreamer.networks import DiscreteDist, TanhNormal, mlp


class TemporalSkillController(nn.Module):
    """The manager/worker subset needed for embodied inference."""

    def __init__(
        self,
        manager: nn.Module,
        worker: nn.Module,
        cont_dim: int,
        grip_modes: int,
        num_skills: int,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.cont_dim = cont_dim
        self.grip_modes = grip_modes
        self.num_skills = num_skills
        self.unimix = unimix
        self.manager = manager
        self.worker = worker

    def manager_dist(self, feat: torch.Tensor) -> DiscreteDist:
        probs = torch.softmax(self.manager(feat).float(), dim=-1)
        probs = (1.0 - self.unimix) * probs + self.unimix / self.num_skills
        return DiscreteDist(probs)

    def action_dists(
        self, feat: torch.Tensor, skill: torch.Tensor
    ) -> tuple[TanhNormal, DiscreteDist]:
        onehot = F.one_hot(skill, self.num_skills).to(dtype=feat.dtype)
        out = self.worker(torch.cat([feat, onehot], dim=-1))
        out = out.float()
        mean = out[..., : self.cont_dim]
        raw_std = out[..., self.cont_dim : 2 * self.cont_dim]
        grip_logits = out[..., 2 * self.cont_dim :]
        std = F.softplus(raw_std) + 0.1
        probs = torch.softmax(grip_logits, dim=-1)
        probs = 0.99 * probs + 0.01 / self.grip_modes
        return TanhNormal(torch.tanh(mean), std), DiscreteDist(probs)

    def controller_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.manager.parameters()
        yield from self.worker.parameters()


class TemporalSkillPolicy(TemporalSkillController):
    """Manager/worker policy plus a learned controllability discriminator."""

    def __init__(
        self,
        feat_dim: int,
        units: int,
        cont_dim: int,
        grip_modes: int,
        num_skills: int,
        unimix: float = 0.01,
    ) -> None:
        if num_skills < 2:
            raise ValueError("temporal_skills.num_skills must be at least 2")
        manager = mlp(feat_dim, units, num_skills, layers=2)
        worker = mlp(
            feat_dim + num_skills,
            units,
            cont_dim * 2 + grip_modes,
            layers=2,
        )
        super().__init__(manager, worker, cont_dim, grip_modes, num_skills, unimix)
        # Read displacement, not absolute state: a skill must make a
        # characteristic difference over time instead of identifying a place.
        self.discriminator = mlp(feat_dim, units, num_skills, layers=2)

    def discrimination_logits(
        self, start_feat: torch.Tensor, end_feat: torch.Tensor
    ) -> torch.Tensor:
        return self.discriminator(end_feat - start_feat)

    def intrinsic_reward(
        self, start_feat: torch.Tensor, end_feat: torch.Tensor, skill: torch.Tensor
    ) -> torch.Tensor:
        """Variational lower-bound reward: log q(z | displacement) + log |Z|."""
        logits = self.discrimination_logits(start_feat, end_feat).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        identified = log_probs.gather(-1, skill.unsqueeze(-1)).squeeze(-1)
        return identified + math.log(self.num_skills)

    def inference_controller(self) -> TemporalSkillController:
        return TemporalSkillController(
            copy.deepcopy(self.manager),
            copy.deepcopy(self.worker),
            self.cont_dim,
            self.grip_modes,
            self.num_skills,
            self.unimix,
        )
