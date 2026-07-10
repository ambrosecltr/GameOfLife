"""Immutable inference snapshots for non-blocking embodied action."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from gol_brains.dreamer.rssm import RSSM
from gol_brains.dreamer.skills import TemporalSkillPolicy


class InferenceSnapshot(nn.Module):
    """Frozen encoder, dynamics, and policy published by the learner.

    Publishing is a single reference assignment. An act already in progress
    retains its old snapshot, while the next act sees the new one; neither
    touches parameters while the optimizer mutates them.
    """

    def __init__(self, encoder: nn.Module, rssm: RSSM, actor: nn.Module) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.rssm = copy.deepcopy(rssm)
        self.actor = (
            actor.inference_controller()
            if isinstance(actor, TemporalSkillPolicy)
            else copy.deepcopy(actor)
        )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def embed(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        flat = torch.cat(
            [
                obs["depth"],
                obs["rgb"].flatten(-2),
                obs["kind_onehot"].flatten(-2),
                obs["proprio"],
                obs["sound"],
                obs["events"],
            ],
            dim=-1,
        )
        return self.encoder(flat)
