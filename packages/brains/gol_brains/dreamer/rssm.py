"""The RSSM: recurrent state-space model with categorical stochastic latents.

DreamerV3 recipe: GRU deterministic path, groups-of-classes one-hot latents
with straight-through gradients, 1% unimix, KL balancing with free bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from gol_brains.dreamer.dynamics import CategoricalLatentDynamics, DynamicsSequence
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


class RSSM(CategoricalLatentDynamics):
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
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        embed: torch.Tensor,
        step_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One posterior step: returns (h', z_post, post_probs, prior_probs)."""
        del step_scale  # A GRU has no explicit continuous-time transition scale.
        h = self._step_deter(h, z, action)
        prior_probs = self._logits_to_probs(self.prior_net(h))
        post_probs = self._logits_to_probs(self.post_net(torch.cat([h, embed], dim=-1)))
        return h, self._sample(post_probs), post_probs, prior_probs

    def observe_sequence(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        first: torch.Tensor,
        wake: torch.Tensor,
        step_scale: torch.Tensor,
        burn_in: int,
    ) -> DynamicsSequence:
        """Sequential posterior unroll used by the beta lineage.

        Stream markers and elapsed scales are accepted as part of the shared
        dynamics boundary. Beta deliberately retains its historical recurrent
        semantics; Aion consumes those signals in its resettable S5 scan.
        """
        del first, wake, step_scale
        batch = embed.shape[0]
        total = embed.shape[1]
        h, z = self.initial(batch, embed.device)
        zero_action = torch.zeros(batch, action.shape[-1], device=embed.device)
        if burn_in > 0:
            with torch.no_grad():
                for index in range(burn_in):
                    previous_action = action[:, index - 1] if index > 0 else zero_action
                    h, z, _, _ = self.obs_step(h, z, previous_action, embed[:, index])

        states_h: list[torch.Tensor] = []
        states_z: list[torch.Tensor] = []
        posts: list[torch.Tensor] = []
        priors: list[torch.Tensor] = []
        for index in range(burn_in, total):
            previous_action = action[:, index - 1] if index > 0 else zero_action
            h, z, post, prior = self.obs_step(h, z, previous_action, embed[:, index])
            states_h.append(h)
            states_z.append(z)
            posts.append(post)
            priors.append(prior)
        return DynamicsSequence(
            h=torch.stack(states_h, dim=1),
            z=torch.stack(states_z, dim=1),
            post=torch.stack(posts, dim=1),
            prior=torch.stack(priors, dim=1),
        )
