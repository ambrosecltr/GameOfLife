"""DreamerBrain: a full DreamerV3-style agent living one unbroken life.

World model (encoder + RSSM + heads) trained on replayed sequences of the
robot's own experience; behavior from an actor-critic trained in imagination;
drives purely intrinsic: Plan2Explore ensemble disagreement (curiosity) +
homeostasis (eat, avoid damage, keep energy up). No tasks, no resets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from gol_world.interface import (
    EVENTS_DIM,
    NUM_GRIP_MODES,
    NUM_RAY_CLASSES,
    PROPRIO_DIM,
    SIGNAL_DIM,
    SOUND_DIM,
    Action,
    BodySpec,
    Observation,
)

from gol_brains.base import Brain
from gol_brains.dreamer.buffer import ReplayBuffer
from gol_brains.dreamer.networks import (
    RunningMeanStd,
    TanhNormal,
    TwoHot,
    mlp,
    percentile_scale,
)
from gol_brains.dreamer.rssm import RSSM, RSSMConfig

PRESETS: dict[str, dict[str, int]] = {
    "nano": {"deter": 256, "groups": 16, "classes": 16, "hidden": 256, "units": 256},
    "small": {"deter": 512, "groups": 24, "classes": 24, "hidden": 512, "units": 512},
    "base": {"deter": 1024, "groups": 32, "classes": 32, "hidden": 768, "units": 768},
}

ACTION_DIM = 2 + SIGNAL_DIM + NUM_GRIP_MODES  # drive(2) + signal(2) + gripper one-hot(4)


class WorldModel(nn.Module):
    def __init__(self, preset: dict[str, int], num_rays: int, wm_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.num_rays = num_rays
        units = preset["units"]
        obs_dim = num_rays * (1 + NUM_RAY_CLASSES) + PROPRIO_DIM + SOUND_DIM + EVENTS_DIM
        self.rssm_cfg = RSSMConfig(
            deter=preset["deter"],
            stoch_groups=preset["groups"],
            stoch_classes=preset["classes"],
            hidden=preset["hidden"],
            unimix=float(wm_cfg.get("unimix", 0.01)),
            free_bits=float(wm_cfg.get("kl_free_bits", 1.0)),
        )
        self.encoder = mlp(obs_dim, units, units, layers=2)
        self.rssm = RSSM(self.rssm_cfg, embed_dim=units, action_dim=ACTION_DIM)
        feat = self.rssm_cfg.feat_dim
        self.head_depth = mlp(feat, units, num_rays, layers=2)
        self.head_class = mlp(feat, units, num_rays * NUM_RAY_CLASSES, layers=2)
        self.head_proprio = mlp(feat, units, PROPRIO_DIM, layers=2)
        self.head_reward = mlp(feat, units, 41, layers=2)  # twohot homeostasis
        self.head_cont = mlp(feat, units, 1, layers=2)
        # Plan2Explore: each ensemble member predicts the NEXT observation
        # embedding from (state, action); their disagreement is epistemic
        # uncertainty, which is the curiosity signal.
        k = int(wm_cfg.get("ensemble_size", 8))
        self.ensemble = nn.ModuleList(
            mlp(feat + ACTION_DIM, units, units, layers=1) for _ in range(k)
        )

    def embed(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        flat = torch.cat(
            [
                obs["depth"],
                obs["class_onehot"].flatten(-2),
                obs["proprio"],
                obs["sound"],
                obs["events"],
            ],
            dim=-1,
        )
        return self.encoder(flat)

    def disagreement(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Plan2Explore intrinsic signal: variance across ensemble predictions."""
        x = torch.cat([feat, action], dim=-1)
        preds = torch.stack([net(x) for net in self.ensemble])  # (K, ..., stoch)
        return preds.var(dim=0).mean(-1)


class DreamerBrain(Brain):
    def __init__(self, cfg: dict[str, Any], seed: int, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.body = BodySpec()
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        preset = PRESETS[str(cfg.get("preset", "nano"))]
        wm_cfg = dict(cfg.get("world_model", {}))
        self.wm = WorldModel(preset, self.body.num_rays, wm_cfg).to(self.device)
        feat = self.wm.rssm_cfg.feat_dim
        units = preset["units"]

        ac_cfg = dict(cfg.get("actor_critic", {}))
        self.horizon = int(ac_cfg.get("imagination_horizon", 15))
        self.gamma = float(ac_cfg.get("gamma", 0.997))
        self.lam = float(ac_cfg.get("lam", 0.95))
        self.entropy_scale = float(ac_cfg.get("entropy", 3e-4))
        self.actor = mlp(feat, units, 4 * 2 + NUM_GRIP_MODES, layers=2).to(self.device)
        self.critic = mlp(feat, units, 41, layers=2).to(self.device)
        self.critic_ema = mlp(feat, units, 41, layers=2).to(self.device)
        self.critic_ema.load_state_dict(self.critic.state_dict())
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)
        self.twohot = TwoHot().to(self.device)

        rw = dict(cfg.get("reward", {}))
        self.w_curiosity = float(rw.get("w_curiosity", 1.0))
        self.w_homeostasis = float(rw.get("w_homeostasis", 1.0))
        self.low_energy_threshold = float(rw.get("low_energy_threshold", 0.25))
        self.low_energy_penalty = float(rw.get("low_energy_penalty", 0.02))
        # Ablation (research question 2): mask other robots out of the
        # curiosity target so agents aren't intrinsically drawn to each other.
        self.curiosity_mask_agents = bool(rw.get("curiosity_mask_agents", False))
        self.curiosity_norm = RunningMeanStd()

        replay = dict(cfg.get("replay", {}))
        self.buffer = ReplayBuffer(
            capacity=int(replay.get("capacity", 100_000)),
            num_rays=self.body.num_rays,
            action_dim=ACTION_DIM,
            seed=seed + 1,
        )
        self.batch_size = int(replay.get("batch_size", 16))
        self.seq_len = int(replay.get("seq_len", 64))
        # ~2 sim-minutes of motor babbling; long warmups starve newborns.
        self.warmup_steps = int(replay.get("warmup_steps", 500))

        tr = dict(cfg.get("training", {}))
        self.opt_model = torch.optim.Adam(self.wm.parameters(), lr=float(tr.get("model_lr", 1e-4)))
        self.opt_actor = torch.optim.Adam(
            self.actor.parameters(), lr=float(tr.get("actor_lr", 3e-5))
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=float(tr.get("critic_lr", 3e-5))
        )
        self.grad_clip = float(tr.get("grad_clip", 100.0))
        self.imag_starts = int(tr.get("imag_starts", 256))

        # Live recurrent state (the robot's stream of consciousness).
        self.h, self.z = self.wm.rssm.initial(1, self.device)
        self.last_action = torch.zeros(1, ACTION_DIM, device=self.device)
        self._return_scale = [1.0]
        self._metrics: dict[str, float] = {}
        self._updates = 0

    # ------------------------------------------------------------------- act

    def _obs_to_tensors(self, obs: Observation) -> dict[str, torch.Tensor]:
        rays = torch.as_tensor(obs["rays"], device=self.device)
        return {
            "depth": rays[..., 0].unsqueeze(0),
            "class_onehot": rays[..., 1:].unsqueeze(0),
            "proprio": torch.as_tensor(obs["proprio"], device=self.device).unsqueeze(0),
            "sound": torch.as_tensor(obs["sound"], device=self.device).unsqueeze(0),
            "events": torch.as_tensor(obs["events"], device=self.device).unsqueeze(0),
        }

    def _policy_dists(
        self, feat: torch.Tensor
    ) -> tuple[TanhNormal, torch.distributions.Categorical]:
        out = self.actor(feat)
        mean, raw_std, grip_logits = out[..., :4], out[..., 4:8], out[..., 8:]
        std = F.softplus(raw_std) + 0.1
        probs = torch.softmax(grip_logits, dim=-1)
        probs = 0.99 * probs + 0.01 / NUM_GRIP_MODES  # unimix keeps exploration alive
        return TanhNormal(torch.tanh(mean), std), torch.distributions.Categorical(probs=probs)

    def _action_to_vec(self, cont: torch.Tensor, grip: int) -> npt.NDArray[np.float32]:
        vec = np.zeros(ACTION_DIM, dtype=np.float32)
        vec[:4] = cont.detach().cpu().numpy()
        vec[4 + grip] = 1.0
        return vec

    def act(self, obs: Observation) -> Action:
        with torch.no_grad():
            tensors = self._obs_to_tensors(obs)
            embed = self.wm.embed(tensors)
            self.h, self.z, _, _ = self.wm.rssm.obs_step(self.h, self.z, self.last_action, embed)
            if len(self.buffer) < self.warmup_steps:
                cont = torch.as_tensor(
                    self.rng.uniform(-1, 1, 4).astype(np.float32), device=self.device
                )
                grip = int(self.rng.integers(0, NUM_GRIP_MODES))
            else:
                feat = self.wm.rssm.feat(self.h, self.z)
                dist_cont, dist_grip = self._policy_dists(feat)
                cont = dist_cont.sample()[0]
                grip = int(dist_grip.sample()[0])

        action_vec = self._action_to_vec(cont, grip)
        self.buffer.add(obs, action_vec)
        self.last_action = torch.as_tensor(action_vec, device=self.device).unsqueeze(0)
        return Action(
            drive=action_vec[:2].copy(),
            gripper=grip,
            signal=action_vec[2:4].copy(),
        )

    # ----------------------------------------------------------------- learn

    def _mask_agents(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rays that hit a robot read as 'nothing at max range' instead."""
        from gol_world.interface import RAY_CLASS_DORMANT, RAY_CLASS_NOTHING, RAY_CLASS_ROBOT

        class_onehot = obs["class_onehot"]
        is_agent = (class_onehot[..., RAY_CLASS_ROBOT] + class_onehot[..., RAY_CLASS_DORMANT]) > 0.5
        masked_class = class_onehot.clone()
        masked_class[is_agent] = 0.0
        masked_class[..., RAY_CLASS_NOTHING][is_agent] = 1.0
        masked_depth = obs["depth"].clone()
        masked_depth[is_agent] = 1.0
        return {**obs, "depth": masked_depth, "class_onehot": masked_class}

    def _homeostasis(self, events: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        ate, damage = events[..., 0], events[..., 1]
        low = (proprio[..., 5] < self.low_energy_threshold).float()
        return ate - damage - self.low_energy_penalty * low

    def learn(self) -> dict[str, float] | None:
        batch_np = self.buffer.sample_sequences(self.batch_size, self.seq_len)
        if batch_np is None or len(self.buffer) < self.warmup_steps:
            return None
        b = {k: torch.as_tensor(v, device=self.device) for k, v in batch_np.items()}
        B, L = b["depth"].shape[:2]
        class_idx = b["ray_class"].long()
        obs = {
            "depth": b["depth"],
            "class_onehot": F.one_hot(class_idx, NUM_RAY_CLASSES).float(),
            "proprio": b["proprio"],
            "sound": b["sound"],
            "events": b["events"],
        }

        # --- world model: posterior unroll over the sequence
        embed = self.wm.embed(obs)  # (B, L, units)
        h, z = self.wm.rssm.initial(B, self.device)
        feats, posts, priors = [], [], []
        zero_action = torch.zeros(B, ACTION_DIM, device=self.device)
        for t in range(L):
            prev_a = b["action"][:, t - 1] if t > 0 else zero_action
            h, z, post, prior = self.wm.rssm.obs_step(h, z, prev_a, embed[:, t])
            feats.append(self.wm.rssm.feat(h, z))
            posts.append(post)
            priors.append(prior)
        feat = torch.stack(feats, dim=1)  # (B, L, F)
        post = torch.stack(posts, dim=1)
        prior = torch.stack(priors, dim=1)

        pred_depth = self.wm.head_depth(feat)
        pred_class = self.wm.head_class(feat).view(B, L, self.wm.num_rays, NUM_RAY_CLASSES)
        pred_proprio = self.wm.head_proprio(feat)
        loss_depth = F.mse_loss(pred_depth, b["depth"], reduction="none").sum(-1)
        loss_class = (
            F.cross_entropy(pred_class.flatten(0, 2), class_idx.flatten(), reduction="none")
            .view(B, L, -1)
            .sum(-1)
        )
        loss_proprio = F.mse_loss(pred_proprio, b["proprio"], reduction="none").sum(-1)
        homeo = self._homeostasis(b["events"], b["proprio"])
        loss_reward = self.twohot.loss(self.wm.head_reward(feat), homeo)
        cont_target = (b["proprio"][..., 5] > 0.01).float()
        loss_cont = F.binary_cross_entropy_with_logits(
            self.wm.head_cont(feat).squeeze(-1), cont_target, reduction="none"
        )
        loss_kl = self.wm.rssm.kl_loss(post, prior)

        # --- Plan2Explore ensemble: predict the next observation embedding.
        # With curiosity_mask_agents, other robots are erased from the target
        # (their rays read as "nothing at max range"), so their unpredictability
        # generates no curiosity.
        with torch.no_grad():
            if self.curiosity_mask_agents:
                ens_target = self.wm.embed(self._mask_agents(obs))[:, 1:]
            else:
                ens_target = embed.detach()[:, 1:]
        ens_in_feat = feat[:, :-1].detach()
        ens_action = b["action"][:, 1:]  # action taken at t leads to obs_{t+1}
        x = torch.cat([ens_in_feat, ens_action], dim=-1)
        ens_losses = [
            F.mse_loss(net(x), ens_target, reduction="none").mean(-1) for net in self.wm.ensemble
        ]
        loss_ens = torch.stack(ens_losses).mean(0)

        model_loss = (
            loss_depth + loss_class + loss_proprio + loss_reward + loss_cont + loss_kl
        ).mean() + loss_ens.mean()
        self.opt_model.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip)
        self.opt_model.step()

        # Curiosity statistics on real experience (keeps normalization honest).
        with torch.no_grad():
            real_disagreement = self.wm.disagreement(feat[:, :-1].detach(), b["action"][:, 1:])
            self.curiosity_norm.update(real_disagreement)

        # --- actor-critic in imagination, from a subsample of posterior states
        flat = feat.detach().flatten(0, 1)  # (B*L, F) = concat(h, z)
        starts = torch.randperm(flat.shape[0], device=self.device)[: self.imag_starts]
        h_i = flat[starts, : self.wm.rssm_cfg.deter]
        z_i = flat[starts, self.wm.rssm_cfg.deter :]

        img_feats, img_logps, img_ents, img_actions = [], [], [], []
        for _ in range(self.horizon):
            f_i = self.wm.rssm.feat(h_i, z_i)
            dist_cont, dist_grip = self._policy_dists(f_i)
            a_cont = dist_cont.sample()
            a_grip = dist_grip.sample()
            logp = dist_cont.log_prob(a_cont) + dist_grip.log_prob(a_grip)
            ent = dist_cont.entropy() + dist_grip.entropy()
            a_vec = torch.cat([a_cont, F.one_hot(a_grip, NUM_GRIP_MODES).float()], dim=-1)
            img_feats.append(f_i)
            img_logps.append(logp)
            img_ents.append(ent)
            img_actions.append(a_vec)
            with torch.no_grad():
                h_i, z_i, _ = self.wm.rssm.img_step(h_i, z_i, a_vec)
        img_feat = torch.stack(img_feats)  # (H, N, F)
        img_logp = torch.stack(img_logps)
        img_ent = torch.stack(img_ents)
        img_action = torch.stack(img_actions)

        with torch.no_grad():
            r_homeo = self.twohot.decode(self.wm.head_reward(img_feat))
            r_cur = self.curiosity_norm.normalize(self.wm.disagreement(img_feat, img_action)).clamp(
                0, 5.0
            )
            reward = self.w_homeostasis * r_homeo + self.w_curiosity * r_cur
            cont = torch.sigmoid(self.wm.head_cont(img_feat).squeeze(-1))
            discount = self.gamma * cont
            value_ema = self.twohot.decode(self.critic_ema(img_feat))
            # lambda-returns, backward pass
            returns = torch.zeros_like(value_ema)
            last = value_ema[-1]
            for t in reversed(range(self.horizon)):
                bootstrap = (
                    (1 - self.lam) * value_ema[t + 1] + self.lam * last
                    if t + 1 < self.horizon
                    else last
                )
                returns[t] = reward[t] + discount[t] * bootstrap
                last = returns[t]

        # Critic: twohot regression to lambda-returns.
        critic_logits = self.critic(img_feat.detach())
        loss_critic = self.twohot.loss(critic_logits, returns.detach()).mean()
        self.opt_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()
        with torch.no_grad():
            for p, p_ema in zip(
                self.critic.parameters(), self.critic_ema.parameters(), strict=True
            ):
                p_ema.lerp_(p, 0.02)

        # Actor: REINFORCE on normalized advantages + entropy bonus.
        with torch.no_grad():
            value = self.twohot.decode(self.critic(img_feat))
            scaled_ret = percentile_scale(returns, self._return_scale)
            scaled_val = value / max(1.0, self._return_scale[0])
            adv = (scaled_ret - scaled_val).detach()
        loss_actor = (-img_logp * adv - self.entropy_scale * img_ent).mean()
        self.opt_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.opt_actor.step()

        self._updates += 1
        self._metrics = {
            "loss_model": float(model_loss.detach()),
            "pred_error_depth": float(loss_depth.detach().mean() / self.wm.num_rays),
            "pred_error_class": float(loss_class.detach().mean() / self.wm.num_rays),
            "kl": float(loss_kl.detach().mean()),
            "curiosity": float(real_disagreement.mean()),
            "curiosity_scaled": float(
                self.curiosity_norm.normalize(real_disagreement).clamp(0, 5).mean()
            ),
            "reward_homeostasis": float(homeo.mean()),
            "value": float(value.mean()),
            "loss_critic": float(loss_critic.detach()),
            "loss_actor": float(loss_actor.detach()),
            "entropy": float(img_ent.detach().mean()),
            "updates": float(self._updates),
            "buffer": float(len(self.buffer)),
        }
        return self._metrics

    def introspect(self) -> dict[str, float]:
        return dict(self._metrics)

    def reset_stream(self) -> None:
        """New body, same mind: reset the live recurrent state only."""
        self.h, self.z = self.wm.rssm.initial(1, self.device)
        self.last_action = torch.zeros(1, ACTION_DIM, device=self.device)

    # ----------------------------------------------------------- persistence

    def state_dict(self) -> dict[str, Any]:
        return {
            "obs_version": 1,
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_ema": self.critic_ema.state_dict(),
            "opt_model": self.opt_model.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "curiosity_norm": self.curiosity_norm.state_dict(),
            "return_scale": self._return_scale[0],
            "updates": self._updates,
            "rng_state": self.rng.bit_generator.state,
            "buffer": self.buffer.state_dict(),
            "h": self.h.cpu().numpy(),
            "z": self.z.cpu().numpy(),
            "last_action": self.last_action.cpu().numpy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        from gol_world.interface import OBS_VERSION

        if state.get("obs_version") != OBS_VERSION:
            raise ValueError(
                f"brain checkpoint has obs_version {state.get('obs_version')}, "
                f"world speaks {OBS_VERSION}: refusing to load across contract changes"
            )
        self.wm.load_state_dict(state["wm"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_ema.load_state_dict(state["critic_ema"])
        self.opt_model.load_state_dict(state["opt_model"])
        self.opt_actor.load_state_dict(state["opt_actor"])
        self.opt_critic.load_state_dict(state["opt_critic"])
        self.curiosity_norm.load_state_dict(state["curiosity_norm"])
        self._return_scale = [float(state["return_scale"])]
        self._updates = int(state["updates"])
        self.rng.bit_generator.state = state["rng_state"]
        self.buffer.load_state_dict(state["buffer"])
        self.h = torch.as_tensor(state["h"], device=self.device)
        self.z = torch.as_tensor(state["z"], device=self.device)
        self.last_action = torch.as_tensor(state["last_action"], device=self.device)
