"""DreamerBrain: a full DreamerV3-style agent living one unbroken life.

World model (encoder + RSSM + heads) trained on replayed sequences of the
robot's own experience; behavior from an actor-critic trained in imagination;
drives purely intrinsic — no tasks, no resets:

- Curiosity: learning progress over self-organized latent regions (interest
  as the derivative of competence; see interest.py), or legacy Plan2Explore
  disagreement as an ablation.
- Homeostasis: HRRL drive-reduction (Keramati & Gutkin) — reward is movement
  of the internal state (energy, integrity, rest) toward setpoints, so the
  same meal is worth more to a starving body than a sated one. The legacy
  ate/damage event bonus remains as an ablation.
- Boredom: a standing cost of being safe and learning nothing — the pressure
  that produces play.
- Temperament: heritable log-normal multipliers over the abstract drive
  knobs, sampled at birth and mutated on inheritance. Individuality is seeded
  abstractly; concrete interests must be discovered.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from gol_world.blocks import SKY_DAY, SKY_NIGHT
from gol_world.interface import (
    EVENTS_DIM,
    GAZE_DIM,
    NUM_GRIP_MODES,
    NUM_RAY_KINDS,
    OBS_VERSION,
    PROPRIO_DIM,
    RAY_DIM,
    RAY_KIND_DORMANT,
    RAY_KIND_ROBOT,
    SIGNAL_DIM,
    SOUND_DIM,
    Action,
    BodySpec,
    Observation,
)

from gol_brains.base import Brain
from gol_brains.dreamer.buffer import ReplayBuffer
from gol_brains.dreamer.interest import LearningProgress, OnlineRegions
from gol_brains.dreamer.networks import (
    DiscreteDist,
    EnsembleMLP,
    RunningMeanStd,
    TanhNormal,
    TwoHot,
    mlp,
    percentile_scale,
)
from gol_brains.dreamer.optim import Muon
from gol_brains.dreamer.rssm import RSSM, RSSMConfig

PRESETS: dict[str, dict[str, int]] = {
    "nano": {"deter": 256, "groups": 16, "classes": 16, "hidden": 256, "units": 256},
    "small": {"deter": 512, "groups": 24, "classes": 24, "hidden": 512, "units": 512},
    "base": {"deter": 1024, "groups": 32, "classes": 32, "hidden": 768, "units": 768},
}

# drive(2) + signal(2) + gaze(2) continuous, then gripper one-hot(4).
CONT_DIM = 2 + SIGNAL_DIM + GAZE_DIM
ACTION_DIM = CONT_DIM + NUM_GRIP_MODES

# Innate individuality: log-normal multipliers on abstract, domain-general
# knobs only — how strongly each drive weighs, how curious, how restless.
# Never object-specific ("likes animals" must be discovered, not wired in).
TEMPERAMENT_KEYS = (
    "w_curiosity",
    "w_homeostasis",
    "drive_energy",
    "drive_integrity",
    "drive_rest",
    "boredom",
    "entropy",
)

# Ray-kind LP partition (ablation): presence-combos of animate hit kinds —
# nothing-alive / robot / dormant / both.
KIND_REGIONS = 4


class WorldModel(nn.Module):
    def __init__(self, preset: dict[str, int], num_rays: int, wm_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.num_rays = num_rays
        units = preset["units"]
        obs_dim = num_rays * RAY_DIM + PROPRIO_DIM + SOUND_DIM + EVENTS_DIM
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
        self.head_rgb = mlp(feat, units, num_rays * 3, layers=2)
        self.head_kind = mlp(feat, units, num_rays * NUM_RAY_KINDS, layers=2)
        self.head_proprio = mlp(feat, units, PROPRIO_DIM, layers=2)
        self.head_reward = mlp(feat, units, 41, layers=2)  # twohot homeostasis
        self.head_cont = mlp(feat, units, 1, layers=2)
        # Plan2Explore: each ensemble member predicts the NEXT observation
        # embedding from (state, action); their disagreement is epistemic
        # uncertainty, which is the curiosity signal. Members are stacked into
        # one batched module (K einsums, not K module calls); pre-swift
        # checkpoints stored a ModuleList and are migrated on load.
        k = int(wm_cfg.get("ensemble_size", 8))
        self.ensemble = EnsembleMLP(k, feat + ACTION_DIM, units, units)

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

    def disagreement(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Plan2Explore intrinsic signal: variance across ensemble predictions."""
        x = torch.cat([feat, action], dim=-1)
        preds = self.ensemble(x)  # (K, ..., units)
        return preds.var(dim=0).mean(-1)


def _migrate_ensemble_state(wm_state: dict[str, torch.Tensor]) -> None:
    """Stack a pre-swift ModuleList ensemble checkpoint into the batched layout.

    Old keys: ensemble.{k}.{0,1,3}.{weight,bias} (Linear, LayerNorm, Linear).
    Mutates wm_state in place; a no-op on already-migrated checkpoints. Kept
    so cloud lives (beta_08/09) load into current code for offline analysis.
    """
    if "ensemble.0.0.weight" not in wm_state:
        return
    k = 0
    while f"ensemble.{k}.0.weight" in wm_state:
        k += 1
    members = [{key: wm_state.pop(f"ensemble.{j}.{key}") for key in
                ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias")}
               for j in range(k)]
    wm_state["ensemble.w1"] = torch.stack([m["0.weight"].T for m in members])
    wm_state["ensemble.b1"] = torch.stack([m["0.bias"] for m in members])
    wm_state["ensemble.ln_w"] = torch.stack([m["1.weight"] for m in members])
    wm_state["ensemble.ln_b"] = torch.stack([m["1.bias"] for m in members])
    wm_state["ensemble.w2"] = torch.stack([m["3.weight"].T for m in members])
    wm_state["ensemble.b2"] = torch.stack([m["3.bias"] for m in members])


class DreamerBrain(Brain):
    def __init__(
        self, cfg: dict[str, Any], seed: int, device: str = "cpu", body: BodySpec | None = None
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.body = body or BodySpec()
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
        self.actor = mlp(feat, units, CONT_DIM * 2 + NUM_GRIP_MODES, layers=2).to(self.device)
        self.critic = mlp(feat, units, 41, layers=2).to(self.device)
        self.critic_ema = mlp(feat, units, 41, layers=2).to(self.device)
        self.critic_ema.load_state_dict(self.critic.state_dict())
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)
        self.twohot = TwoHot().to(self.device)

        rw = dict(cfg.get("reward", {}))
        self.w_curiosity = float(rw.get("w_curiosity", 1.0))
        self.w_homeostasis = float(rw.get("w_homeostasis", 1.0))
        # "drive": HRRL drive-reduction over internal state (energy, integrity,
        # rest). "events": the legacy ate/damage bonus, kept as an ablation and
        # as the default so configs stored in older saves keep the reward their
        # brains were trained on.
        self.homeostasis_mode = str(rw.get("homeostasis", "events"))
        if self.homeostasis_mode not in ("drive", "events"):
            raise ValueError(f"unknown homeostasis mode: {self.homeostasis_mode!r}")
        drv = dict(rw.get("drive", {}))
        self.drive_scale = float(drv.get("scale", 3.0))
        self.drive_level_penalty = float(drv.get("level_penalty", 0.01))
        self.drive_pow_m = float(drv.get("pow_m", 3.0))
        self.drive_pow_n = float(drv.get("pow_n", 2.0))
        self.drive_setpoints = torch.tensor(
            [
                float(drv.get("energy_setpoint", 0.85)),
                float(drv.get("integrity_setpoint", 1.0)),
                float(drv.get("rested_setpoint", 1.0)),
            ],
            device=self.device,
        )
        self.drive_weights = torch.tensor(
            [
                float(drv.get("energy_weight", 1.0)),
                float(drv.get("integrity_weight", 1.0)),
                float(drv.get("rest_weight", 0.5)),
            ],
            device=self.device,
        )
        self.low_energy_threshold = float(rw.get("low_energy_threshold", 0.25))
        self.low_energy_penalty = float(rw.get("low_energy_penalty", 0.02))
        self.low_energy_graded = bool(rw.get("low_energy_graded", True))
        # Ablation (research question 2): mask other robots out of the
        # curiosity target so agents aren't intrinsically drawn to each other.
        self.curiosity_mask_agents = bool(rw.get("curiosity_mask_agents", False))
        # Round-009 signal conditioning: calibrate both curiosity normalizers
        # on the first `norm_anchor_samples` real samples, then freeze the
        # scale. Without an anchor the lifetime running std shrinks with the
        # decaying signal and re-inflates it (beta_08: curiosity_scaled rose
        # 0.09→1.86 while raw LP fell). 0 = legacy lifetime std.
        self.norm_anchor = float(rw.get("norm_anchor_samples", 0))
        self.curiosity_norm = RunningMeanStd(anchor=self.norm_anchor)
        # Curiosity flavor: "lp" rewards learning progress — the *derivative*
        # of competence over self-organized regions (Oudeyer), so interest is
        # a moving frontier and mastered or unlearnable things both go stale.
        # "disagreement" is the legacy Plan2Explore level signal, kept as the
        # ablation and the code default so stored configs keep their reward.
        self.curiosity_mode = str(rw.get("curiosity", "disagreement"))
        if self.curiosity_mode not in ("lp", "disagreement"):
            raise ValueError(f"unknown curiosity mode: {self.curiosity_mode!r}")
        lp_cfg = dict(rw.get("lp", {}))
        self.lp_partition = str(lp_cfg.get("partition", "latent"))
        if self.lp_partition not in ("latent", "kind"):
            raise ValueError(f"unknown lp partition: {self.lp_partition!r}")
        n_regions = (
            int(lp_cfg.get("regions", 32)) if self.lp_partition == "latent" else KIND_REGIONS
        )
        self.regions = OnlineRegions(
            n_regions, feat, lr=float(lp_cfg.get("centroid_lr", 0.05)), device=self.device
        )
        self.lp = LearningProgress(
            n_regions,
            fast=float(lp_cfg.get("ema_fast", 0.02)),
            slow=float(lp_cfg.get("ema_slow", 0.002)),
            relative=bool(lp_cfg.get("relative", True)),
        )
        self.lp_mix_disagreement = float(lp_cfg.get("mix_disagreement", 0.1))
        # The trickle exists for newborn cold-start; annealed to zero over
        # this many act-steps it stops subsidizing adult stimulation
        # (beta_08: the rising normalized-disagreement floor was worth ~40%
        # of the boredom stim gate late in life). 0 = legacy, never anneals.
        self.lp_mix_anneal_steps = int(lp_cfg.get("mix_anneal_steps", 0))
        self.lp_norm = RunningMeanStd(anchor=self.norm_anchor)
        # Boredom: a standing cost of being safe AND learning nothing — the
        # pressure that produces play, and the counterweight that stops an
        # agent coasting once its niche is mastered. Weight 0 disables.
        bd = dict(rw.get("boredom", {}))
        self.boredom_weight = float(bd.get("weight", 0.0))
        self.boredom_stim_threshold = float(bd.get("stim_threshold", 0.5))
        self.boredom_drive_threshold = float(bd.get("drive_threshold", 0.15))
        # Pressure mode: boredom as a leaky-integrated mood over real
        # experience, not an instantaneous gate product. Sustained
        # gate-touching accumulates pressure; relief drains it (beta_08:
        # stimulation sat on the gate for 650k ticks and boredom only ever
        # flickered ≤1.6e-3 — there was no state for it to build in).
        self.boredom_pressure_on = bool(bd.get("pressure", False))
        self.boredom_pressure_rise = float(bd.get("pressure_rise", 0.002))
        self.boredom_pressure_decay = float(bd.get("pressure_decay", 0.0002))

        # Temperament: innate, heritable individuality. Each newborn samples
        # log-normal multipliers over the abstract drive knobs; inheritance
        # (Brain.inherit) copies the donor's and mutates. Nothing here names
        # an object or activity — specific interests must be discovered.
        tp = dict(cfg.get("temperament", {}))
        self.temperament_enabled = bool(tp.get("enabled", False))
        self.temperament_mutation = float(tp.get("mutation_sigma", 0.1))
        t_sigma = float(tp.get("sigma", 0.25))
        self._pre_temperament: dict[str, Any] = {
            "w_curiosity": self.w_curiosity,
            "w_homeostasis": self.w_homeostasis,
            "drive_weights": self.drive_weights.clone(),
            "boredom_weight": self.boredom_weight,
            "entropy_scale": self.entropy_scale,
        }
        if self.temperament_enabled:
            self.temperament = {
                k: float(np.exp(self.rng.normal(0.0, t_sigma))) for k in TEMPERAMENT_KEYS
            }
        else:
            self.temperament = dict.fromkeys(TEMPERAMENT_KEYS, 1.0)
        self._apply_temperament()

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
        # Burn-in: prefix steps unrolled without gradients purely to warm the
        # recurrent state, so seq_len (the expensive grad-carrying part of the
        # unroll) can shrink without every sequence starting from a zero-state
        # lie. 0 = legacy zero-init.
        self.burn_in = int(replay.get("burn_in", 0))
        # Batch rows pinned to the newest experience (DreamerV3 online-queue
        # mixing); the rest sample uniformly over the whole life. 0 = legacy.
        self.recent_slots = int(replay.get("recent", 0))

        tr = dict(cfg.get("training", {}))
        model_lr = float(tr.get("model_lr", 1e-4))
        self.optimizer_kind = str(tr.get("optimizer", "adam"))
        if self.optimizer_kind not in ("adam", "muon"):
            raise ValueError(f"unknown optimizer: {self.optimizer_kind!r}")
        self.opt_model_muon: Muon | None = None
        if self.optimizer_kind == "muon":
            # Muon conditions exactly-2D weight matrices; biases, LayerNorms,
            # and the stacked 3D ensemble stay on Adam (the standard recipe).
            matrices = [p for p in self.wm.parameters() if p.dim() == 2]
            rest = [p for p in self.wm.parameters() if p.dim() != 2]
            self.opt_model_muon = Muon(matrices, lr=float(tr.get("muon_lr", 0.02)))
            self.opt_model = torch.optim.Adam(rest, lr=model_lr)
        else:
            self.opt_model = torch.optim.Adam(self.wm.parameters(), lr=model_lr)
        self.opt_actor = torch.optim.Adam(
            self.actor.parameters(), lr=float(tr.get("actor_lr", 3e-5))
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=float(tr.get("critic_lr", 3e-5))
        )
        self.grad_clip = float(tr.get("grad_clip", 100.0))
        self.imag_starts = int(tr.get("imag_starts", 256))
        # L2-towards-init on the world model: plasticity maintenance for one
        # unbroken life on a nonstationary stream (Kumar et al. — regularizing
        # toward *an* init-distribution draw preserves trainability; the
        # anchor doesn't ride checkpoints because any draw serves). 0 = off.
        self.l2_init_weight = float(tr.get("l2_init", 0.0))
        self._wm_init: list[torch.Tensor] = (
            [p.detach().clone() for p in self.wm.parameters()] if self.l2_init_weight > 0 else []
        )
        # Updates per act-step the learner thread paces toward. Both counters
        # it paces with (_act_steps, _updates) ride state_dict, so the debt
        # math stays coherent across checkpoints and inheritance.
        self.train_ratio = float(tr.get("train_ratio", 0.25))

        # Live recurrent state (the robot's stream of consciousness).
        self.h, self.z = self.wm.rssm.initial(1, self.device)
        self.last_action = torch.zeros(1, ACTION_DIM, device=self.device)
        self._return_scale = [1.0]
        self._metrics: dict[str, float] = {}
        self._updates = 0
        self._act_steps = 0
        self._learn_seconds = 0.0  # EMA of learn() wall time
        self._gate_calm = 0.0  # boredom gate telemetry (imagination means)
        self._gate_dull = 0.0
        self._boredom_pressure = 0.0  # leaky-integrated mood (pressure mode)

        # torch.compile on the sequential hot loops (RSSM steps + encoder):
        # nano-scale unrolls are dispatch-bound, and compilation collapses the
        # per-step Python/kernel-launch overhead. The act-path shape (B=1) is
        # precompiled here so the sim thread never eats a compile stall
        # mid-life; the learn-path shapes compile lazily on the learner
        # thread, where a one-time pause is just a skipped update.
        if bool(tr.get("compile", False)):
            backend = "aot_eager" if self.device.type == "mps" else "inductor"
            for owner, name in ((self.wm.rssm, "obs_step"), (self.wm.rssm, "img_step"),
                                (self.wm, "embed")):
                setattr(owner, name, torch.compile(getattr(owner, name), backend=backend))
            with torch.no_grad():
                embed = torch.zeros(1, preset["units"], device=self.device)
                self.wm.rssm.obs_step(self.h, self.z, self.last_action, embed)

    def _apply_temperament(self) -> None:
        t, base = self.temperament, self._pre_temperament
        self.w_curiosity = base["w_curiosity"] * t["w_curiosity"]
        self.w_homeostasis = base["w_homeostasis"] * t["w_homeostasis"]
        mult = torch.tensor(
            [t["drive_energy"], t["drive_integrity"], t["drive_rest"]], device=self.device
        )
        self.drive_weights = base["drive_weights"] * mult
        self.boredom_weight = base["boredom_weight"] * t["boredom"]
        self.entropy_scale = base["entropy_scale"] * t["entropy"]

    # ------------------------------------------------------------------- act

    def _obs_to_tensors(self, obs: Observation) -> dict[str, torch.Tensor]:
        rays = torch.as_tensor(obs["rays"], device=self.device)
        return {
            "depth": rays[..., 0].unsqueeze(0),
            "rgb": rays[..., 1:4].unsqueeze(0),
            "kind_onehot": rays[..., 4:].unsqueeze(0),
            "proprio": torch.as_tensor(obs["proprio"], device=self.device).unsqueeze(0),
            "sound": torch.as_tensor(obs["sound"], device=self.device).unsqueeze(0),
            "events": torch.as_tensor(obs["events"], device=self.device).unsqueeze(0),
        }

    def _policy_dists(self, feat: torch.Tensor) -> tuple[TanhNormal, DiscreteDist]:
        out = self.actor(feat)
        mean = out[..., :CONT_DIM]
        raw_std = out[..., CONT_DIM : 2 * CONT_DIM]
        grip_logits = out[..., 2 * CONT_DIM :]
        std = F.softplus(raw_std) + 0.1
        probs = torch.softmax(grip_logits, dim=-1)
        probs = 0.99 * probs + 0.01 / NUM_GRIP_MODES  # unimix keeps exploration alive
        return TanhNormal(torch.tanh(mean), std), DiscreteDist(probs)

    def _action_to_vec(self, cont: torch.Tensor, grip: int) -> npt.NDArray[np.float32]:
        vec = np.zeros(ACTION_DIM, dtype=np.float32)
        vec[:CONT_DIM] = cont.detach().cpu().numpy()
        vec[CONT_DIM + grip] = 1.0
        return vec

    def act(self, obs: Observation) -> Action:
        with torch.no_grad():
            tensors = self._obs_to_tensors(obs)
            embed = self.wm.embed(tensors)
            self.h, self.z, _, _ = self.wm.rssm.obs_step(self.h, self.z, self.last_action, embed)
            if len(self.buffer) < self.warmup_steps:
                cont = torch.as_tensor(
                    self.rng.uniform(-1, 1, CONT_DIM).astype(np.float32), device=self.device
                )
                grip = int(self.rng.integers(0, NUM_GRIP_MODES))
            else:
                feat = self.wm.rssm.feat(self.h, self.z)
                dist_cont, dist_grip = self._policy_dists(feat)
                cont = dist_cont.sample()[0]
                grip = int(dist_grip.sample()[0])

        action_vec = self._action_to_vec(cont, grip)
        self.buffer.add(obs, action_vec)
        self._act_steps += 1
        self.last_action = torch.as_tensor(action_vec, device=self.device).unsqueeze(0)
        return Action(
            drive=action_vec[:2].copy(),
            gripper=grip,
            signal=action_vec[2:4].copy(),
            gaze=action_vec[4:6].copy(),
        )

    # ----------------------------------------------------------------- learn

    def _mask_agents(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rays that hit a robot read as 'sky at max range' instead."""
        from gol_world.interface import RAY_KIND_DORMANT, RAY_KIND_NOTHING, RAY_KIND_ROBOT

        kind_onehot = obs["kind_onehot"]
        is_agent = (kind_onehot[..., RAY_KIND_ROBOT] + kind_onehot[..., RAY_KIND_DORMANT]) > 0.5
        masked_kind = kind_onehot.clone()
        masked_kind[is_agent] = 0.0
        masked_kind[..., RAY_KIND_NOTHING][is_agent] = 1.0
        masked_depth = obs["depth"].clone()
        masked_depth[is_agent] = 1.0
        # A masked ray must look like a real miss: the sky at the current
        # light level (proprio[13]), not black.
        light = obs["proprio"][..., 13]
        night = torch.as_tensor(SKY_NIGHT, device=light.device)
        day = torch.as_tensor(SKY_DAY, device=light.device)
        sky = night + (day - night) * light[..., None]  # (..., 3)
        masked_rgb = obs["rgb"].clone()
        masked_rgb[is_agent] = sky.unsqueeze(-2).expand_as(masked_rgb)[is_agent]
        return {**obs, "depth": masked_depth, "rgb": masked_rgb, "kind_onehot": masked_kind}

    def _drive_level(self, proprio: torch.Tensor) -> torch.Tensor:
        """Keramati–Gutkin drive: convex distance from internal setpoints.

        Internal state comes straight from proprio — energy, integrity, and
        restedness (1 - fatigue), all "higher is better". Only deficits below
        setpoint count (surplus is not a drive), and the convex exponents
        (m > n) let the neediest variable dominate: a starving agent is not
        consoled by being well-rested.
        """
        x = torch.stack([proprio[..., 5], proprio[..., 6], 1.0 - proprio[..., 14]], dim=-1)
        # Clamp to the physical range: decoded proprio (boredom's gate reads
        # imagined bodies) can stray outside [0, 1].
        deficit = (self.drive_setpoints - x.clamp(0.0, 1.0)).clamp(min=0.0)
        d = (self.drive_weights * deficit.pow(self.drive_pow_m)).sum(-1)
        return d.pow(1.0 / self.drive_pow_n)

    def _homeostasis(self, events: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if self.homeostasis_mode == "drive":
            # HRRL: reward is drive *reduction*, so valence is need-relative —
            # eating while starving is worth a lot, eating past the setpoint is
            # worth nothing, and satiation needs no stop rule. Damage and
            # fatigue price themselves the same way, through the state they
            # move. A small level penalty keeps a gradient on standing
            # deficits. The first step of a sequence has no predecessor, so
            # its reduction term is zero.
            d = self._drive_level(proprio)
            reduction = torch.zeros_like(d)
            reduction[..., 1:] = d[..., :-1] - d[..., 1:]
            return self.drive_scale * reduction - self.drive_level_penalty * d
        ate, damage = events[..., 0], events[..., 1]
        energy = proprio[..., 5]
        if self.low_energy_graded:
            # Ramp, not cliff: penalty grows as energy falls, so the reward
            # head sees a gradient pointing away from zero everywhere below
            # the threshold. Binary variant kept as an ablation.
            low = ((self.low_energy_threshold - energy) / self.low_energy_threshold).clamp(0.0, 1.0)
        else:
            low = (energy < self.low_energy_threshold).float()
        return ate - damage - self.low_energy_penalty * low

    def _kind_regions_obs(self, kind_onehot: torch.Tensor) -> torch.Tensor:
        """Presence-combo region from real rays: is anything alive in view?"""
        has_robot = kind_onehot[..., RAY_KIND_ROBOT].amax(-1) > 0.5
        has_dormant = kind_onehot[..., RAY_KIND_DORMANT].amax(-1) > 0.5
        return has_robot.long() + 2 * has_dormant.long()

    def _kind_regions_img(self, feat: torch.Tensor) -> torch.Tensor:
        """Same combo on imagined states, via the decoder's kind head."""
        logits = self.wm.head_kind(feat).view(*feat.shape[:-1], self.wm.num_rays, NUM_RAY_KINDS)
        probs = torch.softmax(logits, dim=-1)
        has_robot = probs[..., RAY_KIND_ROBOT].amax(-1) > 0.5
        has_dormant = probs[..., RAY_KIND_DORMANT].amax(-1) > 0.5
        return has_robot.long() + 2 * has_dormant.long()

    def _imagination_reward(
        self, img_feat: torch.Tensor, img_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the intrinsic reward on imagined states.

        Every term is a function of latent state — LP by region lookup,
        drives via decoded proprio — so the actor can plan toward interest
        and away from boredom inside the dream. Returns (reward, stimulation,
        boredom) for the return computation and metrics.
        """
        r_homeo = self.twohot.decode(self.wm.head_reward(img_feat))
        r_dis = self.curiosity_norm.normalize(
            self.wm.disagreement(img_feat, img_action)
        ).clamp(0, 5.0)
        if self.curiosity_mode == "lp":
            if self.lp_partition == "latent":
                idx = self.regions.assign(img_feat)
            else:
                idx = self._kind_regions_img(img_feat)
            r_lp = self.lp_norm.normalize(self.lp.reward(idx)).clamp(0.0, 5.0)
            # A trickle of disagreement keeps a newborn moving before any
            # region has enough history to show progress.
            r_cur = r_lp + self._lp_mix() * r_dis
        else:
            r_cur = r_dis
        reward = self.w_homeostasis * r_homeo + self.w_curiosity * r_cur
        bored = torch.zeros_like(reward)
        if self.boredom_weight > 0:
            # Safe AND learning nothing: both gates must open. An agent in
            # need is never bored (survival is stimulation enough), and an
            # agent making progress is never bored no matter how sated.
            drive = self._drive_level(self.wm.head_proprio(img_feat))
            calm = (1.0 - drive / self.boredom_drive_threshold).clamp(min=0.0)
            dull = (1.0 - r_cur / self.boredom_stim_threshold).clamp(min=0.0)
            # Gate telemetry: the product collapses to 0 whenever either gate
            # is shut, hiding how close the other came — log them separately.
            self._gate_calm = float(calm.mean())
            self._gate_dull = float(dull.mean())
            bored = self.boredom_weight * calm * dull
            if self.boredom_pressure_on:
                # Pressure modulates the instantaneous gates: the penalty is
                # only loud once dull safety has *persisted* (integrated on
                # real experience in learn()), and the actor escapes it by
                # imagining states that shut a gate — which, lived, drains
                # the pressure.
                bored = bored * self._boredom_pressure
            reward = reward - bored
        return reward, r_cur, bored

    def _lp_mix(self) -> float:
        """Cold-start disagreement trickle, annealed out over early life."""
        if self.lp_mix_anneal_steps <= 0:
            return self.lp_mix_disagreement
        frac = 1.0 - self._act_steps / self.lp_mix_anneal_steps
        return self.lp_mix_disagreement * max(0.0, frac)

    def experience_count(self) -> int:
        return self._act_steps

    def target_train_ratio(self) -> float:
        return self.train_ratio

    def learn(self) -> dict[str, float] | None:
        batch_np = self.buffer.sample_sequences(
            self.batch_size, self.burn_in + self.seq_len, recent=self.recent_slots
        )
        if batch_np is None or len(self.buffer) < self.warmup_steps:
            return None
        learn_began = time.monotonic()
        b = {k: torch.as_tensor(v, device=self.device) for k, v in batch_np.items()}
        B = b["depth"].shape[0]
        kind_idx = b["kind"].long()
        obs = {
            "depth": b["depth"],
            "rgb": b["rgb"],
            "kind_onehot": F.one_hot(kind_idx, NUM_RAY_KINDS).float(),
            "proprio": b["proprio"],
            "sound": b["sound"],
            "events": b["events"],
        }

        # --- world model: posterior unroll over the sequence. The burn-in
        # prefix runs gradient-free purely to warm (h, z), then losses see
        # only the seq_len suffix — the recurrent state entering the graded
        # window is honest instead of the zero-state lie.
        embed = self.wm.embed(obs)  # (B, burn_in + L, units)
        h, z = self.wm.rssm.initial(B, self.device)
        zero_action = torch.zeros(B, ACTION_DIM, device=self.device)
        if self.burn_in > 0:
            with torch.no_grad():
                for t in range(self.burn_in):
                    prev_a = b["action"][:, t - 1] if t > 0 else zero_action
                    h, z, _, _ = self.wm.rssm.obs_step(h, z, prev_a, embed[:, t])
        feats, posts, priors = [], [], []
        for t in range(self.burn_in, self.burn_in + self.seq_len):
            prev_a = b["action"][:, t - 1] if t > 0 else zero_action
            h, z, post, prior = self.wm.rssm.obs_step(h, z, prev_a, embed[:, t])
            feats.append(self.wm.rssm.feat(h, z))
            posts.append(post)
            priors.append(prior)
        if self.burn_in > 0:
            # Slice every downstream target to the graded window.
            b = {k: v[:, self.burn_in :] for k, v in b.items()}
            obs = {k: v[:, self.burn_in :] for k, v in obs.items()}
            kind_idx = kind_idx[:, self.burn_in :]
            embed = embed[:, self.burn_in :]
        L = self.seq_len
        feat = torch.stack(feats, dim=1)  # (B, L, F)
        post = torch.stack(posts, dim=1)
        prior = torch.stack(priors, dim=1)

        pred_depth = self.wm.head_depth(feat)
        pred_rgb = self.wm.head_rgb(feat).view(B, L, self.wm.num_rays, 3)
        pred_kind = self.wm.head_kind(feat).view(B, L, self.wm.num_rays, NUM_RAY_KINDS)
        pred_proprio = self.wm.head_proprio(feat)
        loss_depth = F.mse_loss(pred_depth, b["depth"], reduction="none").sum(-1)
        loss_rgb = F.mse_loss(pred_rgb, b["rgb"], reduction="none").sum((-1, -2))
        loss_kind = (
            F.cross_entropy(pred_kind.flatten(0, 2), kind_idx.flatten(), reduction="none")
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
        ens_preds = self.wm.ensemble(x)  # (K, B, L-1, units)
        loss_ens = (ens_preds - ens_target).pow(2).mean(-1).mean(0)

        model_loss = (
            loss_depth + loss_rgb + loss_kind + loss_proprio + loss_reward + loss_cont + loss_kl
        ).mean() + loss_ens.mean()
        l2_dist = 0.0
        if self.l2_init_weight > 0:
            reg = torch.zeros((), device=self.device)
            for p, p0 in zip(self.wm.parameters(), self._wm_init, strict=True):
                reg = reg + (p - p0).pow(2).sum()
            model_loss = model_loss + self.l2_init_weight * reg
            l2_dist = float(reg.detach())
        self.opt_model.zero_grad()
        if self.opt_model_muon is not None:
            self.opt_model_muon.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip)
        self.opt_model.step()
        if self.opt_model_muon is not None:
            self.opt_model_muon.step()

        # Curiosity statistics on real experience (keeps normalization honest).
        lp_reward_mean = 0.0
        lp_idx: torch.Tensor | None = None
        with torch.no_grad():
            real_disagreement = self.wm.disagreement(feat[:, :-1].detach(), b["action"][:, 1:])
            self.curiosity_norm.update(real_disagreement)
            if self.curiosity_mode == "lp":
                # Fold this batch's per-sample model errors into their regions'
                # progress ledgers. Replay mixes old and new experience, so LP
                # here reads as competence progress on a region regardless of
                # when it was lived — retention counts, not just recency.
                err = loss_ens.detach()
                if self.lp_partition == "latent":
                    flat = ens_in_feat.reshape(-1, ens_in_feat.shape[-1])
                    idx = self.regions.adapt(flat)
                else:
                    idx = self._kind_regions_obs(obs["kind_onehot"][:, :-1]).reshape(-1)
                self.lp.update(idx, err)
                lp_idx = idx
                real_lp = self.lp.reward(idx)
                self.lp_norm.update(real_lp)
                lp_reward_mean = float(self.lp_norm.normalize(real_lp).clamp(0, 5.0).mean())
            if self.boredom_weight > 0 and self.boredom_pressure_on:
                # Integrate the mood on lived states, not imagined ones: how
                # much of this batch of real experience was calm AND dull?
                # Sustained gate-touching charges the pressure; lived relief
                # (either gate shutting) lets it leak away.
                r_dis_real = self.curiosity_norm.normalize(real_disagreement).clamp(0, 5.0)
                if self.curiosity_mode == "lp":
                    stim = self.lp_norm.normalize(real_lp).clamp(0.0, 5.0)
                    stim = stim + self._lp_mix() * r_dis_real.reshape(-1)
                else:
                    stim = r_dis_real.reshape(-1)
                drive = self._drive_level(b["proprio"][:, :-1]).reshape(-1)
                calm_r = (1.0 - drive / self.boredom_drive_threshold).clamp(min=0.0)
                dull_r = (1.0 - stim / self.boredom_stim_threshold).clamp(min=0.0)
                gate = float((calm_r * dull_r).mean())
                pressure = self._boredom_pressure
                pressure += (
                    self.boredom_pressure_rise * gate - self.boredom_pressure_decay * pressure
                )
                self._boredom_pressure = min(1.0, max(0.0, pressure))

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
            reward, r_cur, bored = self._imagination_reward(img_feat, img_action)
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
        elapsed = time.monotonic() - learn_began
        self._learn_seconds = (
            elapsed if self._learn_seconds == 0.0 else 0.95 * self._learn_seconds + 0.05 * elapsed
        )
        self._metrics = {
            "loss_model": float(model_loss.detach()),
            "pred_error_depth": float(loss_depth.detach().mean() / self.wm.num_rays),
            "pred_error_rgb": float(loss_rgb.detach().mean() / self.wm.num_rays),
            "pred_error_kind": float(loss_kind.detach().mean() / self.wm.num_rays),
            "kl": float(loss_kl.detach().mean()),
            "curiosity": float(real_disagreement.mean()),
            "curiosity_scaled": float(
                self.curiosity_norm.normalize(real_disagreement).clamp(0, 5).mean()
            ),
            "reward_homeostasis": float(homeo.mean()),
            # Batch mean hides meal spikes; max + spike share make the
            # drive-reward's loud instants visible next to curiosity.
            "homeo_max": float(homeo.max()),
            "homeo_spike_frac": float((homeo > 0.1).float().mean()),
            "value": float(value.mean()),
            "loss_critic": float(loss_critic.detach()),
            "loss_actor": float(loss_actor.detach()),
            "entropy": float(img_ent.detach().mean()),
            "updates": float(self._updates),
            "act_steps": float(self._act_steps),
            "train_ratio_eff": float(self._updates / max(1, self._act_steps)),
            "learn_seconds": self._learn_seconds,
            "buffer": float(len(self.buffer)),
        }
        if self.l2_init_weight > 0:
            self._metrics["l2_init_dist"] = l2_dist
        if self.homeostasis_mode == "drive":
            self._metrics["drive_level"] = float(self._drive_level(b["proprio"]).mean())
        if self.curiosity_mode == "lp":
            self._metrics["lp_reward"] = lp_reward_mean
            self._metrics["lp_regions"] = float(self.lp.regions_seen())
            if self.lp_mix_anneal_steps > 0:
                self._metrics["lp_mix_eff"] = self._lp_mix()
            # Per-region LP telemetry — the "does interest go stale where the
            # model converges" instrument. Stale = a seen region whose raw LP
            # has fallen to ~nothing; occupancy entropy says whether the batch
            # actually spreads over the partition (1 = uniform, 0 = one region).
            seen = self.lp.count > 0
            if bool(seen.any()):
                lp_seen = self.lp.lp()[seen]
                self._metrics["lp_p50"] = float(lp_seen.quantile(0.5))
                self._metrics["lp_p90"] = float(lp_seen.quantile(0.9))
                self._metrics["lp_stale_frac"] = float((lp_seen < 1e-3).float().mean())
            if lp_idx is not None:
                occ = torch.bincount(lp_idx.reshape(-1).cpu(), minlength=self.lp.n).float()
                probs = occ / occ.sum().clamp(min=1.0)
                entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
                self._metrics["lp_occ_entropy"] = float(entropy / np.log(self.lp.n))
        if self.boredom_weight > 0:
            self._metrics["boredom"] = float(bored.mean())
            self._metrics["stimulation"] = float(r_cur.mean())
            self._metrics["boredom_calm_gate"] = self._gate_calm
            self._metrics["boredom_dull_gate"] = self._gate_dull
            if self.boredom_pressure_on:
                self._metrics["boredom_pressure"] = self._boredom_pressure
        return self._metrics

    def introspect(self) -> dict[str, float]:
        out = dict(self._metrics)
        if self.temperament_enabled:
            out.update({f"temperament_{k}": v for k, v in self.temperament.items()})
        return out

    def reset_stream(self) -> None:
        """The stream broke (respawn or wake): reset live recurrent state only."""
        self.h, self.z = self.wm.rssm.initial(1, self.device)
        self.last_action = torch.zeros(1, ACTION_DIM, device=self.device)

    # ----------------------------------------------------------- persistence

    def state_dict(self) -> dict[str, Any]:
        state_muon = self.opt_model_muon.state_dict() if self.opt_model_muon else None
        return {
            "obs_version": OBS_VERSION,
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_ema": self.critic_ema.state_dict(),
            "opt_model": self.opt_model.state_dict(),
            "opt_model_muon": state_muon,
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "curiosity_norm": self.curiosity_norm.state_dict(),
            "temperament": dict(self.temperament),
            "lp_regions": self.regions.state_dict(),
            "lp_tracker": self.lp.state_dict(),
            "lp_norm": self.lp_norm.state_dict(),
            "return_scale": self._return_scale[0],
            "boredom_pressure": self._boredom_pressure,
            "updates": self._updates,
            "act_steps": self._act_steps,
            "rng_state": self.rng.bit_generator.state,
            "buffer": self.buffer.state_dict(),
            "h": self.h.cpu().numpy(),
            "z": self.z.cpu().numpy(),
            "last_action": self.last_action.cpu().numpy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("obs_version") != OBS_VERSION:
            raise ValueError(
                f"brain checkpoint has obs_version {state.get('obs_version')}, "
                f"world speaks {OBS_VERSION}: refusing to load across contract changes"
            )
        migrated = "ensemble.0.0.weight" in state["wm"]
        _migrate_ensemble_state(state["wm"])
        self.wm.load_state_dict(state["wm"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_ema.load_state_dict(state["critic_ema"])
        if not migrated:
            # A migrated ensemble reshuffles the model's param list, so the
            # stored Adam moments no longer map; the model optimizer restarts
            # fresh (offline analysis of old lives, not live resumes).
            self.opt_model.load_state_dict(state["opt_model"])
            if self.opt_model_muon is not None and state.get("opt_model_muon") is not None:
                self.opt_model_muon.load_state_dict(state["opt_model_muon"])
        self.opt_actor.load_state_dict(state["opt_actor"])
        self.opt_critic.load_state_dict(state["opt_critic"])
        self.curiosity_norm.load_state_dict(state["curiosity_norm"])
        # Keys guarded: checkpoints from before the interest/temperament work
        # load cleanly, keeping their fresh defaults for the new machinery.
        if "temperament" in state:
            self.temperament = dict(state["temperament"])
            self._apply_temperament()
        if "lp_tracker" in state:
            self.regions.load_state_dict(state["lp_regions"])
            self.lp.load_state_dict(state["lp_tracker"])
            self.lp_norm.load_state_dict(state["lp_norm"])
        self._return_scale = [float(state["return_scale"])]
        # Guarded: pre-pressure checkpoints start with a fresh mood.
        self._boredom_pressure = float(state.get("boredom_pressure", 0.0))
        self._updates = int(state["updates"])
        # Guarded: pre-pacing checkpoints carry no act-step counter; seed it
        # at the stored buffer's size so the update/act-step pair stays
        # roughly coherent.
        self._act_steps = int(state.get("act_steps", len(state["buffer"]["depth"])))
        self.rng.bit_generator.state = state["rng_state"]
        self.buffer.load_state_dict(state["buffer"])
        self.h = torch.as_tensor(state["h"], device=self.device)
        self.z = torch.as_tensor(state["z"], device=self.device)
        self.last_action = torch.as_tensor(state["last_action"], device=self.device)

    def inherit(self, state: dict[str, Any]) -> None:
        """Warm-start a newborn from a living donor: weights, memories, and
        temperament — the temperament mutated, so lineages drift through
        temperament space and transmission has something to select on."""
        self.load_state_dict(state)
        # A newborn is not born jaded: inherited knowledge keeps the donor's
        # boredom-relevant *scales* (normalizers ride state_dict), but the
        # accumulated mood itself resets with the new body.
        self._boredom_pressure = 0.0
        if self.temperament_enabled and self.temperament_mutation > 0:
            self.temperament = {
                k: v * float(np.exp(self.rng.normal(0.0, self.temperament_mutation)))
                for k, v in self.temperament.items()
            }
            self._apply_temperament()
        self.reset_stream()
