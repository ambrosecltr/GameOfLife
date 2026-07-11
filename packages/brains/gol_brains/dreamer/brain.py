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
- Temporal skills: an optional unnamed manager/worker hierarchy learns reusable
  multi-action control from self-generated controllability.
- Temperament: heritable log-normal multipliers over abstract drive knobs.

No task reward, skill label, demonstration, pretrained behavior, or designer
fitness score enters the learner.
"""

from __future__ import annotations

import threading
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

from gol_brains import feeling
from gol_brains.base import Brain
from gol_brains.dreamer.buffer import ReplayBuffer
from gol_brains.dreamer.dynamics import CategoricalDynamicsConfig, CategoricalLatentDynamics
from gol_brains.dreamer.inference import InferenceSnapshot
from gol_brains.dreamer.interest import LearningProgress, OnlineRegions
from gol_brains.dreamer.networks import (
    DiscreteDist,
    EnsembleMLP,
    RunningMeanStd,
    TanhNormal,
    TwoHot,
    bounded_policy_std,
    mlp,
    percentile_scale,
)
from gol_brains.dreamer.optim import Muon
from gol_brains.dreamer.rssm import RSSM, RSSMConfig
from gol_brains.dreamer.skills import TemporalSkillController, TemporalSkillPolicy
from gol_brains.precision import PrecisionMode, PrecisionPolicy, register_process_precision

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

# Separate value functions prevent unlike endogenous signals from erasing one
# another before the actor chooses how to trade them off.
AFFECT_NAMES = ("comfort", "viability", "curiosity", "boredom", "fear", "skill")
PAIN_AFFECT = "pain"


class WorldModel(nn.Module):
    def __init__(
        self,
        preset: dict[str, int],
        num_rays: int,
        wm_cfg: dict[str, Any],
        dynamics: CategoricalLatentDynamics | None = None,
    ) -> None:
        super().__init__()
        self.num_rays = num_rays
        units = preset["units"]
        obs_dim = num_rays * RAY_DIM + PROPRIO_DIM + SOUND_DIM + EVENTS_DIM
        self.encoder = mlp(obs_dim, units, units, layers=2)
        if dynamics is None:
            rssm_cfg = RSSMConfig(
                deter=preset["deter"],
                stoch_groups=preset["groups"],
                stoch_classes=preset["classes"],
                hidden=preset["hidden"],
                unimix=float(wm_cfg.get("unimix", 0.01)),
                free_bits=float(wm_cfg.get("kl_free_bits", 1.0)),
            )
            dynamics = RSSM(rssm_cfg, embed_dim=units, action_dim=ACTION_DIM)
        self.rssm = dynamics
        self.rssm_cfg = dynamics.cfg
        feat = self.rssm_cfg.feat_dim
        self.head_depth = mlp(feat, units, num_rays, layers=2)
        self.head_rgb = mlp(feat, units, num_rays * 3, layers=2)
        self.head_kind = mlp(feat, units, num_rays * NUM_RAY_KINDS, layers=2)
        self.head_proprio = mlp(feat, units, PROPRIO_DIM, layers=2)
        self.head_reward = mlp(feat, units, 41, layers=2)  # twohot homeostasis
        self.head_cont = mlp(feat, units, 1, layers=2)
        self.head_damage = (
            mlp(feat, units, 1, layers=2) if bool(wm_cfg.get("predict_damage", False)) else None
        )
        # Plan2Explore: each ensemble member predicts the NEXT observation
        # embedding from (state, action); their disagreement is epistemic
        # uncertainty, which is the curiosity signal. Members are stacked into
        # one batched module (K einsums, not K module calls); pre-swift
        # checkpoints stored a ModuleList and are migrated on load.
        k = int(wm_cfg.get("ensemble_size", 8))
        self.ensemble = EnsembleMLP(k, feat + ACTION_DIM, units, units)

    @property
    def dynamics(self) -> CategoricalLatentDynamics:
        """Architecture-neutral dynamics access; `rssm` keeps checkpoint keys stable."""
        return self.rssm

    @property
    def dynamics_cfg(self) -> CategoricalDynamicsConfig:
        return self.rssm_cfg

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
        """Plan2Explore intrinsic signal: variance across ensemble predictions.

        Variance via two means (E[x^2] - E[x]^2, Bessel-corrected): reductions
        over the leading dim vectorize, while Tensor.var(dim=0) takes a slow
        CPU path that alone cost ~40% of a nano learn() (measured 59ms/call).
        """
        x = torch.cat([feat, action], dim=-1)
        preds = self.ensemble(x)  # (K, ..., units)
        k = preds.shape[0]
        var = (preds.pow(2).mean(0) - preds.mean(0).pow(2)) * (k / (k - 1))
        return var.clamp(min=0.0).mean(-1)


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
    members = [
        {
            key: wm_state.pop(f"ensemble.{j}.{key}")
            for key in ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias")
        }
        for j in range(k)
    ]
    wm_state["ensemble.w1"] = torch.stack([m["0.weight"].T for m in members])
    wm_state["ensemble.b1"] = torch.stack([m["0.bias"] for m in members])
    wm_state["ensemble.ln_w"] = torch.stack([m["1.weight"] for m in members])
    wm_state["ensemble.ln_b"] = torch.stack([m["1.bias"] for m in members])
    wm_state["ensemble.w2"] = torch.stack([m["3.weight"].T for m in members])
    wm_state["ensemble.b2"] = torch.stack([m["3.bias"] for m in members])


class DreamerBrain(Brain):
    brain_family = "dreamer"

    def _build_world_model(
        self, preset: dict[str, int], num_rays: int, wm_cfg: dict[str, Any]
    ) -> WorldModel:
        return WorldModel(preset, num_rays, wm_cfg)

    def _migrate_world_model_state(self, state: dict[str, torch.Tensor]) -> bool:
        migrated = "ensemble.0.0.weight" in state
        _migrate_ensemble_state(state)
        return migrated

    def __init__(
        self, cfg: dict[str, Any], seed: int, device: str = "cpu", body: BodySpec | None = None
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        tr = dict(cfg.get("training", {}))
        self.precision = PrecisionPolicy.from_config(tr, self.device)
        self.body = body or BodySpec()
        self._learn_lock = threading.Lock()
        self._experience_lock = threading.Lock()
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        preset = PRESETS[str(cfg.get("preset", "nano"))]
        rw = dict(cfg.get("reward", {}))
        pain = dict(rw.get("pain", {}))
        self.pain_weight = float(pain.get("weight", 0.0))
        self.pain_event_loss_weight = float(pain.get("event_loss_weight", 32.0))
        if self.pain_weight < 0.0:
            raise ValueError("reward.pain.weight cannot be negative")
        if self.pain_event_loss_weight < 1.0:
            raise ValueError("reward.pain.event_loss_weight must be at least 1")
        self.pain_on = self.pain_weight > 0.0
        wm_cfg = dict(cfg.get("world_model", {}))
        wm_cfg["predict_damage"] = self.pain_on
        self.wm = self._build_world_model(preset, self.body.num_rays, wm_cfg).to(self.device)
        feat = self.wm.dynamics_cfg.feat_dim
        units = preset["units"]

        ac_cfg = dict(cfg.get("actor_critic", {}))
        self.horizon = int(ac_cfg.get("imagination_horizon", 15))
        self.gamma = float(ac_cfg.get("gamma", 0.997))
        self.lam = float(ac_cfg.get("lam", 0.95))
        self.entropy_scale = float(ac_cfg.get("entropy", 3e-4))
        self.vector_critic = bool(ac_cfg.get("vector_critic", False))
        self.affect_names = AFFECT_NAMES + ((PAIN_AFFECT,) if self.pain_on else ())
        self.value_channels = len(self.affect_names) if self.vector_critic else 1

        skill_cfg = dict(cfg.get("temporal_skills", {}))
        self.skills_enabled = bool(skill_cfg.get("enabled", False))
        self.skill_duration = int(skill_cfg.get("duration", 5))
        if self.skills_enabled and self.skill_duration < 2:
            raise ValueError("temporal_skills.duration must be at least 2")
        self.skill_weight = float(skill_cfg.get("intrinsic_weight", 0.0))
        self.skill_manager_entropy = float(skill_cfg.get("manager_entropy", 3e-4))
        self.num_skills = int(skill_cfg.get("num_skills", 8))
        if self.skills_enabled:
            self.actor: nn.Module = TemporalSkillPolicy(
                feat,
                units,
                CONT_DIM,
                NUM_GRIP_MODES,
                self.num_skills,
                unimix=float(skill_cfg.get("unimix", 0.01)),
            ).to(self.device)
        else:
            self.actor = mlp(feat, units, CONT_DIM * 2 + NUM_GRIP_MODES, layers=2).to(self.device)
        critic_out = 41 * self.value_channels
        self.critic = mlp(feat, units, critic_out, layers=2).to(self.device)
        self.critic_ema = mlp(feat, units, critic_out, layers=2).to(self.device)
        self.critic_ema.load_state_dict(self.critic.state_dict())
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)
        self.twohot = TwoHot().to(self.device)

        self.w_curiosity = float(rw.get("w_curiosity", 1.0))
        self.w_homeostasis = float(rw.get("w_homeostasis", 1.0))
        # Legacy brains ask a reward head to rediscover the body's known
        # interoceptive equations. The organism path instead applies those
        # equations directly to predicted proprioception in imagination; the
        # learned reward head remains as an auditable diagnostic target.
        self.imagined_homeostasis = str(rw.get("imagined_homeostasis", "head"))
        if self.imagined_homeostasis not in ("head", "proprio"):
            raise ValueError(f"unknown reward.imagined_homeostasis: {self.imagined_homeostasis!r}")
        self.terminal_loss_weight = float(rw.get("terminal_loss_weight", 1.0))
        if self.terminal_loss_weight < 1.0:
            raise ValueError("reward.terminal_loss_weight must be at least 1")
        self.fear_weight = float(rw.get("fear_weight", 0.0))
        # "drive": HRRL drive-reduction over internal state (energy, integrity,
        # rest). "events": the legacy ate/damage bonus, kept as an ablation and
        # as the default so configs stored in older saves keep the reward their
        # brains were trained on.
        self.homeostasis_mode = str(rw.get("homeostasis", "events"))
        if self.homeostasis_mode not in ("drive", "events"):
            raise ValueError(f"unknown homeostasis mode: {self.homeostasis_mode!r}")
        if self.pain_on and (
            self.homeostasis_mode != "drive" or self.imagined_homeostasis != "proprio"
        ):
            raise ValueError(
                "reward.pain requires homeostasis: drive and imagined_homeostasis: proprio"
            )
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
        # Round-012 viability drive: a log-barrier on distance to the LETHAL
        # floor (energy→dormancy, integrity→true death), added to the HRRL
        # comfort drive. The comfort drive is convex distance to a comfort
        # SETPOINT (0.85 energy is nice); viability is unbounded distance to a
        # BOUNDARY (0 energy is annihilation), so the marginal value of a
        # calorie explodes as you starve and being deep in the danger zone
        # carries a standing cost the comfort drive never imposes. It rewards
        # no action and names no behaviour — a pure function of internal state,
        # like every other drive. With reduction, tax, and wellbeing all zero,
        # the viability channel is off and legacy behavior is exact.
        via = dict(rw.get("viability", {}))
        self.via_scale = float(via.get("scale", 0.0))
        self.via_floor = float(via.get("floor", 0.0))  # standing cost on V (the danger-zone tax)
        self.via_barrier_cap = float(via.get("barrier_cap", 4.0))  # keep −log finite at the floor
        self.via_total_cap = float(via.get("total_cap", 0.0))
        self.via_e_lethal = float(via.get("energy_lethal", 0.0))
        self.via_e_safe = float(via.get("energy_safe", 0.25))  # normalized brownout threshold
        self.via_i_lethal = float(via.get("integrity_lethal", 0.0))
        self.via_i_safe = float(via.get("integrity_safe", 0.5))
        self.via_w_energy = float(via.get("energy_weight", 1.0))
        self.via_w_integ = float(via.get("integrity_weight", 1.0))
        wellbeing = dict(rw.get("wellbeing", {}))
        self.wellbeing_weight = float(wellbeing.get("weight", 0.0))
        self.wellbeing_comfort_decay = float(wellbeing.get("comfort_decay", 1.0))
        if self.wellbeing_weight < 0.0:
            raise ValueError("reward.wellbeing.weight cannot be negative")
        if self.wellbeing_comfort_decay < 0.0:
            raise ValueError("reward.wellbeing.comfort_decay cannot be negative")
        self.wellbeing_on = self.wellbeing_weight > 0.0
        # The drive is "on" if reduction, standing tax, or regulated-body
        # wellbeing uses the boundary. Offline
        # calibration on dreamer_042 showed the reduction term alone recreates
        # the hibernation attractor — near-floor states become high-value
        # launchpads because escaping pays — so the standing cost carries the
        # mortality gradient; both are kept as separable knobs.
        self.via_on = self.via_scale > 0.0 or self.via_floor > 0.0 or self.wellbeing_on
        if self.via_on and self.homeostasis_mode != "drive":
            raise ValueError("reward.viability requires homeostasis: drive")
        # True death (integrity → lethal) terminates the imagined stream so its
        # absorbing ~0 return backs up through the critic — a functional fear of
        # death from prediction, though the body is never actually experienced
        # dying (invariant: no episodes). Recoverable dormancy (the energy
        # floor) stays non-terminal — fear the slope, not the sleep. false =
        # beta_10 (blackout flag decides cont alone).
        self.death_terminal = bool(rw.get("death_terminal", False))
        # Round-011 reachability: how the dormancy blackout enters the learned
        # stream. "cut" (legacy) severs it — wake resets the live state and
        # the salience chain, and near-zero energy trains the continuation
        # head to 0, so imagination discount-terminates at the crash.
        # "priced" makes the blackout one visible transition (the pre-collapse
        # state is the predecessor of the wake observation, carrying the gap's
        # real energy/integrity delta) so HRRL prices the crash with no new
        # reward terms. Four decisions, each deliberate:
        #   1. salience survives the wake (the delta is a real spike, so
        #      reward-aware replay can find every blackout);
        #   2. the buffer keeps the pair adjacent (true in both modes — add()
        #      never breaks — so replayed windows can span the gap);
        #   3. the live recurrent state still resets (the mind was off; only
        #      the *consequence* was lived, not the gap);
        #   4. blackout is no longer a termination: the continuation head
        #      trains to 1 everywhere, since a survivable transition that
        #      imagination discount-kills at would leave the priced crash
        #      unreachable by the actor. Death stays unexperienced (the
        #      stream just stops); that is a deliberately separate gap.
        # "suspended" keeps elapsed S5 context but severs affect across the
        # unconscious gap and trains gamma**elapsed continuation. Respawn into
        # a new body remains a hard cut in every mode.
        self.blackout = str(rw.get("blackout", "cut"))
        if self.blackout not in ("cut", "priced", "suspended"):
            raise ValueError(f"unknown blackout mode: {self.blackout!r}")
        if self.blackout in ("priced", "suspended") and self.homeostasis_mode != "drive":
            raise ValueError(
                f"blackout: {self.blackout} requires homeostasis: drive"
            )
        # Spike-weighted reward loss (pre-registered 010/011 follow-up): the
        # twohot head sees so few |reward| spikes that even oversampled meals
        # can drown in the batch mean; weight > 0 multiplies spike samples'
        # reward loss by (1 + weight). 0 = legacy unweighted loss.
        self.spike_loss_weight = float(rw.get("spike_loss_weight", 0.0))
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
        # What "calm" (safe enough to be bored) reads. "drive" (beta_10): the
        # comfort drive, so any deficit below setpoint shuts the boredom gate —
        # which couples boredom to hunger, the round-011 concern (an agent a
        # little peckish can't be bored). "viability": read the barrier, so an
        # agent far from the lethal floor is safe-to-be-bored even while below
        # its comfort setpoint — boredom-play decouples from mere hunger and
        # only true danger shuts the gate. Read against the same threshold.
        self.boredom_gate = str(bd.get("gate", "drive"))
        if self.boredom_gate not in ("drive", "viability"):
            raise ValueError(f"unknown boredom gate: {self.boredom_gate!r}")
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
        capacity = int(replay.get("capacity", 100_000))
        self.batch_size = int(replay.get("batch_size", 16))
        self.seq_len = int(replay.get("seq_len", 64))
        # ~2 sim-minutes of motor babbling; long warmups starve newborns.
        self.warmup_steps = int(replay.get("warmup_steps", 500))
        # Burn-in: prefix steps unrolled without gradients purely to warm the
        # recurrent state, so seq_len (the expensive grad-carrying part of the
        # unroll) can shrink without every sequence starting from a zero-state
        # lie. 0 = legacy zero-init.
        self.burn_in = int(replay.get("burn_in", 0))
        if capacity < 1 or self.batch_size < 1 or self.seq_len < 1 or self.warmup_steps < 1:
            raise ValueError(
                "replay capacity, batch_size, seq_len, and warmup_steps must be positive"
            )
        if self.burn_in < 0:
            raise ValueError("replay.burn_in cannot be negative")
        required_steps = self.burn_in + self.seq_len + 2
        if self.warmup_steps < required_steps:
            raise ValueError(
                "replay.warmup_steps must be at least "
                f"replay.burn_in + replay.seq_len + 2 ({required_steps})"
            )
        if capacity < required_steps:
            raise ValueError(
                "replay.capacity must be at least "
                f"replay.burn_in + replay.seq_len + 2 ({required_steps})"
            )
        if capacity < self.warmup_steps:
            raise ValueError("replay.capacity must be at least replay.warmup_steps")
        self.buffer = ReplayBuffer(
            capacity=capacity,
            num_rays=self.body.num_rays,
            action_dim=ACTION_DIM,
            seed=seed + 1,
        )
        # Batch rows pinned to the newest experience (DreamerV3 online-queue
        # mixing); the rest sample uniformly over the whole life. 0 = legacy.
        self.recent_slots = int(replay.get("recent", 0))
        # Reward-aware replay (round 009's reachability finding): "reward"
        # draws prioritize_rows of the batch from windows containing an
        # ate/damage event so the twohot reward head actually trains on the
        # rare loud moments. Learned-from changes; rewarded never does.
        self.prioritize = str(replay.get("prioritize", "none"))
        if self.prioritize not in ("none", "reward"):
            raise ValueError(f"unknown replay prioritize mode: {self.prioritize!r}")
        self.prioritize_rows = (
            int(replay.get("prioritize_rows", max(1, self.batch_size // 4)))
            if self.prioritize == "reward"
            else 0
        )
        # A step is salient when |realized homeostasis reward| clears this
        # (same bar as the homeo_spike_frac metric). Salience — not event
        # flags — is the priority signal: under HRRL drive reward a meal at
        # satiety is an ate event worth zero (swift_01 measured exactly that).
        self.prioritize_threshold = float(replay.get("prioritize_threshold", 0.1))
        self._prev_drive: float | None = None  # act-stream drive level, for salience
        self._prev_via: float | None = None  # act-stream barrier level, for salience
        self._prev_integrity: float | None = None
        # Exact per-life realized return, accumulated on the lived stream (the
        # replay batch mean can only infer the sign). Reset at a stream break;
        # exposed via introspect so metrics.ndjson carries the running integral.
        self._life_return_homeo = 0.0
        self._life_return_via = 0.0
        self._life_return_pain = 0.0
        # The next recorded step has no lived predecessor (fresh mind, respawn,
        # or a cut wake): it lands in the buffer as a stream-break marker so
        # replayed windows never read a fictional drive delta across the gap.
        self._stream_first = True
        self._stream_wake = False
        self._step_scale = 1.0

        model_lr = float(tr.get("model_lr", 1e-4))
        self.optimizer_kind = str(tr.get("optimizer", "adam"))
        if self.optimizer_kind not in ("adam", "muon"):
            raise ValueError(f"unknown optimizer: {self.optimizer_kind!r}")
        self.opt_model_muon: Muon | None = None
        if self.optimizer_kind == "muon":
            # Muon conditions exactly-2D weight matrices; biases, LayerNorms,
            # and the stacked 3D ensemble stay on Adam (the standard recipe).
            matrices = [p for p in self.wm.parameters() if p.dim() == 2 and not p.is_complex()]
            rest = [p for p in self.wm.parameters() if p.dim() != 2 or p.is_complex()]
            self.opt_model_muon = Muon(matrices, lr=float(tr.get("muon_lr", 0.02)))
            self.opt_model = torch.optim.Adam(rest, lr=model_lr, foreach=True)
        else:
            self.opt_model = torch.optim.Adam(self.wm.parameters(), lr=model_lr, foreach=True)
        # foreach batches the per-param step into fused multi-tensor kernels
        # (~3x fewer optimizer-step ops; measured ~12ms -> ~4ms per update).
        self._actor_parameters = (
            list(self.actor.controller_parameters())
            if isinstance(self.actor, TemporalSkillPolicy)
            else list(self.actor.parameters())
        )
        self.opt_actor = torch.optim.Adam(
            self._actor_parameters, lr=float(tr.get("actor_lr", 3e-5)), foreach=True
        )
        self.opt_skill: torch.optim.Adam | None = None
        if isinstance(self.actor, TemporalSkillPolicy):
            self.opt_skill = torch.optim.Adam(
                self.actor.discriminator.parameters(),
                lr=float(tr.get("skill_lr", 1e-4)),
                foreach=True,
            )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=float(tr.get("critic_lr", 3e-5)), foreach=True
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
        if self.train_ratio < 0.0:
            raise ValueError("training.train_ratio cannot be negative")
        self.async_inference = bool(tr.get("async_inference", False))
        self.publish_every = int(tr.get("publish_every", 16))
        if self.publish_every < 1:
            raise ValueError("training.publish_every must be at least 1")

        # Live recurrent state (the robot's stream of consciousness).
        self.h, self.z = self.wm.dynamics.initial(1, self.device)
        self.last_action = torch.zeros(1, ACTION_DIM, device=self.device)
        self._return_scale = [1.0]
        self._metrics: dict[str, float] = {}
        self._updates = 0
        self._act_steps = 0
        self._schedule_credit_origin = 0.0
        self._dropped_update_credit = 0.0
        self._learn_seconds = 0.0  # EMA of learn() wall time
        self._gate_calm = 0.0  # boredom gate telemetry (imagination means)
        self._gate_dull = 0.0
        self._boredom_pressure = 0.0  # leaky-integrated mood (pressure mode)
        self._online_stimulation = self.boredom_stim_threshold
        self._active_skill = -1
        self._skill_remaining = 0
        self._skill_switches = 0
        self._inference: InferenceSnapshot | None = None
        self._published_updates = 0
        register_process_precision(self, self.precision)

        # torch.compile on the sequential hot loops (RSSM steps + encoder):
        # nano-scale unrolls are dispatch-bound, and compilation collapses the
        # per-step Python/kernel-launch overhead. The act-path shape (B=1) is
        # precompiled here so the sim thread never eats a compile stall
        # mid-life; the learn-path shapes compile lazily on the learner
        # thread, where a one-time wall-clock pause leaves earned credit intact.
        compile_enabled = bool(tr.get("compile", False))
        if compile_enabled and self.async_inference:
            raise ValueError("training.compile is not compatible with async_inference snapshots")
        if compile_enabled:
            backend = "aot_eager" if self.device.type == "mps" else "inductor"
            for owner, name in (
                (self.wm.dynamics, "obs_step"),
                (self.wm.dynamics, "img_step"),
                (self.wm, "embed"),
            ):
                setattr(owner, name, torch.compile(getattr(owner, name), backend=backend))
            with torch.no_grad():
                embed = torch.zeros(1, preset["units"], device=self.device)
                self.wm.dynamics.obs_step(self.h, self.z, self.last_action, embed)
        if self.async_inference:
            self._publish_inference()

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

    def _publish_inference(self) -> None:
        """Atomically publish a frozen controller snapshot to the sim thread."""
        self._inference = InferenceSnapshot(self.wm.encoder, self.wm.dynamics, self.actor)
        self._published_updates = self._updates

    def allows_concurrent_learning(self) -> bool:
        return self.async_inference

    def precision_mode(self) -> str:
        return self.precision.mode.value

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

    def _policy_dists(
        self,
        feat: torch.Tensor,
        actor: nn.Module | None = None,
        skill: torch.Tensor | None = None,
    ) -> tuple[TanhNormal, DiscreteDist]:
        policy = actor or self.actor
        if isinstance(policy, TemporalSkillController):
            if skill is None:
                raise ValueError("temporal worker requires a skill")
            return policy.action_dists(feat, skill)
        out = policy(feat).float()
        mean = out[..., :CONT_DIM]
        raw_std = out[..., CONT_DIM : 2 * CONT_DIM]
        grip_logits = out[..., 2 * CONT_DIM :]
        std = bounded_policy_std(raw_std)
        probs = torch.softmax(grip_logits, dim=-1)
        probs = 0.99 * probs + 0.01 / NUM_GRIP_MODES  # unimix keeps exploration alive
        return TanhNormal(torch.tanh(mean), std), DiscreteDist(probs)

    def _critic_logits(self, critic: nn.Module, feat: torch.Tensor) -> torch.Tensor:
        logits = critic(feat)
        if self.vector_critic:
            return logits.view(*feat.shape[:-1], self.value_channels, self.twohot.bins)
        return logits

    def _critic_value(self, critic: nn.Module, feat: torch.Tensor) -> torch.Tensor:
        return self.twohot.decode(self._critic_logits(critic, feat))

    def _update_boredom_pressure(self, proprio: torch.Tensor) -> None:
        """Advance boredom on the chronological lived stream.

        Learning publishes its latest stimulation estimate; the actual body
        supplies safety. Replay order can therefore improve the estimate but
        cannot rewrite the organism's mood history.
        """
        if not (self.boredom_weight > 0 and self.boredom_pressure_on):
            return
        safety = (
            self._viability(proprio)
            if self.boredom_gate == "viability"
            else self._drive_level(proprio)
        )
        calm = float((1.0 - safety / self.boredom_drive_threshold).clamp(min=0.0)[0])
        dull = max(0.0, 1.0 - self._online_stimulation / self.boredom_stim_threshold)
        gate = calm * dull
        pressure = self._boredom_pressure
        pressure += self.boredom_pressure_rise * gate - self.boredom_pressure_decay * pressure
        self._boredom_pressure = min(1.0, max(0.0, pressure))

    def _action_to_vec(self, cont: torch.Tensor, grip: int) -> npt.NDArray[np.float32]:
        vec = np.zeros(ACTION_DIM, dtype=np.float32)
        vec[:CONT_DIM] = cont.detach().cpu().numpy()
        vec[CONT_DIM + grip] = 1.0
        return vec

    def act(self, obs: Observation) -> Action:
        with torch.no_grad():
            tensors = self._obs_to_tensors(obs)
            self._update_boredom_pressure(tensors["proprio"])
            # Reward salience of this lived step (recorded even when
            # prioritization is off, so an old life can turn it on later).
            # The first step after a stream break reads 0: the gap's drive
            # delta was never experienced (same rule as reset_stream).
            if self.homeostasis_mode == "drive":
                d = float(self._drive_level(tensors["proprio"])[0])
                if self._prev_drive is not None:
                    salience = abs(self.drive_scale * (self._prev_drive - d))
                    self._life_return_homeo += self.drive_scale * (self._prev_drive - d)
                else:
                    salience = 0.0
                # Level terms accrue every lived step (reduction masks at a
                # stream break, the standing costs don't — same contract as
                # _homeostasis/_viability_reward), so the integral is exact.
                self._life_return_homeo -= self.drive_level_penalty * d
                self._prev_drive = d
                if self.via_on:
                    # The barrier's realized reward rides the lived stream so
                    # near-death moments are salient to prioritized replay.
                    # BOTH terms: the delta (scale) and the standing tax
                    # (floor·V) — the staged form is floor-only, and without
                    # the tax the barrier added zero replay priority.
                    v = float(self._viability(tensors["proprio"])[0])
                    if self._prev_via is not None:
                        salience += abs(self.via_scale * (self._prev_via - v)) + self.via_floor * v
                        self._life_return_via += self.via_scale * (self._prev_via - v)
                    self._life_return_via -= self.via_floor * v
                    if self.wellbeing_on:
                        wb = float(
                            feeling.wellbeing(
                                tensors["proprio"].new_tensor(v),
                                tensors["proprio"].new_tensor(d),
                                weight=self.wellbeing_weight,
                                barrier_cap=self.via_barrier_cap,
                                comfort_decay=self.wellbeing_comfort_decay,
                            )
                        )
                        self._life_return_via += wb
                    self._prev_via = v
                integrity = float(tensors["proprio"][0, feeling.INTEGRITY_IDX])
                if (
                    self.pain_on
                    and self._prev_integrity is not None
                    and float(tensors["events"][0, 1]) > 0.5
                ):
                    pain = self.pain_weight * max(0.0, self._prev_integrity - integrity)
                    salience += pain
                    self._life_return_pain -= pain
                self._prev_integrity = integrity
            else:
                salience = float(obs["events"][0] + obs["events"][1])
            with self.precision.autocast():
                inference = self._inference
                controller = inference if inference is not None else self.wm
                policy_actor = inference.actor if inference is not None else self.actor
                embed = controller.embed(tensors)
                self.h, self.z, _, _ = controller.dynamics.obs_step(
                    self.h, self.z, self.last_action, embed, step_scale=self._step_scale
                )
                skill = -1
                if len(self.buffer) < self.warmup_steps:
                    cont = torch.as_tensor(
                        self.rng.uniform(-1, 1, CONT_DIM).astype(np.float32),
                        device=self.device,
                    )
                    grip = int(self.rng.integers(0, NUM_GRIP_MODES))
                else:
                    feat = controller.dynamics.feat(self.h, self.z)
                    skill_tensor: torch.Tensor | None = None
                    if isinstance(policy_actor, TemporalSkillController):
                        if self._skill_remaining <= 0:
                            previous = self._active_skill
                            self._active_skill = int(policy_actor.manager_dist(feat).sample()[0])
                            self._skill_remaining = self.skill_duration
                            if previous >= 0 and previous != self._active_skill:
                                self._skill_switches += 1
                        skill = self._active_skill
                        skill_tensor = torch.tensor([skill], device=self.device)
                        self._skill_remaining -= 1
                    dist_cont, dist_grip = self._policy_dists(
                        feat, actor=policy_actor, skill=skill_tensor
                    )
                    cont = dist_cont.sample()[0]
                    grip = int(dist_grip.sample()[0])
            self.h = self.h.float()
            self.z = self.z.float()

        action_vec = self._action_to_vec(cont, grip)
        with self._experience_lock:
            self.buffer.add(
                obs,
                action_vec,
                salience=salience,
                first=self._stream_first,
                wake=self._stream_wake,
                step_scale=self._step_scale,
                skill=skill,
            )
            self._act_steps += 1
        self._stream_first = False
        self._stream_wake = False
        self._step_scale = 1.0
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
        """Keramati–Gutkin comfort drive (shared definition, gol_brains.feeling)."""
        return feeling.drive_level(
            proprio, self.drive_setpoints, self.drive_weights, self.drive_pow_m, self.drive_pow_n
        )

    def _viability(self, proprio: torch.Tensor) -> torch.Tensor:
        """Log-barrier distance to the lethal floor (shared, gol_brains.feeling)."""
        return feeling.viability(
            proprio,
            barrier_cap=self.via_barrier_cap,
            total_cap=self.via_total_cap,
            energy_lethal=self.via_e_lethal,
            energy_safe=self.via_e_safe,
            integrity_lethal=self.via_i_lethal,
            integrity_safe=self.via_i_safe,
            energy_weight=self.via_w_energy,
            integrity_weight=self.via_w_integ,
        )

    def _viability_reward(
        self, proprio: torch.Tensor, first: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Boundary reduction/tax plus optional regulated-body wellbeing."""
        V = self._viability(proprio)
        reward = self.via_scale * feeling.reduction(V, first) - self.via_floor * V
        if self.wellbeing_on:
            reward = reward + feeling.wellbeing(
                V,
                self._drive_level(proprio),
                weight=self.wellbeing_weight,
                barrier_cap=self.via_barrier_cap,
                comfort_decay=self.wellbeing_comfort_decay,
            )
        return reward

    def _pain_reward(
        self,
        events: torch.Tensor,
        proprio: torch.Tensor,
        discontinuity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.pain_on:
            return torch.zeros_like(proprio[..., 0])
        loss = feeling.acute_integrity_loss(proprio, events[..., 1], discontinuity)
        return -self.pain_weight * loss

    def _reward_discontinuity(
        self, first: torch.Tensor, wake: torch.Tensor
    ) -> torch.Tensor:
        if self.blackout == "suspended":
            return torch.maximum(first, wake)
        return first

    def _homeostasis(
        self, events: torch.Tensor, proprio: torch.Tensor, first: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.homeostasis_mode == "drive":
            # HRRL: reward is drive *reduction*, so valence is need-relative —
            # eating while starving is worth a lot, eating past the setpoint is
            # worth nothing, and satiation needs no stop rule. Damage and
            # fatigue price themselves the same way, through the state they
            # move. A small level penalty keeps a gradient on standing
            # deficits. The first step of a sequence has no predecessor, so
            # its reduction term is zero — and so does a stream-break step
            # (`first`): without the mask a window spanning a respawn pays the
            # newborn's full tank as a +3.9 "reduction" (measured on beta_09's
            # dreamer_043; a real meal is +0.5). Priced blackouts don't mark
            # the wake, so their true cross-gap delta stays in.
            d = self._drive_level(proprio)
            return self.drive_scale * feeling.reduction(d, first) - self.drive_level_penalty * d
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

    def _continuation_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weight = torch.where(
            target < 0.5,
            torch.full_like(target, self.terminal_loss_weight),
            torch.ones_like(target),
        )
        return raw, raw * weight

    def _continuation_target(
        self, proprio: torch.Tensor, step_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        death = proprio[..., feeling.INTEGRITY_IDX] <= self.via_i_lethal + 1e-3
        if self.death_terminal:
            target = (~death).float()
        elif self.blackout in ("priced", "suspended"):
            target = torch.ones_like(step_scale)
        else:
            target = (proprio[..., feeling.ENERGY_IDX] > 0.01).float()
        if self.blackout == "suspended":
            elapsed_discount = torch.pow(
                torch.full_like(step_scale, self.gamma),
                (step_scale - 1.0).clamp_min(0.0),
            )
            target = target * elapsed_discount
        return target, death

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

    def _imagination_affect(
        self,
        policy_feat: torch.Tensor,
        outcome_feat: torch.Tensor,
        img_action: torch.Tensor,
        skill_reward: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Endogenous affect components for one imagined transition.

        ``policy_feat`` is the state that chose the action; ``outcome_feat``
        is the resulting state. This alignment matters for direct
        interoception: the action is valued by the bodily change it predicts.
        """
        outcome_proprio = self.wm.head_proprio(outcome_feat)
        if self.imagined_homeostasis == "proprio":
            policy_proprio = self.wm.head_proprio(policy_feat)
            drive_before = self._drive_level(policy_proprio)
            drive_after = self._drive_level(outcome_proprio)
            r_comfort = self.drive_scale * (drive_before - drive_after)
            r_comfort = r_comfort - self.drive_level_penalty * drive_after
            if self.via_on:
                via_before = self._viability(policy_proprio)
                via_after = self._viability(outcome_proprio)
                r_viability = self.via_scale * (via_before - via_after)
                r_viability = r_viability - self.via_floor * via_after
                if self.wellbeing_on:
                    r_viability = r_viability + feeling.wellbeing(
                        via_after,
                        drive_after,
                        weight=self.wellbeing_weight,
                        barrier_cap=self.via_barrier_cap,
                        comfort_decay=self.wellbeing_comfort_decay,
                    )
            else:
                r_viability = torch.zeros_like(r_comfort)
            if self.pain_on:
                if self.wm.head_damage is None:
                    raise RuntimeError("pain-enabled world model has no damage head")
                damage_probability = torch.sigmoid(
                    self.wm.head_damage(outcome_feat).squeeze(-1).float()
                )
                integrity_loss = (
                    policy_proprio[..., feeling.INTEGRITY_IDX]
                    - outcome_proprio[..., feeling.INTEGRITY_IDX]
                ).clamp_min(0.0)
                r_pain = -self.pain_weight * damage_probability * integrity_loss
            else:
                r_pain = torch.zeros_like(r_comfort)
        else:
            # Backward-compatible ablation: this head was trained on the sum
            # of comfort and viability, so keep it in one channel.
            r_comfort = self.twohot.decode(self.wm.head_reward(outcome_feat))
            r_viability = torch.zeros_like(r_comfort)
            r_pain = torch.zeros_like(r_comfort)
        if self.curiosity_mode == "lp":
            if self.lp_partition == "latent":
                idx = self.regions.assign(policy_feat)
            else:
                idx = self._kind_regions_img(policy_feat)
            r_cur = self.lp_norm.normalize(self.lp.reward(idx)).clamp(0.0, 5.0)
            # A trickle of disagreement keeps a newborn moving before any
            # region has enough history to show progress. Once the trickle
            # has annealed to zero the ensemble pass over imagined states is
            # a multiply-by-zero — skip it (it was ~1/4 of an adult update).
            mix = self._lp_mix()
            if mix > 0.0:
                r_dis = self.curiosity_norm.normalize(
                    self.wm.disagreement(policy_feat, img_action)
                ).clamp(0, 5.0)
                r_cur = r_cur + mix * r_dis
        else:
            r_cur = self.curiosity_norm.normalize(
                self.wm.disagreement(policy_feat, img_action)
            ).clamp(0, 5.0)
        bored = torch.zeros_like(r_comfort)
        if self.boredom_weight > 0:
            # Safe AND learning nothing: both gates must open. An agent in
            # danger is never bored (survival is stimulation enough), and an
            # agent making progress is never bored no matter how safe. Under
            # gate:viability "safe" means far from the lethal floor (a merely
            # peckish agent can still be bored); under gate:drive it means at
            # the comfort setpoint (beta_10 — couples boredom to any hunger).
            safety = (
                self._viability(outcome_proprio)
                if self.boredom_gate == "viability"
                else self._drive_level(outcome_proprio)
            )
            calm = (1.0 - safety / self.boredom_drive_threshold).clamp(min=0.0)
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
        cont = self._imagination_continuation(outcome_feat, outcome_proprio)
        fear = self.fear_weight * torch.log(cont.clamp_min(1e-6))
        skill = torch.zeros_like(r_comfort) if skill_reward is None else skill_reward
        component_values = [
            self.w_homeostasis * r_comfort,
            self.w_homeostasis * r_viability,
            self.w_curiosity * r_cur,
            -bored,
            fear,
            self.skill_weight * skill,
        ]
        if self.pain_on:
            component_values.append(self.w_homeostasis * r_pain)
        components = torch.stack(component_values, dim=-1)
        return components, r_cur, bored

    def _imagination_continuation(
        self,
        outcome_feat: torch.Tensor,
        outcome_proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Learned continuation constrained by exact predicted bodily boundaries."""
        cont = torch.sigmoid(self.wm.head_cont(outcome_feat).squeeze(-1).float())
        proprio = (
            self.wm.head_proprio(outcome_feat)
            if outcome_proprio is None
            else outcome_proprio
        )
        if self.death_terminal:
            alive = proprio[..., feeling.INTEGRITY_IDX] > self.via_i_lethal + 1e-3
            cont = cont * alive.to(cont.dtype)
        if self.blackout == "suspended":
            conscious = proprio[..., feeling.ENERGY_IDX] > self.via_e_lethal + 1e-3
            cont = cont * conscious.to(cont.dtype)
        return cont

    def _imagination_reward(
        self,
        img_feat: torch.Tensor,
        img_action: torch.Tensor,
        outcome_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility wrapper returning scalar total affect."""
        components, r_cur, bored = self._imagination_affect(
            img_feat,
            img_feat if outcome_feat is None else outcome_feat,
            img_action,
        )
        return components.sum(-1), r_cur, bored

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

    def _raw_update_credit(self) -> float:
        eligible_steps = max(0, self._act_steps - self.warmup_steps + 1)
        return eligible_steps * self.train_ratio

    def pending_update_credit(self) -> float:
        scheduled = self._raw_update_credit() - self._schedule_credit_origin
        return max(0.0, scheduled - self._dropped_update_credit - self._updates)

    def drop_update_credit(self, amount: float) -> None:
        if amount < 0.0:
            raise ValueError("dropped update credit cannot be negative")
        with self._learn_lock:
            self._dropped_update_credit += min(amount, self.pending_update_credit())

    def learn(self) -> dict[str, float] | None:
        with self._learn_lock:
            return self._learn()

    def _synchronize_learning_stream(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _learn(self) -> dict[str, float] | None:
        learn_began = time.monotonic()
        with torch.profiler.record_function("learn/replay_sample"), self._experience_lock:
            batch_np = self.buffer.sample_sequences(
                self.batch_size,
                self.burn_in + self.seq_len,
                recent=self.recent_slots,
                prioritized=self.prioritize_rows,
                spike_offset=self.burn_in,
                spike_threshold=self.prioritize_threshold,
            )
            experience = len(self.buffer)
        if batch_np is None or experience < self.warmup_steps:
            return None
        with torch.profiler.record_function("learn/replay_transfer"):
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

        # --- world model: posterior dynamics over the replay sequence. Beta's
        # GRU implementation unrolls recurrently; Aion's S5 implementation
        # performs a resettable associative scan. Both return the same latent
        # contract to the organism stack below.
        with self.precision.autocast(), torch.profiler.record_function("learn/world_model_forward"):
            if self.burn_in > 0:
                with torch.no_grad():
                    burn_embed = self.wm.embed(
                        {name: value[:, : self.burn_in] for name, value in obs.items()}
                    )
                graded_embed = self.wm.embed(
                    {name: value[:, self.burn_in :] for name, value in obs.items()}
                )
                embed = torch.cat([burn_embed, graded_embed], dim=1)
            else:
                embed = self.wm.embed(obs)
            sequence = self.wm.dynamics.observe_sequence(
                embed,
                b["action"],
                b["first"],
                b["wake"],
                b["step_scale"],
                self.burn_in,
            )
            if self.burn_in > 0:
                # Slice every downstream target to the graded window.
                b = {k: v[:, self.burn_in :] for k, v in b.items()}
                obs = {k: v[:, self.burn_in :] for k, v in obs.items()}
                kind_idx = kind_idx[:, self.burn_in :]
                embed = embed[:, self.burn_in :]
            L = self.seq_len
            feat = self.wm.dynamics.feat(sequence.h, sequence.z)
            post = sequence.post
            prior = sequence.prior

            pred_depth = self.wm.head_depth(feat)
            pred_rgb = self.wm.head_rgb(feat).view(B, L, self.wm.num_rays, 3)
            pred_kind = self.wm.head_kind(feat).view(B, L, self.wm.num_rays, NUM_RAY_KINDS)
            pred_proprio = self.wm.head_proprio(feat)
            loss_depth = F.mse_loss(pred_depth.float(), b["depth"], reduction="none").sum(-1)
            loss_rgb = F.mse_loss(pred_rgb.float(), b["rgb"], reduction="none").sum((-1, -2))
            loss_kind = (
                F.cross_entropy(
                    pred_kind.float().flatten(0, 2), kind_idx.flatten(), reduction="none"
                )
                .view(B, L, -1)
                .sum(-1)
            )
            loss_proprio = F.mse_loss(pred_proprio.float(), b["proprio"], reduction="none").sum(-1)
            reward_break = self._reward_discontinuity(b["first"], b["wake"])
            homeo_drive = self._homeostasis(b["events"], b["proprio"], reward_break)
            # The viability barrier is priced through the same reward head as
            # the comfort drive; separate forensic channels show which paid.
            if self.via_on:
                via = self._viability_reward(b["proprio"], reward_break)
            else:
                via = torch.zeros_like(homeo_drive)
            pain = self._pain_reward(b["events"], b["proprio"], reward_break)
            homeo = homeo_drive + via + pain
            reward_logits = self.wm.head_reward(feat)
            loss_reward = self.twohot.loss(reward_logits, homeo)
            if self.spike_loss_weight > 0.0:
                spike_w = (homeo.abs() > self.prioritize_threshold).float()
                loss_reward = loss_reward * (1.0 + self.spike_loss_weight * spike_w)
            cont_target, death_target = self._continuation_target(
                b["proprio"], b["step_scale"]
            )
            cont_logits = self.wm.head_cont(feat).squeeze(-1).float()
            loss_cont_raw, loss_cont = self._continuation_loss(cont_logits, cont_target)
            loss_damage = torch.zeros_like(loss_cont)
            if self.pain_on:
                if self.wm.head_damage is None:
                    raise RuntimeError("pain-enabled world model has no damage head")
                damage_logits = self.wm.head_damage(feat).squeeze(-1).float()
                damage_target = b["events"][..., 1]
                loss_damage = F.binary_cross_entropy_with_logits(
                    damage_logits, damage_target, reduction="none"
                )
                damage_weight = torch.where(
                    damage_target > 0.5,
                    torch.full_like(damage_target, self.pain_event_loss_weight),
                    torch.ones_like(damage_target),
                )
                loss_damage = loss_damage * damage_weight
            loss_kl = self.wm.dynamics.kl_loss(post, prior)

            # --- Plan2Explore ensemble: predict the next observation embedding.
            with torch.no_grad():
                if self.curiosity_mask_agents:
                    ens_target = self.wm.embed(self._mask_agents(obs))[:, 1:].float()
                else:
                    ens_target = embed.detach()[:, 1:].float()
            ens_in_feat = feat[:, :-1].detach()
            ens_action = b["action"][:, :-1]
            x = torch.cat([ens_in_feat, ens_action], dim=-1)
            ens_preds = self.wm.ensemble(x)
            loss_ens = (ens_preds.float() - ens_target).pow(2).mean(-1).mean(0)

            model_loss = (
                loss_depth
                + loss_rgb
                + loss_kind
                + loss_proprio
                + loss_reward
                + loss_cont
                + loss_damage
                + loss_kl
            ).float().mean() + loss_ens.float().mean()
            l2_dist = 0.0
            if self.l2_init_weight > 0:
                reg = torch.zeros((), device=self.device)
                for p, p0 in zip(self.wm.parameters(), self._wm_init, strict=True):
                    reg = reg + (p.float() - p0.float()).pow(2).sum()
                model_loss = model_loss + self.l2_init_weight * reg
                l2_dist = float(reg.detach())
        self.opt_model.zero_grad()
        if self.opt_model_muon is not None:
            self.opt_model_muon.zero_grad()
        with torch.profiler.record_function("learn/world_model_backward"):
            model_loss.backward()
            nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip)
        with torch.profiler.record_function("learn/world_model_optimizer"):
            self.opt_model.step()
            if self.opt_model_muon is not None:
                self.opt_model_muon.step()

        # Curiosity statistics on real experience (keeps normalization honest).
        lp_reward_mean = 0.0
        lp_idx: torch.Tensor | None = None
        with torch.no_grad(), self.precision.autocast():
            real_disagreement = self.wm.disagreement(
                feat[:, :-1].detach(), b["action"][:, :-1]
            ).float()
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
            if self.boredom_weight > 0:
                # Publish a learned stimulation estimate. act() combines it
                # with each chronological body's actual safety; replay may
                # refine perception, but cannot advance mood out of order.
                r_dis_real = self.curiosity_norm.normalize(real_disagreement).clamp(0, 5.0)
                if self.curiosity_mode == "lp":
                    stim = self.lp_norm.normalize(real_lp).clamp(0.0, 5.0)
                    stim = stim + self._lp_mix() * r_dis_real.reshape(-1)
                else:
                    stim = r_dis_real.reshape(-1)
                self._online_stimulation = float(stim.mean())

        # --- temporal-skill discriminator on real temporal displacement
        loss_skill_disc = torch.zeros((), device=self.device)
        skill_disc_accuracy = 0.0
        skill_usage_entropy = 0.0
        if isinstance(self.actor, TemporalSkillPolicy) and self.skill_duration < L:
            span = L - self.skill_duration
            labels = b["skill"][:, :span].long()
            valid = labels >= 0
            for offset in range(1, self.skill_duration):
                valid &= b["skill"][:, offset : offset + span].long() == labels
            for offset in range(1, self.skill_duration + 1):
                valid &= b["first"][:, offset : offset + span] < 0.5
            start_skill_feat = feat[:, :span].detach()
            end_skill_feat = feat[:, self.skill_duration :].detach()
            with self.precision.autocast(), torch.profiler.record_function("learn/skill_forward"):
                skill_logits = self.actor.discrimination_logits(start_skill_feat, end_skill_feat)
            if bool(valid.any()):
                loss_skill_disc = F.cross_entropy(skill_logits.float()[valid], labels[valid])
                if self.opt_skill is None:
                    raise RuntimeError("temporal skill discriminator has no optimizer")
                self.opt_skill.zero_grad()
                with torch.profiler.record_function("learn/skill_backward_optimizer"):
                    loss_skill_disc.backward()
                    nn.utils.clip_grad_norm_(self.actor.discriminator.parameters(), self.grad_clip)
                    self.opt_skill.step()
                with torch.no_grad():
                    skill_disc_accuracy = float(
                        (skill_logits[valid].argmax(-1) == labels[valid]).float().mean()
                    )
                    usage = torch.bincount(labels[valid].cpu(), minlength=self.num_skills).float()
                    usage = usage / usage.sum().clamp(min=1.0)
                    entropy = -(usage[usage > 0] * usage[usage > 0].log()).sum()
                    skill_usage_entropy = float(entropy / np.log(self.num_skills))

        # --- actor-critic in imagination, from a subsample of posterior states
        flat = feat.detach().flatten(0, 1)  # (B*L, F) = concat(h, z)
        starts = torch.randperm(flat.shape[0], device=self.device)[: self.imag_starts]
        h_i = flat[starts, : self.wm.dynamics_cfg.deter]
        z_i = flat[starts, self.wm.dynamics_cfg.deter :]

        with self.precision.autocast(), torch.profiler.record_function("learn/imagination_forward"):
            img_feats, img_outcomes = [], []
            img_logps, img_ents, img_manager_ents, img_actions, img_stds, img_skills = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            imagined_skill: torch.Tensor | None = None
            for t in range(self.horizon):
                f_i = self.wm.dynamics.feat(h_i, z_i)
                manager_logp = torch.zeros(f_i.shape[0], device=self.device)
                manager_ent = torch.zeros_like(manager_logp)
                if isinstance(self.actor, TemporalSkillPolicy):
                    if t % self.skill_duration == 0:
                        manager_dist = self.actor.manager_dist(f_i)
                        imagined_skill = manager_dist.sample()
                        manager_logp = manager_dist.log_prob(imagined_skill)
                        manager_ent = manager_dist.entropy()
                    if imagined_skill is None:
                        raise RuntimeError("temporal skill manager did not select an intent")
                dist_cont, dist_grip = self._policy_dists(f_i, skill=imagined_skill)
                a_cont = dist_cont.sample_for_reinforce()
                a_grip = dist_grip.sample()
                logp = dist_cont.log_prob(a_cont) + dist_grip.log_prob(a_grip) + manager_logp
                ent = dist_cont.entropy() + dist_grip.entropy()
                a_vec = torch.cat([a_cont, F.one_hot(a_grip, NUM_GRIP_MODES).float()], dim=-1)
                img_feats.append(f_i)
                img_logps.append(logp)
                img_ents.append(ent)
                img_manager_ents.append(manager_ent)
                img_actions.append(a_vec)
                img_stds.append(dist_cont.std)
                if imagined_skill is not None:
                    img_skills.append(imagined_skill)
                with torch.no_grad():
                    h_i, z_i, _ = self.wm.dynamics.img_step(h_i, z_i, a_vec)
                    img_outcomes.append(self.wm.dynamics.feat(h_i.float(), z_i.float()))
            img_feat = torch.stack(img_feats)
            img_outcome = torch.stack(img_outcomes)
            img_logp = torch.stack(img_logps)
            img_ent = torch.stack(img_ents)
            img_manager_ent = torch.stack(img_manager_ents)
            img_action = torch.stack(img_actions)
            img_std = torch.stack(img_stds)

            with torch.no_grad():
                imagined_skill_reward = torch.zeros_like(img_logp)
                if isinstance(self.actor, TemporalSkillPolicy):
                    skill_tensor = torch.stack(img_skills)
                    for start in range(0, self.horizon, self.skill_duration):
                        end = start + self.skill_duration
                        if end <= self.horizon:
                            imagined_skill_reward[end - 1] = self.actor.intrinsic_reward(
                                img_feat[start], img_outcome[end - 1], skill_tensor[start]
                            )
                components, r_cur, bored = self._imagination_affect(
                    img_feat,
                    img_outcome,
                    img_action,
                    skill_reward=imagined_skill_reward,
                )
                reward = components if self.vector_critic else components.sum(-1)
                cont = self._imagination_continuation(img_outcome)
                discount = self.gamma * cont
                value_ema = self._critic_value(self.critic_ema, img_feat)
                returns = torch.zeros_like(value_ema)
                last = value_ema[-1]
                for t in reversed(range(self.horizon)):
                    bootstrap = (
                        (1 - self.lam) * value_ema[t + 1] + self.lam * last
                        if t + 1 < self.horizon
                        else last
                    )
                    step_discount = discount[t].unsqueeze(-1) if self.vector_critic else discount[t]
                    returns[t] = reward[t].float() + step_discount * bootstrap
                    last = returns[t]

            # Critic forward/loss construction stays under autocast; twohot
            # normalization and reduction are explicitly FP32.
            critic_logits = self._critic_logits(self.critic, img_feat.detach())
            loss_critic = self.twohot.loss(critic_logits, returns.detach()).float().mean()
        self.opt_critic.zero_grad()
        with torch.profiler.record_function("learn/critic_backward_optimizer"):
            loss_critic.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.opt_critic.step()
        with torch.no_grad():
            for p, p_ema in zip(
                self.critic.parameters(), self.critic_ema.parameters(), strict=True
            ):
                p_ema.lerp_(p, 0.02)

        # Actor: REINFORCE on normalized advantages + entropy bonus.
        with torch.no_grad(), self.precision.autocast():
            value_components = self._critic_value(self.critic, img_feat)
            value = value_components.sum(-1) if self.vector_critic else value_components
            total_returns = returns.sum(-1) if self.vector_critic else returns
            scaled_ret = percentile_scale(total_returns, self._return_scale)
            scaled_val = value / max(1.0, self._return_scale[0])
            adv = (scaled_ret - scaled_val).detach()
        loss_actor = (
            -img_logp * adv
            - self.entropy_scale * img_ent
            - self.skill_manager_entropy * img_manager_ent
        ).mean()
        self.opt_actor.zero_grad()
        with torch.profiler.record_function("learn/actor_backward_optimizer"):
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self._actor_parameters, self.grad_clip)
            self.opt_actor.step()

        self._updates += 1
        if self.async_inference and self._updates % self.publish_every == 0:
            self._publish_inference()
        self._synchronize_learning_stream()
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
            # The comfort-drive channel alone (continuous with beta_08–10);
            # the barrier channel is logged separately below so "which drive
            # paid" is readable. homeo_max/spike_frac stay on the SUM — that is
            # what the reward head actually sees.
            "reward_homeostasis": float(homeo_drive.mean()),
            "homeo_max": float(homeo.max()),
            "homeo_spike_frac": float((homeo > 0.1).float().mean()),
            "loss_reward": float(loss_reward.detach().mean()),
            "loss_cont": float(loss_cont_raw.detach().mean()),
            "terminal_frac": float(death_target.float().mean()),
            "elapsed_discount_frac": float(
                ((cont_target < 1.0) & ~death_target).float().mean()
            ),
            "value": float(value.mean()),
            "loss_critic": float(loss_critic.detach()),
            "loss_actor": float(loss_actor.detach()),
            "entropy": float(img_ent.detach().mean()),
            "policy_cont_std_mean": float(img_std.detach().mean()),
            "policy_cont_std_max": float(img_std.detach().max()),
            "policy_action_abs_mean": float(img_action[..., :CONT_DIM].detach().abs().mean()),
            "policy_action_saturation_frac": float(
                (img_action[..., :CONT_DIM].detach().abs() > 0.95).float().mean()
            ),
            "policy_rest_sample_frac": float(
                (img_action[..., :2].detach().abs().amax(-1) < 0.1).float().mean()
            ),
            "updates": float(self._updates),
            "act_steps": float(self._act_steps),
            "train_ratio_eff": float(self._updates / max(1, self._act_steps)),
            "learn_seconds": self._learn_seconds,
            "graded_timepoints_per_second": float(B * L / max(elapsed, 1e-9)),
            "context_steps": float(L),
            "buffer": float(len(self.buffer)),
            "precision_ieee_fp32": float(self.precision.mode is PrecisionMode.IEEE_FP32),
            "precision_tf32": float(self.precision.mode is PrecisionMode.TF32),
            "precision_amp_bf16": float(self.precision.mode is PrecisionMode.AMP_BF16),
        }
        with torch.no_grad():
            cont_probability = torch.sigmoid(cont_logits)
            if bool(death_target.any()):
                self._metrics["cont_terminal"] = float(
                    cont_probability[death_target].mean()
                )
            ordinary_alive = (~death_target) & (b["step_scale"] <= 1.0)
            if bool(ordinary_alive.any()):
                self._metrics["cont_alive"] = float(cont_probability[ordinary_alive].mean())
            elapsed_transition = (~death_target) & (b["step_scale"] > 1.0)
            if bool(elapsed_transition.any()):
                self._metrics["cont_elapsed"] = float(
                    cont_probability[elapsed_transition].mean()
                )
        if self.vector_critic:
            for index, name in enumerate(self.affect_names):
                self._metrics[f"value_{name}"] = float(value_components[..., index].mean())
                self._metrics[f"return_{name}"] = float(returns[..., index].mean())
                self._metrics[f"affect_{name}"] = float(components[..., index].mean())
        if self.pain_on:
            self._metrics["reward_pain"] = float(pain.mean())
            self._metrics["loss_damage"] = float(loss_damage.detach().mean())
            damage_probability = torch.sigmoid(damage_logits.detach())
            damaged = b["events"][..., 1] > 0.5
            if bool(damaged.any()):
                self._metrics["damage_probability_positive"] = float(
                    damage_probability[damaged].mean()
                )
            if bool((~damaged).any()):
                self._metrics["damage_probability_negative"] = float(
                    damage_probability[~damaged].mean()
                )
        if isinstance(self.actor, TemporalSkillPolicy):
            self._metrics.update(
                {
                    "skill_discriminator_loss": float(loss_skill_disc.detach()),
                    "skill_discriminator_accuracy": skill_disc_accuracy,
                    "skill_usage_entropy": skill_usage_entropy,
                    "skill_imagined_reward": float(imagined_skill_reward.mean()),
                    "skill_manager_entropy": float(img_manager_ent.detach().mean()),
                    "skill_switches": float(self._skill_switches),
                    "active_skill": float(self._active_skill),
                }
            )
        # Can the reward head see the loud moments? |decoded - realized| on
        # spike samples is the reachability gauge (009: a head that never
        # trains on meals can't let the actor plan toward one), and the
        # spike-row fraction shows what prioritized replay is feeding it.
        with torch.no_grad():
            spike = homeo.abs() > 0.1
            if bool(spike.any()):
                pred_r = self.twohot.decode(reward_logits.detach())
                self._metrics["reward_head_spike_err"] = float(
                    (pred_r[spike] - homeo[spike]).abs().mean()
                )
            if self.prioritize_rows > 0:
                self._metrics["spike_row_frac"] = float(spike.any(dim=1).float().mean())
        if self.l2_init_weight > 0:
            self._metrics["l2_init_dist"] = l2_dist
        if self.homeostasis_mode == "drive":
            self._metrics["drive_level"] = float(self._drive_level(b["proprio"]).mean())
        if self.via_on:
            self._metrics["reward_viability"] = float(via.mean())
            self._metrics["viability_level"] = float(self._viability(b["proprio"]).mean())
            self._metrics["viability_max"] = float(self._viability(b["proprio"]).max())
            if self.wellbeing_on:
                wellbeing = feeling.wellbeing(
                    self._viability(b["proprio"]),
                    self._drive_level(b["proprio"]),
                    weight=self.wellbeing_weight,
                    barrier_cap=self.via_barrier_cap,
                    comfort_decay=self.wellbeing_comfort_decay,
                )
                self._metrics["wellbeing"] = float(wellbeing.mean())
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
            self._metrics["stimulation_online"] = self._online_stimulation
            self._metrics["boredom_calm_gate"] = self._gate_calm
            self._metrics["boredom_dull_gate"] = self._gate_dull
            if self.boredom_pressure_on:
                self._metrics["boredom_pressure"] = self._boredom_pressure
        return self._metrics

    def introspect(self) -> dict[str, float]:
        out = dict(self._metrics)
        out["pending_update_credit"] = self.pending_update_credit()
        out["dropped_update_credit"] = self._dropped_update_credit
        if self.temperament_enabled:
            out.update({f"temperament_{k}": v for k, v in self.temperament.items()})
        # Exact per-life realized return (updated per lived tick in act(), so
        # it is fresh even between learns) — the direct integral the reframe
        # could only infer from the batch-mean sign.
        if self.homeostasis_mode == "drive":
            out["life_return_homeo"] = self._life_return_homeo
            if self.via_on:
                out["life_return_via"] = self._life_return_via
                if self.wellbeing_on:
                    out["life_return_wellbeing"] = self._life_return_via
            if self.pain_on:
                out["life_return_pain"] = self._life_return_pain
        if self.async_inference:
            out["inference_lag_updates"] = float(self._updates - self._published_updates)
        return out

    def reset_stream(self) -> None:
        """The stream broke for good (respawn into a new body): reset live
        recurrent state and sever the salience chain — a newborn must not
        inherit a drive-delta spike across a gap nobody lived."""
        self.h, self.z = self.wm.dynamics.initial(1, self.device)
        self.last_action = torch.zeros(1, ACTION_DIM, device=self.device)
        self._prev_drive = None
        self._prev_via = None
        self._prev_integrity = None
        self._life_return_homeo = 0.0
        self._life_return_via = 0.0
        self._life_return_pain = 0.0
        self._stream_first = True
        self._stream_wake = False
        self._step_scale = 1.0
        self._active_skill = -1
        self._skill_remaining = 0

    def wake(self, dormant_steps: int = 0) -> None:
        """First act after a dormant spell (see the blackout flag's contract).

        cut: the gap is a stream break like any other. priced: the live
        recurrent state still resets (the mind was off) but _prev_drive
        survives and the wake step is NOT marked as a stream break, so the
        gap's real drive delta enters both the replayed reward and the
        salience chain. suspended preserves elapsed-time context but severs
        felt deltas: an unconscious interval earns no reward or pain.
        """
        if dormant_steps < 0:
            raise ValueError("dormant_steps cannot be negative")
        if self.blackout == "cut":
            self.reset_stream()
            return
        self.h, self.z = self.wm.dynamics.initial(1, self.device)
        self.last_action = torch.zeros(1, ACTION_DIM, device=self.device)
        if self.blackout == "suspended":
            self._prev_drive = None
            self._prev_via = None
            self._prev_integrity = None
            self._stream_wake = True
            self._step_scale = float(dormant_steps + 1)
        self._active_skill = -1
        self._skill_remaining = 0

    def _death_transition_context(self, dormant: bool, dormant_steps: int) -> tuple[bool, float]:
        del dormant, dormant_steps
        return False, 1.0

    def record_death(self, obs: Observation, dormant: bool = False, dormant_steps: int = 0) -> None:
        """The body's true end, delivered by the runtime (see Brain.record_death).

        Recorded only when death_terminal is on: this is the sample that gives
        the continuation head a real terminal target (cont = integrity >
        lethal). Without it no at-the-floor observation can ever exist —
        dormant bodies are unobserved and the death tick removes the robot
        before sensing — so the head would train "continue" everywhere and
        death_terminal would be inert (round-012 review). Off keeps the buffer
        contents identical to beta_10.

        The delivered obs is the last one the body produced (stale by the
        length of a dormant spell), with the vitals set to the state the world
        actually reached: integrity 0 always; energy 0 too when the body died
        hibernating (the energy collapse is what started the integrity clock).
        Events are cleared and the action recorded is zero — nothing happened
        at death but death. Under priced blackout the step reads the real
        drive/barrier plunge from its predecessor (the same one-visible-
        transition contract as wake()); under cut a dormant death is a stream
        break, so no fictional delta is read across the gap.
        """
        if not self.death_terminal:
            return
        proprio = np.array(obs["proprio"], dtype=np.float32, copy=True)
        proprio[6] = 0.0  # integrity: the death condition itself
        if dormant:
            proprio[5] = 0.0  # hibernation death: energy had already collapsed
        dead_obs = Observation(
            rays=obs["rays"],
            proprio=proprio,
            sound=obs["sound"],
            events=np.zeros_like(obs["events"]),
        )
        first = self._stream_first or (dormant and self.blackout == "cut")
        wake, step_scale = self._death_transition_context(dormant, dormant_steps)
        salience = 0.0
        with torch.no_grad():
            if self.homeostasis_mode == "drive":
                tensors = self._obs_to_tensors(dead_obs)
                d = float(self._drive_level(tensors["proprio"])[0])
                reward_break = first or (dormant and self.blackout == "suspended")
                prev_d = None if reward_break else self._prev_drive
                if prev_d is not None:
                    salience = abs(self.drive_scale * (prev_d - d))
                    self._life_return_homeo += self.drive_scale * (prev_d - d)
                self._life_return_homeo -= self.drive_level_penalty * d
                if self.via_on:
                    v = float(self._viability(tensors["proprio"])[0])
                    prev_v = None if reward_break else self._prev_via
                    if prev_v is not None:
                        salience += abs(self.via_scale * (prev_v - v)) + self.via_floor * v
                    self._life_return_via -= self.via_floor * v
                    if self.wellbeing_on:
                        self._life_return_via += float(
                            feeling.wellbeing(
                                tensors["proprio"].new_tensor(v),
                                tensors["proprio"].new_tensor(d),
                                weight=self.wellbeing_weight,
                                barrier_cap=self.via_barrier_cap,
                                comfort_decay=self.wellbeing_comfort_decay,
                            )
                        )
        with self._experience_lock:
            self.buffer.add(
                dead_obs,
                np.zeros(ACTION_DIM, dtype=np.float32),
                salience=salience,
                first=first,
                wake=wake,
                step_scale=step_scale,
            )
        self._stream_first = False

    def _recompute_salience(self) -> None:
        """Backfill per-step reward salience for a pre-salience checkpoint.

        Right after ReplayBuffer.load_state_dict the ring is in time order,
        so the drive deltas below pair consecutive lived steps. Stream-break
        markers, when the blob carries them, zero the fictional cross-gap
        spikes; pre-marker blobs' breaks are unrecoverable — those steps read
        wrong (a respawn backfills as a ~3.9 spike), which a screen must
        discount by hand.
        """
        n = len(self.buffer)
        if n == 0:
            return
        if self.homeostasis_mode == "drive":
            proprio = torch.as_tensor(
                self.buffer.proprio[:n].astype(np.float32), device=self.device
            )
            d = self._drive_level(proprio)
            reduction = torch.zeros_like(d)
            reduction[1:] = d[:-1] - d[1:]
            spike = (self.drive_scale * reduction).abs()
            if self.via_on:
                # Near-death moments are salient too, so a screen with the
                # barrier on oversamples them from the recorded life. Same
                # form as act(): |scale·ΔV| + floor·V — the standing tax is
                # what carries it in the staged floor-only configuration.
                V = self._viability(proprio)
                v_red = torch.zeros_like(V)
                v_red[1:] = V[:-1] - V[1:]
                spike = spike + (self.via_scale * v_red).abs() + self.via_floor * V
            discontinuity = torch.as_tensor(
                self.buffer.first[:n].astype(np.float32), device=self.device
            )
            if self.blackout == "suspended":
                discontinuity = torch.maximum(
                    discontinuity,
                    torch.as_tensor(
                        self.buffer.wake[:n].astype(np.float32), device=self.device
                    ),
                )
            if self.pain_on:
                events = torch.as_tensor(
                    self.buffer.events[:n].astype(np.float32), device=self.device
                )
                spike = spike + self.pain_weight * feeling.acute_integrity_loss(
                    proprio, events[..., 1], discontinuity
                )
            sal = spike.cpu().numpy()
            sal[discontinuity.cpu().numpy() > 0.5] = 0.0
        else:
            sal = self.buffer.events[:n, :2].astype(np.float32).sum(-1)
        self.buffer.salience[:n] = sal.astype(np.float16)

    # ----------------------------------------------------------- persistence

    def state_dict(self) -> dict[str, Any]:
        with self._learn_lock, self._experience_lock:
            return self._state_dict()

    def _state_dict(self) -> dict[str, Any]:
        state_muon = self.opt_model_muon.state_dict() if self.opt_model_muon else None
        return {
            "obs_version": OBS_VERSION,
            "brain_family": self.brain_family,
            "precision": self.precision.mode.value,
            "learning_contract": self._learning_contract(),
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_ema": self.critic_ema.state_dict(),
            "opt_model": self.opt_model.state_dict(),
            "opt_model_muon": state_muon,
            "opt_actor": self.opt_actor.state_dict(),
            "opt_skill": self.opt_skill.state_dict() if self.opt_skill is not None else None,
            "opt_critic": self.opt_critic.state_dict(),
            "curiosity_norm": self.curiosity_norm.state_dict(),
            "temperament": dict(self.temperament),
            "lp_regions": self.regions.state_dict(),
            "lp_tracker": self.lp.state_dict(),
            "lp_norm": self.lp_norm.state_dict(),
            "return_scale": self._return_scale[0],
            "boredom_pressure": self._boredom_pressure,
            "online_stimulation": self._online_stimulation,
            "active_skill": self._active_skill,
            "skill_remaining": self._skill_remaining,
            "skill_switches": self._skill_switches,
            # Ride the checkpoint so a resume during a dormant spell doesn't
            # silently cut a blackout the priced mode should have seen.
            "prev_drive": self._prev_drive,
            "prev_via": self._prev_via,
            "prev_integrity": self._prev_integrity,
            "life_return_homeo": self._life_return_homeo,
            "life_return_via": self._life_return_via,
            "life_return_pain": self._life_return_pain,
            "stream_first": self._stream_first,
            "stream_wake": self._stream_wake,
            "step_scale": self._step_scale,
            "updates": self._updates,
            "published_updates": self._published_updates,
            "inference": self._inference.state_dict() if self._inference is not None else None,
            "act_steps": self._act_steps,
            "schedule_credit_origin": self._schedule_credit_origin,
            "dropped_update_credit": self._dropped_update_credit,
            "rng_state": self.rng.bit_generator.state,
            "buffer": self.buffer.state_dict(),
            "h": self.h.cpu().numpy(),
            "z": self.z.cpu().numpy(),
            "last_action": self.last_action.cpu().numpy(),
        }

    def _learning_contract(self) -> dict[str, int | float | str]:
        return {
            "train_ratio": self.train_ratio,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "burn_in": self.burn_in,
            "warmup_steps": self.warmup_steps,
            "recent": self.recent_slots,
            "prioritize": self.prioritize,
            "publish_every": self.publish_every,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stored_family = state.get("brain_family", "dreamer")
        if stored_family != self.brain_family:
            raise ValueError(
                f"brain checkpoint belongs to {stored_family!r}, not {self.brain_family!r}"
            )
        if state.get("obs_version") != OBS_VERSION:
            raise ValueError(
                f"brain checkpoint has obs_version {state.get('obs_version')}, "
                f"world speaks {OBS_VERSION}: refusing to load across contract changes"
            )
        stored_precision = state.get("precision")
        if stored_precision is not None:
            try:
                checkpoint_precision = PrecisionMode(str(stored_precision))
            except ValueError as exc:
                raise ValueError(f"unknown checkpoint precision {stored_precision!r}") from exc
            if checkpoint_precision is not self.precision.mode:
                raise ValueError(
                    f"brain checkpoint precision is {checkpoint_precision.value!r}, "
                    f"but config requests {self.precision.mode.value!r}"
                )
        stored_contract = state.get("learning_contract")
        if stored_contract is not None and stored_contract != self._learning_contract():
            raise ValueError(
                "brain checkpoint learning contract differs from the configured replay, "
                "train-ratio, or publication schedule"
            )
        migrated = self._migrate_world_model_state(state["wm"])
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
        if self.opt_skill is not None and state.get("opt_skill") is not None:
            self.opt_skill.load_state_dict(state["opt_skill"])
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
        self._online_stimulation = float(
            state.get("online_stimulation", self.boredom_stim_threshold)
        )
        self._active_skill = int(state.get("active_skill", -1))
        self._skill_remaining = int(state.get("skill_remaining", 0))
        self._skill_switches = int(state.get("skill_switches", 0))
        prev_drive = state.get("prev_drive")
        self._prev_drive = float(prev_drive) if prev_drive is not None else None
        # Guarded: pre-viability checkpoints carry no barrier/return state.
        prev_via = state.get("prev_via")
        self._prev_via = float(prev_via) if prev_via is not None else None
        prev_integrity = state.get("prev_integrity")
        self._prev_integrity = float(prev_integrity) if prev_integrity is not None else None
        self._life_return_homeo = float(state.get("life_return_homeo", 0.0))
        self._life_return_via = float(state.get("life_return_via", 0.0))
        self._life_return_pain = float(state.get("life_return_pain", 0.0))
        self._stream_first = bool(state.get("stream_first", False))
        self._stream_wake = bool(state.get("stream_wake", False))
        self._step_scale = float(state.get("step_scale", 1.0))
        self._updates = int(state["updates"])
        self._published_updates = int(state.get("published_updates", self._updates))
        # Guarded: pre-pacing checkpoints carry no act-step counter; seed it
        # at the stored buffer's size so the update/act-step pair stays
        # roughly coherent.
        self._act_steps = int(state.get("act_steps", len(state["buffer"]["depth"])))
        self._dropped_update_credit = float(state.get("dropped_update_credit", 0.0))
        if "schedule_credit_origin" in state:
            self._schedule_credit_origin = float(state["schedule_credit_origin"])
        else:
            # Older schedulers discarded their process-local debt on resume.
            # Baseline that historical checkpoint at zero debt, then preserve
            # every newly accrued update credit from this migration onward.
            self._schedule_credit_origin = (
                self._raw_update_credit() - self._dropped_update_credit - self._updates
            )
        self.rng.bit_generator.state = state["rng_state"]
        self.buffer.load_state_dict(state["buffer"])
        if "salience" not in state["buffer"]:
            self._recompute_salience()
        self.h = torch.as_tensor(state["h"], device=self.device)
        self.z = torch.as_tensor(state["z"], device=self.device)
        self.last_action = torch.as_tensor(state["last_action"], device=self.device)
        if self.async_inference:
            self._publish_inference()
            inference_state = state.get("inference")
            if inference_state is not None and self._inference is not None:
                self._inference.load_state_dict(inference_state)
                self._published_updates = int(state.get("published_updates", self._updates))

    def inherit(self, state: dict[str, Any]) -> None:
        """Warm-start a newborn from a living donor: weights, memories, and
        temperament — the temperament mutated, so lineages drift through
        temperament space and transmission has something to select on."""
        self.load_state_dict(state)
        # The donor already earned its outstanding updates. A copied newborn
        # starts a fresh schedule while retaining the inherited replay and
        # optimizer state; lineage respawn reuses the original brain instead.
        self._dropped_update_credit = 0.0
        self._schedule_credit_origin = self._raw_update_credit() - self._updates
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
