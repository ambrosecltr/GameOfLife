"""PlasticBrain — the anima family (proposal 002).

Feeling gates learning instead of being maximized. Each act step:

  1. compute the neuromodulator `M` from the body's interoception — the shared
     `gol_brains.feeling` LEVELS read directly (anima_06): comfort as satisfaction
     relative to a neutral hunger level (`d_ref`), plus a standing viability tax.
     A change-based M taught ~nothing (anima_05: comfort M negative on 99.9% of
     steps, telescoped net-negative over a mortal life, and the rectified
     viability gate was inert — m_viability ≡ 0 across 1.4M ticks) because a
     plastic brain has no critic to integrate a stream of reductions back into a
     value. So it reads the level: being fed is continuously positive, hunger and
     danger continuously negative. Offline-screened (anima_valence_screen.py):
     level return correlates +0.70 with mean energy and +0.4 with eating, where
     the reduction form correlated −0.80 / −0.41 — it rewarded the opposite;
  2. consolidate the eligibility trace with `M` (crediting the previous step's
     activity, which produced this step's outcome);
  3. forward pass (encoder → GRU → readout), sample an action with heritable
     restlessness — scaled up by the current drive when appetite is on, so a
     hungry body searches harder — keeping the discrete EAT mode tried;
  4. fold this step's pre⊗post into the trace (the readout's discrete credit
     lands on the *taken* gripper mode).

No gradients, no replay buffer, no learner thread: learning is this local rule,
in `act`. `experience_count`/`target_train_ratio` stay 0 so the learner thread
never schedules this brain (invariant 5, trivially).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gol_world.interface import (
    EVENTS_DIM,
    GAZE_DIM,
    NUM_GRIP_MODES,
    PROPRIO_DIM,
    SIGNAL_DIM,
    SOUND_DIM,
    Action,
    BodySpec,
    Observation,
)

from gol_brains import feeling
from gol_brains.base import Brain
from gol_brains.plastic.network import PlasticNet

CONT_DIM = 2 + SIGNAL_DIM + GAZE_DIM  # drive(2) + signal(2) + gaze(2)
OUT_DIM = CONT_DIM + NUM_GRIP_MODES

# Heritable gene multipliers (log-normal around 1.0, drift on inherit). Applied
# to the founder means so a lineage can evolve how it feels and how it learns.
GENE_KEYS = (
    "alpha",
    "tau",
    "decay",
    "restlessness",
    "appetite_gain",
    "comfort_gain",
    "viability_gain",
    "standing_gain",
    "mod_gain",
    "energy_weight",
    "integrity_weight",
    "rest_weight",
    "via_energy_weight",
    "via_integrity_weight",
)


class PlasticBrain(Brain):
    def __init__(
        self, cfg: dict[str, Any], seed: int, device: str = "cpu", body: BodySpec | None = None
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.body = body or BodySpec()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        core = dict(cfg.get("core", {}))
        self.hidden = int(core.get("hidden", 256))
        pl = dict(cfg.get("plasticity", {}))
        self._alpha0 = float(pl.get("alpha", 0.1))
        self._tau0 = float(pl.get("tau", 20.0))
        self._decay0 = float(pl.get("decay", 1.0e-3))
        self.w_clip = float(pl.get("w_clip", 2.0))
        self.plastic_on = bool(pl.get("enabled", True))  # False = frozen-net control (P3)

        self._restless0 = float(cfg.get("restlessness", 0.2))
        # Appetite (anima_03): drive scales restlessness, so hunger raises search
        # effort innately — the coupling STRENGTH is heritable, so selection can
        # tune it (to zero if it's wrong). 0 disables (anima_01/02 behavior).
        self._appetite0 = float(cfg.get("appetite_gain", 0.0))

        # --- valence (the neuromodulator M) — forked from beta_11's drive block ---
        val = dict(cfg.get("valence", {}))
        drv = dict(val.get("drive", {}))
        self._comfort_gain0 = float(val.get("comfort_gain", 3.0))
        # anima_06: the neutral hunger level where comfort valence crosses from
        # reward (fed, d < d_ref) to punishment (hungry, d > d_ref). ~0.40 puts
        # neutral just above the brownout floor (screened; see class docstring).
        self.d_ref = float(drv.get("d_ref", 0.40))
        self.drive_pow_m = float(drv.get("pow_m", 3.0))
        self.drive_pow_n = float(drv.get("pow_n", 2.0))
        self._drive_setpoints = torch.tensor(
            [
                float(drv.get("energy_setpoint", 0.85)),
                float(drv.get("integrity_setpoint", 1.0)),
                float(drv.get("rested_setpoint", 1.0)),
            ],
            device=self.device,
        )
        self._drive_w0 = torch.tensor(
            [
                float(drv.get("energy_weight", 1.0)),
                float(drv.get("integrity_weight", 1.0)),
                float(drv.get("rest_weight", 0.5)),
            ],
            device=self.device,
        )
        via = dict(val.get("viability", {}))
        # reduction gate ON for anima (mirror of beta_11); standing tax ~0 (ablation).
        # rectified (anima_03): consolidate only ESCAPES (positive reductions).
        # anima_02 measured the unrectified gate net-negative over awake life
        # (life_return_via −11..−33) because decline is felt but recovery happens
        # inside dormancy behind a stream reset — the negative half anti-
        # consolidates whatever a starving agent was doing, including foraging.
        self.via_rectified = bool(via.get("rectified", False))
        self._viability_gain0 = float(via.get("viability_gain", 3.0))
        self._standing_gain0 = float(via.get("standing_gain", 0.0))
        self.via_cap = float(via.get("barrier_cap", 4.0))
        self.via_e_lethal = float(via.get("energy_lethal", 0.0))
        self.via_e_safe = float(via.get("energy_safe", 0.25))
        self.via_i_lethal = float(via.get("integrity_lethal", 0.0))
        self.via_i_safe = float(via.get("integrity_safe", 0.5))
        self._via_we0 = float(via.get("energy_weight", 1.0))
        self._via_wi0 = float(via.get("integrity_weight", 1.0))
        mod = dict(val.get("modulator", {}))
        self._mod_gain0 = float(mod.get("gain", 1.0))
        self.m_clip = float(mod.get("clip", 5.0))

        gen = dict(cfg.get("genome", {}))
        self.genome_enabled = bool(gen.get("enabled", True))
        self.gene_sigma = float(gen.get("sigma", 0.25))
        self.mutation_sigma = float(gen.get("mutation_sigma", 0.1))
        self.w_mutation_sigma = float(gen.get("weight_mutation_sigma", 0.05))
        # genome = Darwinian (reinit W_fast); lineage = Lamarckian (carry W_fast).
        self.inherit_mode = str(cfg.get("inherit_mode", "genome"))
        if self.inherit_mode not in ("genome", "lineage"):
            raise ValueError(f"unknown inherit_mode: {self.inherit_mode!r}")

        # Founder genome: log-normal multipliers around 1.0 (diversity sigma).
        if self.genome_enabled:
            self.genes = {
                k: float(np.exp(self.rng.normal(0.0, self.gene_sigma))) for k in GENE_KEYS
            }
        else:
            self.genes = dict.fromkeys(GENE_KEYS, 1.0)

        self.in_dim = self.body.num_rays * (1 + 3 + 4) + PROPRIO_DIM + SOUND_DIM + EVENTS_DIM
        self._build_net()

        self.h = torch.zeros(1, self.hidden, device=self.device)
        self._prev_d: float | None = None
        self._prev_v: float | None = None
        self._act_steps = 0
        self._life_return_comfort = 0.0
        self._life_return_via = 0.0
        self._m_last = 0.0
        self._m_comfort_last = 0.0
        self._m_via_last = 0.0
        self._arousal_last = 0.0

    # ------------------------------------------------------------- gene → params
    def _build_net(self) -> None:
        g = self.genes
        plastic_kw = {
            "alpha": self._alpha0 * g["alpha"] if self.plastic_on else 0.0,
            "tau": max(1.0, self._tau0 * g["tau"]),
            "decay": self._decay0 * g["decay"],
            "w_clip": self.w_clip,
            "plastic": self.plastic_on,
        }
        self.net = PlasticNet(
            self.in_dim,
            self.hidden,
            OUT_DIM,
            plastic_kw=plastic_kw,
            rng=self.rng,
        ).to(self.device)
        # gene-scaled valence params (rebuilt whenever the genome changes).
        self.comfort_gain = self._comfort_gain0 * g["comfort_gain"]
        self.viability_gain = self._viability_gain0 * g["viability_gain"]
        self.standing_gain = self._standing_gain0 * g["standing_gain"]
        self.mod_gain = self._mod_gain0 * g["mod_gain"]
        self.restlessness = float(np.clip(self._restless0 * g["restlessness"], 0.0, 0.6))
        self.appetite_gain = self._appetite0 * g["appetite_gain"]
        self.drive_weights = self._drive_w0 * torch.tensor(
            [g["energy_weight"], g["integrity_weight"], g["rest_weight"]], device=self.device
        )
        self.via_we = self._via_we0 * g["via_energy_weight"]
        self.via_wi = self._via_wi0 * g["via_integrity_weight"]

    # --------------------------------------------------------------------- feel
    def _drive(self, proprio: torch.Tensor) -> float:
        d = feeling.drive_level(
            proprio,
            self._drive_setpoints,
            self.drive_weights,
            self.drive_pow_m,
            self.drive_pow_n,
        )
        return float(d[0])

    def _viability(self, proprio: torch.Tensor) -> float:
        return float(
            feeling.viability(
                proprio,
                barrier_cap=self.via_cap,
                energy_lethal=self.via_e_lethal,
                energy_safe=self.via_e_safe,
                integrity_lethal=self.via_i_lethal,
                integrity_safe=self.via_i_safe,
                energy_weight=self.via_we,
                integrity_weight=self.via_wi,
            )[0]
        )

    def _encode(self, obs: Observation) -> torch.Tensor:
        rays = np.asarray(obs["rays"], dtype=np.float32).reshape(-1)
        vec = np.concatenate(
            [rays, obs["proprio"], obs["sound"], obs["events"]], dtype=np.float32
        )
        return torch.from_numpy(vec).to(self.device).unsqueeze(0)

    # ---------------------------------------------------------------------- act
    def act(self, obs: Observation) -> Action:
        with torch.no_grad():
            proprio = torch.as_tensor(
                np.asarray(obs["proprio"], dtype=np.float32), device=self.device
            ).unsqueeze(0)
            d = self._drive(proprio)
            v = self._viability(proprio)
            # Level valence (anima_06): the modulator reads the felt LEVEL, not
            # its change. Comfort = satisfaction relative to the neutral hunger
            # level d_ref (fed → positive, hungry → negative); viability enters
            # as a standing danger tax (−standing_gain·v), which fires whenever
            # energy/integrity are near the lethal floor — where the old
            # rectified escape-gate was inert. Both are defined from the first
            # step; the one-step-delayed consolidate() harmlessly no-ops on a
            # fresh stream because the trace is zero after reset.
            m_comfort = self.comfort_gain * (self.d_ref - d)
            m_via = -self.standing_gain * v
            m = float(np.clip(self.mod_gain * (m_comfort + m_via), -self.m_clip, self.m_clip))

            # Consolidate the trace (activity that led to this outcome) BEFORE
            # this step's forward — one-step-delayed neuromodulated credit.
            self.net.consolidate(m)

            x = self._encode(obs)
            out, self.h, e, cand = self.net(x, self.h)
            out = out.flatten()

            # Arousal: hunger scales search effort (motor noise + exploration
            # floor). With appetite_gain 0 this is exactly restlessness.
            arousal = float(
                np.clip(self.restlessness * (1.0 + self.appetite_gain * d), 0.0, 0.6)
            )
            noise = torch.from_numpy(
                (self.rng.standard_normal(CONT_DIM) * arousal).astype(np.float32)
            ).to(self.device)
            cont = torch.tanh(out[:CONT_DIM] + noise)
            logits = out[CONT_DIM:]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            eps = min(arousal, 0.5)  # discrete exploration floor keeps EAT reachable
            probs = (1.0 - eps) * probs + eps / NUM_GRIP_MODES
            probs = probs / probs.sum()
            grip = int(self.rng.choice(NUM_GRIP_MODES, p=probs))

            grip_onehot = torch.zeros(NUM_GRIP_MODES, device=self.device)
            grip_onehot[grip] = 1.0
            readout_post = torch.cat([cont, grip_onehot])
            self.net.accumulate(e, cand, readout_post)

            if self._prev_d is not None:
                self._life_return_comfort += m_comfort
                self._life_return_via += m_via
            self._prev_d, self._prev_v = d, v
            self._m_last, self._m_comfort_last, self._m_via_last = m, m_comfort, m_via
            self._arousal_last = arousal
            self._act_steps += 1

            cont_np = cont.cpu().numpy()
        return Action(
            drive=cont_np[:2].copy(),
            gripper=grip,
            signal=cont_np[2:4].copy(),
            gaze=cont_np[4:6].copy(),
        )

    # ------------------------------------------------------------ stream breaks
    def reset_stream(self) -> None:
        self.h = torch.zeros(1, self.hidden, device=self.device)
        self.net.reset_trace()
        self._prev_d = None
        self._prev_v = None
        self._life_return_comfort = 0.0
        self._life_return_via = 0.0

    def wake(self) -> None:
        # No world model / critic here: a dormancy gap is just a stream break.
        self.reset_stream()

    # ------------------------------------------------------------- observability
    def introspect(self) -> dict[str, float]:
        out = {
            "m": self._m_last,
            "m_comfort": self._m_comfort_last,
            "m_viability": self._m_via_last,
            "w_fast_norm": self.net.fast_norm(),
            "life_return_comfort": self._life_return_comfort,
            "life_return_via": self._life_return_via,
            "restlessness": self.restlessness,
            "arousal": self._arousal_last,
        }
        if self.genome_enabled:
            out.update({f"gene_{k}": v for k, v in self.genes.items()})
        return out

    # -------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict[str, Any]:
        return {
            "genes": dict(self.genes),
            "net": self.net.state_dict(),
            "h": self.h.detach().cpu(),
            "prev_d": self._prev_d,
            "prev_v": self._prev_v,
            "act_steps": self._act_steps,
            "life_return_comfort": self._life_return_comfort,
            "life_return_via": self._life_return_via,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.genome_enabled and "genes" in state:
            # Fill genes added after a checkpoint was written with the neutral
            # multiplier, so older anima saves keep loading across gene-set growth.
            self.genes = {k: float(state["genes"].get(k, 1.0)) for k in GENE_KEYS}
            self._build_net()
        self.net.load_state_dict(state["net"])
        self.h = state["h"].to(self.device)
        self._prev_d = state["prev_d"]
        self._prev_v = state["prev_v"]
        self._act_steps = int(state.get("act_steps", 0))
        self._life_return_comfort = float(state.get("life_return_comfort", 0.0))
        self._life_return_via = float(state.get("life_return_via", 0.0))
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]

    def inherit(self, state: dict[str, Any]) -> None:
        """Warm-start a newborn from a donor. The genome and the innate wiring
        (`W_slow`) mutate so lineages drift and differential survival can select;
        `W_fast` is reinitialised (Darwinian) or carried (Lamarckian) per
        `inherit_mode`. Then the stream resets — a newborn must not consolidate
        on a trace it never lived."""
        self.load_state_dict(state)
        if self.genome_enabled and self.mutation_sigma > 0:
            self.genes = {
                k: v * float(np.exp(self.rng.normal(0.0, self.mutation_sigma)))
                for k, v in self.genes.items()
            }
            self._build_net()
            # _build_net rebuilt a fresh net; reload the donor's inherited weights.
            self.net.load_state_dict(state["net"])
        if self.w_mutation_sigma > 0:
            with torch.no_grad():
                for p in self.net.parameters():  # W_slow + innate gates (no grad, but present)
                    jitter = torch.from_numpy(
                        np.exp(
                            self.rng.normal(0.0, self.w_mutation_sigma, tuple(p.shape))
                        ).astype(np.float32)
                    ).to(p.device)
                    p.mul_(jitter)
        if self.inherit_mode == "genome":
            self.net.reset_fast()
        self.reset_stream()
