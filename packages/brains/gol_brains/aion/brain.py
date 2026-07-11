"""Aion: an S5 world-model organism with multi-timescale continuity."""

from __future__ import annotations

from typing import Any

import torch
from gol_world.interface import BodySpec

from gol_brains.aion.s5 import (
    S5_CHECKPOINT_FORMAT,
    S5Dynamics,
    S5DynamicsConfig,
    migrate_complex_s5_state,
)
from gol_brains.dreamer.brain import ACTION_DIM, DreamerBrain, WorldModel


class AionBrain(DreamerBrain):
    """The Aion lineage shares organism drives but owns its temporal substrate."""

    brain_family = "aion"

    def __init__(
        self, cfg: dict[str, Any], seed: int, device: str = "cpu", body: BodySpec | None = None
    ) -> None:
        normalized = dict(cfg)
        reward = dict(cfg.get("reward", {}))
        explicit_blackout = reward.get("blackout")
        if explicit_blackout not in (None, "priced", "suspended"):
            raise ValueError(
                "Aion requires reward.blackout: priced or suspended for continuous identity"
            )
        reward.setdefault("homeostasis", "drive")
        reward.setdefault("blackout", "priced")
        normalized["reward"] = reward
        super().__init__(normalized, seed=seed, device=device, body=body)

    def _build_world_model(
        self, preset: dict[str, int], num_rays: int, wm_cfg: dict[str, Any]
    ) -> WorldModel:
        s5 = dict(wm_cfg.get("s5", {}))
        blocks = int(s5.get("blocks", 4))
        default_state_dim = max(2, preset["deter"] // (2 * blocks))
        cfg = S5DynamicsConfig(
            model_dim=int(s5.get("model_dim", preset["units"])),
            state_dim=int(s5.get("state_dim", default_state_dim)),
            blocks=blocks,
            stoch_groups=preset["groups"],
            stoch_classes=preset["classes"],
            hidden=preset["hidden"],
            slow_fraction=float(s5.get("slow_fraction", 0.5)),
            dt_min=float(s5.get("dt_min", 0.001)),
            dt_max=float(s5.get("dt_max", 0.1)),
            projection_precision=str(s5.get("projection_precision", "fp32")),
            unimix=float(wm_cfg.get("unimix", 0.01)),
            free_bits=float(wm_cfg.get("kl_free_bits", 1.0)),
        )
        dynamics = S5Dynamics(cfg, embed_dim=preset["units"], action_dim=ACTION_DIM)
        return WorldModel(preset, num_rays, wm_cfg, dynamics=dynamics)

    def _migrate_world_model_state(self, state: dict[str, torch.Tensor]) -> bool:
        s5_migrated = migrate_complex_s5_state(state)
        shared_migrated = super()._migrate_world_model_state(state)
        return s5_migrated or shared_migrated

    def _state_dict(self) -> dict[str, Any]:
        state = super()._state_dict()
        state["aion_s5_format"] = S5_CHECKPOINT_FORMAT
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stored_family = state.get("brain_family", "dreamer")
        if stored_family != self.brain_family:
            super().load_state_dict(state)
            return
        wm_state = state.get("wm", {})
        has_complex = any(
            key.endswith(".input_matrix") or key.endswith(".output_matrix")
            for key in wm_state
        )
        has_paired = any(key.endswith(".input_matrix_real") for key in wm_state)
        stored_format = state.get("aion_s5_format")
        if stored_format is None:
            if has_paired:
                raise ValueError("paired-real Aion checkpoint is missing aion_s5_format")
            if not has_complex:
                raise ValueError("Aion checkpoint has no recognized S5 parameter format")
        elif stored_format != S5_CHECKPOINT_FORMAT:
            raise ValueError(
                f"unsupported Aion S5 checkpoint format {stored_format!r}; "
                f"expected {S5_CHECKPOINT_FORMAT!r}"
            )
        elif has_complex:
            raise ValueError("paired-real Aion checkpoint declares native-complex parameters")
        super().load_state_dict(state)

    @property
    def s5(self) -> S5Dynamics:
        dynamics = self.wm.dynamics
        if not isinstance(dynamics, S5Dynamics):
            raise RuntimeError("Aion was constructed without S5 dynamics")
        return dynamics

    def wake(self, dormant_steps: int = 0) -> None:
        """Wake the same organism after a measured sensory blackout.

        Fast sensorimotor modes and the observation-grounded stochastic state
        are cleared. Slow identity/context modes remain and advance by the
        number of missed perception cycles on the first wake transition.
        """
        if dormant_steps < 0:
            raise ValueError("dormant_steps cannot be negative")
        live_dynamics = self._inference.dynamics if self._inference is not None else self.s5
        if not isinstance(live_dynamics, S5Dynamics):
            raise RuntimeError("Aion inference snapshot does not contain S5 dynamics")
        self.h = live_dynamics.reset_fast(self.h)
        self.z = torch.zeros_like(self.z)
        self.last_action = torch.zeros_like(self.last_action)
        self._stream_wake = True
        self._step_scale = float(dormant_steps + 1)
        if self.blackout == "suspended":
            self._prev_drive = None
            self._prev_via = None
            self._prev_integrity = None
        self._active_skill = -1
        self._skill_remaining = 0
        self._metrics["blackout_steps"] = float(dormant_steps)
        self._metrics["blackout_discount"] = self.gamma**dormant_steps

    def _death_transition_context(self, dormant: bool, dormant_steps: int) -> tuple[bool, float]:
        if dormant_steps < 0:
            raise ValueError("dormant_steps cannot be negative")
        return dormant, float(dormant_steps + 1) if dormant else 1.0
