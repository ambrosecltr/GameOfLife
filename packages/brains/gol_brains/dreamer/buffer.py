"""Per-agent replay: one unbroken life, stored quantized.

No episodes, no dones — a robot's replay is a single continuous sequence.
Rays are stored as uint8 depth + uint8 RGB + uint8 hit-kind, everything else
float16, so a 200k-step life stays tens of MB and survives checkpointing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gol_world.interface import EVENTS_DIM, PROPRIO_DIM, SOUND_DIM, Observation


class ReplayBuffer:
    def __init__(self, capacity: int, num_rays: int, action_dim: int, seed: int) -> None:
        self.capacity = capacity
        self.num_rays = num_rays
        self.rng = np.random.default_rng(seed)
        self.depth = np.zeros((capacity, num_rays), dtype=np.uint8)
        self.rgb = np.zeros((capacity, num_rays, 3), dtype=np.uint8)
        self.kind = np.zeros((capacity, num_rays), dtype=np.uint8)
        self.proprio = np.zeros((capacity, PROPRIO_DIM), dtype=np.float16)
        self.sound = np.zeros((capacity, SOUND_DIM), dtype=np.float16)
        self.events = np.zeros((capacity, EVENTS_DIM), dtype=np.uint8)
        self.action = np.zeros((capacity, action_dim), dtype=np.float16)  # action taken AT obs
        # Optional temporal-skill identity selected at this observation.
        # -1 is the legacy flat policy. Skills are internal controller state,
        # not part of the world observation contract.
        self.skill = np.full(capacity, -1, dtype=np.int16)
        # Per-step reward salience (|realized homeostasis reward|), supplied
        # by the brain at add(): the priority signal for reward-aware replay.
        # Event flags are the wrong signal under HRRL drive reward — a meal
        # at satiety is an ate event worth exactly zero (measured on
        # swift_01: 4 meals, all at energy >= 0.96, all worthless).
        self.salience = np.zeros(capacity, dtype=np.float16)
        # Stream-break marker: 1 on a step with no lived predecessor (first
        # act of a new body, or a wake the brain treats as a cut). The ring
        # stays adjacent across breaks; this is how learn() knows a window's
        # cross-gap drive delta is fictional. Census of beta_09's dreamer_043
        # measured what happens without it: 8 death->rebirth stitches read as
        # +3.9 reward (vs +0.5 for a real meal) and landed in ~61% of batches.
        self.first = np.zeros(capacity, dtype=np.uint8)
        # A wake is not a new life: Aion clears fast sensorimotor modes while
        # its slow context survives. It therefore needs a marker distinct
        # from `first`, which always clears the entire recurrent state.
        self.wake = np.zeros(capacity, dtype=np.uint8)
        # Number of missed perception/action opportunities represented by
        # this transition. Normally 1; a wake carries the blackout duration
        # so continuous-time dynamics decay by the same simulated time live
        # and during replay consolidation.
        self.step_scale = np.ones(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(
        self,
        obs: Observation,
        action: npt.NDArray[np.float32],
        salience: float = 0.0,
        first: bool = False,
        wake: bool = False,
        step_scale: float = 1.0,
        skill: int = -1,
    ) -> None:
        if not np.isfinite(step_scale) or step_scale < 1.0:
            raise ValueError("step_scale must be finite and at least 1")
        i = self.pos
        self.salience[i] = salience
        self.first[i] = first
        self.wake[i] = wake
        self.step_scale[i] = step_scale
        self.depth[i] = np.clip(obs["rays"][:, 0] * 255, 0, 255).astype(np.uint8)
        self.rgb[i] = np.clip(obs["rays"][:, 1:4] * 255, 0, 255).astype(np.uint8)
        self.kind[i] = obs["rays"][:, 4:].argmax(axis=1).astype(np.uint8)
        self.proprio[i] = obs["proprio"]
        self.sound[i] = obs["sound"]
        self.events[i] = np.clip(obs["events"], 0, 1).astype(np.uint8)
        self.action[i] = action
        self.skill[i] = skill
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample_sequences(
        self,
        batch: int,
        length: int,
        recent: int = 0,
        prioritized: int = 0,
        spike_offset: int = 0,
        spike_threshold: float = 0.1,
    ) -> dict[str, npt.NDArray[np.float32]] | None:
        """Contiguous sequences that do not cross the ring's write seam.

        `recent` pins that many of the batch's rows to the newest experience
        (row r ends r*length steps before the write head), DreamerV3's
        online-queue mixing: without it, a long lifelong buffer means most
        gradient flows to ancient experience and fresh events wait ~capacity/
        batch*length updates for their first replay.

        `prioritized` rows are drawn from windows containing a reward-salient
        step (salience > spike_threshold; reward-aware replay, round 009's
        reachability finding: ~53 meals in 2.8M ticks means a uniformly
        sampled reward head trains on essentially zero loud-homeostasis
        events — the actor cannot plan toward a spike its head has never
        learned to predict). This changes what is learned from, never what
        is rewarded. The spike lands at window position >= `spike_offset`
        (callers pass burn_in so it falls in the graded region). No salient
        steps lived yet -> those rows fall back uniform.
        """
        n = len(self)
        if n < length + 2:
            return None
        rows: list[npt.NDArray[np.int64]] = []
        for r in range(min(recent, batch)):
            end = self.pos - r * length
            if end - length < self.pos - n:
                break  # staggered window fell off the oldest edge; go uniform
            # Modulo walks the ring across the array end — that crossing is
            # time-contiguous; only the write seam at pos is a discontinuity,
            # and these windows end at (or stagger back from) exactly there.
            rows.append(np.arange(end - length, end, dtype=np.int64) % self.capacity)
        if prioritized > 0:
            spikes_raw = np.flatnonzero(self.salience[:n] > spike_threshold)
            if spikes_raw.size:
                # Work in time coordinates (0 = oldest): windows clamped to
                # [0, n - length] are seam-safe by construction, then map back
                # to ring positions. When full, time t lives at raw (pos+t).
                t_spikes = (spikes_raw - self.pos) % self.capacity if self.full else spikes_raw
                for _ in range(min(prioritized, batch - len(rows))):
                    t = int(t_spikes[self.rng.integers(0, t_spikes.size)])
                    off = int(self.rng.integers(min(spike_offset, length - 1), length))
                    start = min(max(t - off, 0), n - length)
                    raw = (self.pos + start) % self.capacity if self.full else start
                    rows.append((raw + np.arange(length, dtype=np.int64)) % self.capacity)
        starts = []
        for _ in range(batch - len(rows)):
            for _attempt in range(20):
                s = int(self.rng.integers(0, n - length))
                # Skip sequences spanning the seam (only matters once full).
                if self.full and s < self.pos <= s + length:
                    continue
                starts.append(s)
                break
            else:
                starts.append(0)
        rows.extend(np.arange(s, s + length, dtype=np.int64) for s in starts)
        idx = np.stack(rows)  # (B, L)
        return {
            "depth": (self.depth[idx].astype(np.float32) / 255.0),
            "rgb": (self.rgb[idx].astype(np.float32) / 255.0),
            "kind": self.kind[idx].astype(np.int64).astype(np.float32),
            "proprio": self.proprio[idx].astype(np.float32),
            "sound": self.sound[idx].astype(np.float32),
            "events": self.events[idx].astype(np.float32),
            "action": self.action[idx].astype(np.float32),
            "skill": self.skill[idx].astype(np.int64).astype(np.float32),
            "first": self.first[idx].astype(np.float32),
            "wake": self.wake[idx].astype(np.float32),
            "step_scale": self.step_scale[idx].astype(np.float32),
        }

    def state_dict(self) -> dict[str, Any]:
        n = len(self)
        order = (
            np.concatenate([np.arange(self.pos, self.capacity), np.arange(self.pos)])
            if self.full
            else np.arange(self.pos)
        )
        return {
            "depth": self.depth[order][-n:],
            "rgb": self.rgb[order][-n:],
            "kind": self.kind[order][-n:],
            "proprio": self.proprio[order][-n:],
            "sound": self.sound[order][-n:],
            "events": self.events[order][-n:],
            "action": self.action[order][-n:],
            "skill": self.skill[order][-n:],
            "salience": self.salience[order][-n:],
            "first": self.first[order][-n:],
            "wake": self.wake[order][-n:],
            "step_scale": self.step_scale[order][-n:],
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = min(len(state["depth"]), self.capacity)
        for name in ("depth", "rgb", "kind", "proprio", "sound", "events", "action"):
            getattr(self, name)[:n] = state[name][-n:]
        if "skill" in state:
            self.skill[:n] = state["skill"][-n:]
        # Pre-salience checkpoints: leave zeros; DreamerBrain recomputes from
        # stored proprio/events on load when prioritization is on.
        if "salience" in state:
            self.salience[:n] = state["salience"][-n:]
        # Pre-marker checkpoints: zeros, i.e. legacy no-breaks behavior (the
        # stored life's breaks are unrecoverable — same rule as salience).
        if "first" in state:
            self.first[:n] = state["first"][-n:]
        self.wake[:n] = 0
        if "wake" in state:
            self.wake[:n] = state["wake"][-n:]
        self.step_scale[:n] = 1.0
        if "step_scale" in state:
            self.step_scale[:n] = state["step_scale"][-n:]
        self.pos = n % self.capacity
        self.full = n == self.capacity
        self.rng.bit_generator.state = state["rng_state"]
