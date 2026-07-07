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
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs: Observation, action: npt.NDArray[np.float32]) -> None:
        i = self.pos
        self.depth[i] = np.clip(obs["rays"][:, 0] * 255, 0, 255).astype(np.uint8)
        self.rgb[i] = np.clip(obs["rays"][:, 1:4] * 255, 0, 255).astype(np.uint8)
        self.kind[i] = obs["rays"][:, 4:].argmax(axis=1).astype(np.uint8)
        self.proprio[i] = obs["proprio"]
        self.sound[i] = obs["sound"]
        self.events[i] = np.clip(obs["events"], 0, 1).astype(np.uint8)
        self.action[i] = action
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
    ) -> dict[str, npt.NDArray[np.float32]] | None:
        """Contiguous sequences that do not cross the ring's write seam.

        `recent` pins that many of the batch's rows to the newest experience
        (row r ends r*length steps before the write head), DreamerV3's
        online-queue mixing: without it, a long lifelong buffer means most
        gradient flows to ancient experience and fresh events wait ~capacity/
        batch*length updates for their first replay.

        `prioritized` rows are drawn from windows containing an ate/damage
        event (reward-aware replay, round 009's reachability finding: ~53
        meals in 2.8M ticks means a uniformly-sampled reward head trains on
        essentially zero positive-homeostasis events — the actor cannot plan
        toward a spike its head has never learned to predict). This changes
        what is learned from, never what is rewarded. The spike lands at
        window position >= `spike_offset` (callers pass burn_in so it falls
        in the graded region). No events lived yet -> rows fall back uniform.
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
            spikes_raw = np.flatnonzero(self.events[:n, :2].max(axis=1) > 0)
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
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = min(len(state["depth"]), self.capacity)
        for name in ("depth", "rgb", "kind", "proprio", "sound", "events", "action"):
            getattr(self, name)[:n] = state[name][-n:]
        self.pos = n % self.capacity
        self.full = n == self.capacity
        self.rng.bit_generator.state = state["rng_state"]
