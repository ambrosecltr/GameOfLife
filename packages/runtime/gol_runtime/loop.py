"""The persistent simulation loop.

One loop, one world, forever (or until asked to stop). Wall-clock paced with a
live speed multiplier; headless mode runs unpaced. Checkpoints happen at tick
boundaries so world and brains are always saved coherently together.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from gol_world import persistence
from gol_world.world import World

from gol_runtime.config import RunConfig

FrameLogger = Callable[[World], None]
BrainStateFn = Callable[[], dict[str, bytes]]
ActStepFn = Callable[[World], None]


class SimLoop:
    def __init__(
        self,
        world: World,
        save_dir: Path,
        run_cfg: RunConfig,
        log_frame: FrameLogger | None = None,
        brain_states: BrainStateFn | None = None,
        act_step: ActStepFn | None = None,
        on_tick: ActStepFn | None = None,
    ) -> None:
        self.world = world
        self.save_dir = save_dir
        self.cfg = run_cfg
        self.log_frame = log_frame
        self.brain_states: BrainStateFn = brain_states or dict
        self.act_step = act_step
        self.on_tick = on_tick
        # Live controls (mutated by the control API from another thread).
        self.speed = 1.0
        self.paused = False
        self.stop_requested = False
        self.checkpoint_requested = False

    def checkpoint(self) -> Path:
        return persistence.save_checkpoint(self.save_dir, self.world, self.brain_states())

    def run(self, max_ticks: int | None = None, paced: bool = True) -> None:
        """Run until stop is requested (or max_ticks more ticks have passed)."""
        end_tick = None if max_ticks is None else self.world.tick + max_ticks
        frame_interval = 1.0 / max(1, self.cfg.observability.rerun_fps)
        last_frame = 0.0
        next_tick_time = time.monotonic()

        while not self.stop_requested:
            if end_tick is not None and self.world.tick >= end_tick:
                break
            if self.paused:
                time.sleep(0.05)
                next_tick_time = time.monotonic()
                continue

            self.world.step()

            if self.act_step is not None and self.world.tick % self.cfg.act_every == 0:
                self.act_step(self.world)

            if self.on_tick is not None:
                self.on_tick(self.world)

            if (
                self.world.tick % self.cfg.checkpoint_interval_ticks == 0
                or self.checkpoint_requested
            ):
                self.checkpoint_requested = False
                self.checkpoint()

            now = time.monotonic()
            if self.log_frame is not None and now - last_frame >= frame_interval:
                self.log_frame(self.world)
                last_frame = now

            if paced:
                next_tick_time += 1.0 / (self.cfg.tick_rate * self.speed)
                sleep_for = next_tick_time - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif sleep_for < -1.0:
                    # Fell badly behind (e.g. laptop slept): resync instead of
                    # sprinting to catch up.
                    next_tick_time = time.monotonic()

        # Always leave a coherent checkpoint behind.
        self.checkpoint()
        if self.log_frame is not None:
            self.log_frame(self.world)
