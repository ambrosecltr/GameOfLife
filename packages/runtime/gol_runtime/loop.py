"""The persistent simulation loop.

One loop, one world, forever (or until asked to stop). Wall-clock paced with a
live speed multiplier; headless mode runs unpaced. Checkpoints happen at tick
boundaries so world and brains are always saved coherently together.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from gol_world import persistence
from gol_world.world import World

from gol_runtime.config import RunConfig

if TYPE_CHECKING:
    from gol_runtime.governor import VirtualTimeGovernor

FrameLogger = Callable[[World], None]
BrainStateFn = Callable[[], dict[str, bytes]]
ActStepFn = Callable[[World], None]
FastForwardReadyFn = Callable[[], bool]
FastForwardOpportunityFn = Callable[[int, int], None]
FastForwardLogFn = Callable[[World, int], None]
NextBoundaryFn = Callable[[], int | None]


class SimLoop:
    def __init__(
        self,
        world: World,
        save_dir: Path,
        run_cfg: RunConfig,
        log_frame: FrameLogger | None = None,
        brain_states: BrainStateFn | None = None,
        act_step: ActStepFn | None = None,
        after_world_step: ActStepFn | None = None,
        on_tick: ActStepFn | None = None,
        on_backpressure: ActStepFn | None = None,
        governor: VirtualTimeGovernor | None = None,
        fast_forward_ready: FastForwardReadyFn | None = None,
        fast_forward_opportunities: FastForwardOpportunityFn | None = None,
        on_fast_forward: FastForwardLogFn | None = None,
        next_lifecycle_tick: NextBoundaryFn | None = None,
        profiling: bool = False,
    ) -> None:
        self.world = world
        self.save_dir = save_dir
        self.cfg = run_cfg
        self.log_frame = log_frame
        self.brain_states: BrainStateFn = brain_states or dict
        self.act_step = act_step
        self.after_world_step = after_world_step
        self.on_tick = on_tick
        self.on_backpressure = on_backpressure
        self.governor = governor
        self.fast_forward_ready = fast_forward_ready
        self.fast_forward_opportunities = fast_forward_opportunities
        self.on_fast_forward = on_fast_forward
        self.next_lifecycle_tick = next_lifecycle_tick
        self.profiling = profiling
        self._state_lock = threading.RLock()
        # Live controls (mutated by the control API from another thread).
        self.speed = 1.0
        self.paused = False
        self.stop_requested = False
        self.checkpoint_requested = False

    def checkpoint(self) -> Path:
        with self._state_lock, self._profile_scope("runtime/checkpoint"):
            return persistence.save_checkpoint(self.save_dir, self.world, self.brain_states())

    def _profile_scope(self, name: str) -> AbstractContextManager[None]:
        if self.profiling:
            return torch.profiler.record_function(name)
        return nullcontext()

    def _service_housekeeping(self, last_frame: float, frame_interval: float) -> float:
        with self._state_lock:
            if self.checkpoint_requested:
                self.checkpoint_requested = False
                self.checkpoint()
            now = time.monotonic()
            if self.log_frame is not None and now - last_frame >= frame_interval:
                self.log_frame(self.world)
                return now
        return last_frame

    def run(self, max_ticks: int | None = None, paced: bool = True) -> None:
        """Run until stop is requested (or max_ticks more ticks have passed)."""
        end_tick = None if max_ticks is None else self.world.tick + max_ticks
        frame_interval = 1.0 / max(1, self.cfg.observability.rerun_fps)
        last_frame = 0.0
        next_tick_time = time.monotonic()
        backpressure_active = False

        while not self.stop_requested:
            if end_tick is not None and self.world.tick >= end_tick:
                break
            if self.paused:
                last_frame = self._service_housekeeping(last_frame, frame_interval)
                time.sleep(0.05)
                next_tick_time = time.monotonic()
                continue

            with self._state_lock:
                all_dormant = bool(self.world.robots) and all(
                    robot.dormant for robot in self.world.robots.values()
                )
                decision = (
                    self.governor.decision(all_dormant) if self.governor is not None else None
                )
            if decision is not None and decision.backpressured:
                with self._state_lock:
                    if not backpressure_active and self.on_backpressure is not None:
                        with self._profile_scope("runtime/logs"):
                            self.on_backpressure(self.world)
                    backpressure_active = True
                last_frame = self._service_housekeeping(last_frame, frame_interval)
                time.sleep(0.005)
                next_tick_time = time.monotonic()
                continue
            backpressure_active = False

            with self._state_lock:
                if (
                    all_dormant
                    and self.cfg.dormancy_acceleration.event_fast_forward
                    and not self.checkpoint_requested
                    and self.fast_forward_ready is not None
                    and self.fast_forward_ready()
                ):
                    maximum = self.cfg.dormancy_acceleration.max_jump_ticks
                    if end_tick is not None:
                        maximum = min(maximum, end_tick - self.world.tick)
                    checkpoint_tick = (
                        self.world.tick // self.cfg.checkpoint_interval_ticks + 1
                    ) * self.cfg.checkpoint_interval_ticks
                    metrics_tick = (
                        self.world.tick // self.cfg.observability.metrics_every_ticks + 1
                    ) * self.cfg.observability.metrics_every_ticks
                    maximum = min(
                        maximum,
                        checkpoint_tick - self.world.tick - 1,
                        metrics_tick - self.world.tick - 1,
                    )
                    if self.next_lifecycle_tick is not None:
                        lifecycle_tick = self.next_lifecycle_tick()
                        if lifecycle_tick is not None:
                            maximum = min(maximum, lifecycle_tick - self.world.tick - 1)
                    if maximum > 0:
                        start_tick = self.world.tick
                        advance_began = time.monotonic()
                        with self._profile_scope("runtime/dormant_fast_forward"):
                            advanced = self.world.fast_forward_dormant(maximum)
                        if advanced > 0:
                            if self.fast_forward_opportunities is not None:
                                self.fast_forward_opportunities(start_tick, self.world.tick)
                            if self.on_fast_forward is not None:
                                self.on_fast_forward(self.world, start_tick)
                            if self.governor is not None:
                                self.governor.observe_advance(
                                    advanced,
                                    time.monotonic() - advance_began,
                                    world_timing=False,
                                )
                            next_tick_time = time.monotonic()
                            continue

                advance_began = time.monotonic()
                with self._profile_scope("runtime/world_step"):
                    self.world.step()
                if self.after_world_step is not None:
                    with self._profile_scope("runtime/lifecycle"):
                        self.after_world_step(self.world)

                act_seconds = None
                if self.act_step is not None and self.world.tick % self.cfg.act_every == 0:
                    act_began = time.monotonic()
                    with self._profile_scope("runtime/inference"):
                        self.act_step(self.world)
                    act_seconds = time.monotonic() - act_began

                if self.on_tick is not None:
                    with self._profile_scope("runtime/logs"):
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

                if self.governor is not None:
                    self.governor.observe_advance(
                        1, time.monotonic() - advance_began, act_seconds=act_seconds
                    )

            dormancy_unpaced = all_dormant and self.cfg.dormancy_acceleration.exact_unpaced
            governor_paced = self.cfg.pacing.mode == "adaptive"
            if (paced or governor_paced) and not dormancy_unpaced:
                tick_rate = decision.tick_rate if decision is not None else self.cfg.tick_rate
                speed_scale = min(1.0, self.speed) if governor_paced else self.speed
                next_tick_time += 1.0 / (tick_rate * speed_scale)
                sleep_for = next_tick_time - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif sleep_for < -1.0:
                    # Fell badly behind (e.g. laptop slept): resync instead of
                    # sprinting to catch up.
                    next_tick_time = time.monotonic()
            else:
                next_tick_time = time.monotonic()

        # Always leave a coherent checkpoint behind.
        with self._state_lock:
            self.checkpoint()
            if self.log_frame is not None:
                self.log_frame(self.world)

    def request_checkpoint(self) -> int:
        with self._state_lock:
            self.checkpoint_requested = True
            return self.world.tick

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "tick": self.world.tick,
                "day_fraction": round(self.world.day_fraction, 4),
                "light_level": round(self.world.light_level, 3),
                "population": len(self.world.robots),
                "paused": self.paused,
                "speed": self.speed,
                "runtime": self._runtime_status(),
            }

    def runtime_status(self) -> dict[str, float | str]:
        with self._state_lock:
            return self._runtime_status()

    def _runtime_status(self) -> dict[str, float | str]:
        if self.governor is None:
            return {
                "safe_ticks_per_second": float(self.cfg.tick_rate * self.speed),
                "limiting_subsystem": "configured",
            }
        return self.governor.status()
