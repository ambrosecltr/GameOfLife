"""Causal wall-clock governor for a fixed virtual-time research schedule."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from gol_runtime.config import PacingConfig
from gol_runtime.scheduler import LearnerSnapshot, LearnerThread, Population


@dataclass(frozen=True)
class GovernorDecision:
    tick_rate: float
    limiting_subsystem: str
    backpressured: bool
    reason: str


class VirtualTimeGovernor:
    """Adapt execution throughput without changing virtual work or cadence."""

    def __init__(
        self,
        population: Population,
        learner: LearnerThread | None,
        config: PacingConfig,
        nominal_tick_rate: float,
        act_every: int,
    ) -> None:
        self.population = population
        self.learner = learner
        self.config = config
        self.nominal_tick_rate = nominal_tick_rate
        self.act_every = act_every
        self._target_tick_rate = nominal_tick_rate
        self._limiting_subsystem = "configured"
        self._backpressured = False
        self._backpressure_reason = "none"
        self._world_seconds_per_tick = 0.0
        self._inference_seconds = 0.0
        self._actual_tick_rate = 0.0
        self._window_began = time.monotonic()
        self._window_ticks = 0
        self._inference_deadline_misses = 0

    def _learner_snapshot(self) -> LearnerSnapshot:
        if self.learner is None:
            return LearnerSnapshot(0.0, {}, {}, {})
        return self.learner.snapshot()

    def _learner_safe_rate(self, snapshot: LearnerSnapshot) -> float:
        awake_ids = self.population.awake_learning_ids()
        ratio_budget = sum(
            self.population.brains[rid].target_train_ratio()
            for rid in awake_ids
            if rid in self.population.brains
        )
        if ratio_budget <= 0.0:
            return math.inf
        awake_capacity = self._awake_learner_capacity(snapshot, awake_ids)
        if awake_capacity <= 0.0:
            # Warmup creates no learning credit, so it has no learner limit.
            # Once the first whole update is owed, return to the nominal rate
            # until one measured completion establishes real capacity.
            payable = any(snapshot.debt_by_brain.get(rid, 0.0) >= 1.0 for rid in awake_ids)
            return self.nominal_tick_rate if payable else math.inf
        aggregate_safe_rate = awake_capacity * self.act_every / ratio_budget * self.config.headroom
        # Independent minds cannot transfer spare optimizer capacity. Preserve
        # the requested aggregate identity, but also prevent a fast sibling or
        # GPU from hiding causal lag in a slower brain.
        individual_safe_rates = []
        for rid in awake_ids:
            ratio = self.population.brains[rid].target_train_ratio()
            if ratio <= 0.0:
                continue
            seconds = snapshot.update_seconds_by_brain.get(rid, 0.0)
            if seconds > 0.0:
                individual_safe_rates.append(
                    (1.0 / seconds) * self.act_every / ratio * self.config.headroom
                )
            elif snapshot.debt_by_brain.get(rid, 0.0) >= 1.0:
                individual_safe_rates.append(self.nominal_tick_rate)
        return min(aggregate_safe_rate, *individual_safe_rates)

    @staticmethod
    def _awake_learner_capacity(snapshot: LearnerSnapshot, awake_ids: list[str]) -> float:
        return sum(
            1.0 / snapshot.update_seconds_by_brain[rid]
            for rid in awake_ids
            if snapshot.update_seconds_by_brain.get(rid, 0.0) > 0.0
        )

    def _capacity_rate(self, seconds: float, work_per_call: float) -> float:
        if seconds <= 0.0:
            return math.inf
        return work_per_call / seconds * self.config.headroom

    def decision(self, all_dormant: bool) -> GovernorDecision:
        snapshot = self._learner_snapshot()
        max_debt = max(snapshot.debt_by_brain.values(), default=0.0)
        if self.config.debt_policy == "backpressure":
            if self._backpressured:
                self._backpressured = max_debt > self.config.resume_debt_updates
            elif max_debt >= self.config.max_debt_updates:
                self._backpressured = True
        else:
            self._backpressured = False
        if all_dormant and any(debt >= 1.0 for debt in snapshot.debt_by_brain.values()):
            # Dormancy acceleration cannot outrun already earned consolidation,
            # even in an explicit credit-dropping ablation. Drop mode first
            # caps its debt, then pays every whole update that remains.
            self._backpressured = True
            self._backpressure_reason = "dormancy_consolidation"
        elif self._backpressured:
            self._backpressure_reason = "causal_lag"
        else:
            self._backpressure_reason = "none"

        if self.config.mode == "fixed":
            self._target_tick_rate = self.nominal_tick_rate
            self._limiting_subsystem = "configured"
        else:
            candidates = {
                "learner": self._learner_safe_rate(snapshot),
                "inference": self._capacity_rate(self._inference_seconds, self.act_every),
                "world": self._capacity_rate(self._world_seconds_per_tick, 1.0),
                "configured_max": self.config.max_tick_rate,
            }
            limiting, safe_rate = min(candidates.items(), key=lambda item: item[1])
            safe_rate = min(self.config.max_tick_rate, max(self.config.min_tick_rate, safe_rate))
            relative_change = abs(safe_rate - self._target_tick_rate) / max(
                self._target_tick_rate, 1e-9
            )
            if relative_change >= self.config.hysteresis:
                self._target_tick_rate = safe_rate
            self._limiting_subsystem = limiting

        return GovernorDecision(
            tick_rate=self._target_tick_rate,
            limiting_subsystem=(
                "learner_debt" if self._backpressured else self._limiting_subsystem
            ),
            backpressured=self._backpressured,
            reason=self._backpressure_reason,
        )

    def observe_advance(
        self,
        ticks: int,
        processing_seconds: float,
        act_seconds: float | None = None,
        *,
        world_timing: bool = True,
    ) -> None:
        if ticks < 1:
            raise ValueError("governor observations require at least one advanced tick")
        if not world_timing and act_seconds is not None:
            raise ValueError("fast-forward observations cannot include inference timing")
        if world_timing:
            per_tick = processing_seconds / ticks
            self._world_seconds_per_tick = self._ewma(self._world_seconds_per_tick, per_tick)
        if world_timing and act_seconds is not None:
            self._inference_seconds = self._ewma(self._inference_seconds, act_seconds)
            deadline = self.act_every / max(self._target_tick_rate, 1e-9)
            if act_seconds > deadline:
                self._inference_deadline_misses += 1
        self._window_ticks += ticks
        now = time.monotonic()
        elapsed = now - self._window_began
        if elapsed >= 0.25:
            measured = self._window_ticks / elapsed
            self._actual_tick_rate = self._ewma(self._actual_tick_rate, measured)
            self._window_began = now
            self._window_ticks = 0

    @staticmethod
    def _ewma(previous: float, sample: float, alpha: float = 0.1) -> float:
        return sample if previous == 0.0 else (1.0 - alpha) * previous + alpha * sample

    def status(self) -> dict[str, float | str]:
        snapshot = self._learner_snapshot()
        awake_capacity = self._awake_learner_capacity(
            snapshot, self.population.awake_learning_ids()
        )
        precision = ",".join(self.population.learning_precision_modes()) or "not_applicable"
        return {
            "precision": precision,
            "safe_ticks_per_second": self._target_tick_rate,
            "actual_virtual_ticks_per_second": self._actual_tick_rate,
            "learner_updates_per_second": awake_capacity,
            "total_learner_updates_per_second": snapshot.aggregate_updates_per_second,
            "max_learner_debt": max(snapshot.debt_by_brain.values(), default=0.0),
            "dropped_update_credit": sum(snapshot.dropped_credit_by_brain.values()),
            "inference_deadline_misses": float(self._inference_deadline_misses),
            "limiting_subsystem": (
                "learner_debt" if self._backpressured else self._limiting_subsystem
            ),
            "backpressure_reason": self._backpressure_reason,
        }


__all__ = ["GovernorDecision", "VirtualTimeGovernor"]
