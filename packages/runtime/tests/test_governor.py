"""Adaptive wall-clock rate derives from measured work, never GPU labels."""

from typing import cast

import pytest
from gol_brains.base import Brain
from gol_runtime.config import PacingConfig
from gol_runtime.governor import VirtualTimeGovernor
from gol_runtime.scheduler import LearnerSnapshot, LearnerThread, Population
from gol_world.interface import Action, Observation


class RatioBrain(Brain):
    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def target_train_ratio(self) -> float:
        return self.ratio


class FakePopulation:
    def __init__(self, count: int, ratio: float) -> None:
        self.brains = {f"aion_{index}": RatioBrain(ratio) for index in range(count)}
        self.awake = list(self.brains)

    def awake_learning_ids(self) -> list[str]:
        return self.awake


class FakeLearner:
    def __init__(
        self,
        capacity: float,
        debts: dict[str, float],
        capacity_by_brain: dict[str, float] | None = None,
    ) -> None:
        self.capacity = capacity
        self.debts = debts
        if capacity_by_brain is None:
            per_brain = capacity / len(debts) if debts else 0.0
            capacity_by_brain = {rid: per_brain for rid in debts}
        self.capacity_by_brain = capacity_by_brain

    def snapshot(self) -> LearnerSnapshot:
        update_seconds = {
            rid: 1.0 / rate for rid, rate in self.capacity_by_brain.items() if rate > 0.0
        }
        return LearnerSnapshot(self.capacity, dict(self.debts), update_seconds, {})


def _governor(
    population: FakePopulation,
    learner: FakeLearner,
    config: PacingConfig,
) -> VirtualTimeGovernor:
    return VirtualTimeGovernor(
        cast(Population, population),
        cast(LearnerThread, learner),
        config,
        nominal_tick_rate=20,
        act_every=5,
    )


def test_safe_rate_uses_measured_aggregate_learner_capacity() -> None:
    population = FakePopulation(count=3, ratio=0.25)
    learner = FakeLearner(capacity=12.0, debts={rid: 0.0 for rid in population.brains})
    governor = _governor(
        population,
        learner,
        PacingConfig(mode="adaptive", headroom=0.8, hysteresis=0.0),
    )

    decision = governor.decision(all_dormant=False)

    assert decision.tick_rate == pytest.approx(64.0)
    assert decision.limiting_subsystem == "learner"


def test_safe_rate_excludes_dormant_learner_capacity() -> None:
    population = FakePopulation(count=2, ratio=0.25)
    population.awake = ["aion_0"]
    learner = FakeLearner(
        capacity=12.0,
        debts={rid: 0.0 for rid in population.brains},
        capacity_by_brain={"aion_0": 2.0, "aion_1": 10.0},
    )
    governor = _governor(
        population,
        learner,
        PacingConfig(mode="adaptive", headroom=0.8, hysteresis=0.0),
    )

    decision = governor.decision(all_dormant=False)

    assert decision.tick_rate == pytest.approx(32.0)


def test_fast_brain_cannot_subsidize_slow_independent_brain() -> None:
    population = FakePopulation(count=2, ratio=0.25)
    learner = FakeLearner(
        capacity=12.0,
        debts={rid: 0.0 for rid in population.brains},
        capacity_by_brain={"aion_0": 2.0, "aion_1": 10.0},
    )
    governor = _governor(
        population,
        learner,
        PacingConfig(mode="adaptive", headroom=0.8, hysteresis=0.0),
    )

    assert governor.decision(all_dormant=False).tick_rate == pytest.approx(32.0)


def test_backpressure_hysteresis_bounds_causal_lag() -> None:
    population = FakePopulation(count=1, ratio=0.25)
    learner = FakeLearner(capacity=2.0, debts={"aion_0": 4.0})
    governor = _governor(population, learner, PacingConfig())

    assert governor.decision(all_dormant=False).backpressured
    learner.debts["aion_0"] = 2.5
    assert governor.decision(all_dormant=False).backpressured
    learner.debts["aion_0"] = 2.0
    assert not governor.decision(all_dormant=False).backpressured


def test_universal_dormancy_waits_for_whole_update_then_releases() -> None:
    population = FakePopulation(count=1, ratio=0.25)
    learner = FakeLearner(capacity=2.0, debts={"aion_0": 1.0})
    governor = _governor(population, learner, PacingConfig())

    decision = governor.decision(all_dormant=True)
    assert decision.backpressured and decision.reason == "dormancy_consolidation"
    learner.debts["aion_0"] = 0.75
    assert not governor.decision(all_dormant=True).backpressured


def test_drop_ablation_still_pays_remaining_debt_before_dormancy_accelerates() -> None:
    population = FakePopulation(count=1, ratio=0.25)
    learner = FakeLearner(capacity=2.0, debts={"aion_0": 1.0})
    governor = _governor(
        population,
        learner,
        PacingConfig(debt_policy="drop"),
    )

    assert governor.decision(all_dormant=True).reason == "dormancy_consolidation"
    learner.debts["aion_0"] = 0.75
    assert not governor.decision(all_dormant=True).backpressured


def test_warmup_is_work_conserving_until_first_update_needs_measurement() -> None:
    population = FakePopulation(count=1, ratio=0.25)
    learner = FakeLearner(capacity=0.0, debts={"aion_0": 0.0})
    governor = _governor(
        population,
        learner,
        PacingConfig(mode="adaptive", hysteresis=0.0, max_tick_rate=500.0),
    )
    assert governor.decision(all_dormant=False).tick_rate == 500.0
    learner.debts["aion_0"] = 1.0
    assert governor.decision(all_dormant=False).tick_rate == 20.0


def test_fast_forward_does_not_pollute_world_step_capacity() -> None:
    population = FakePopulation(count=0, ratio=0.25)
    learner = FakeLearner(capacity=0.0, debts={})
    governor = _governor(
        population,
        learner,
        PacingConfig(mode="adaptive", headroom=0.8, hysteresis=0.0),
    )
    governor.observe_advance(1, processing_seconds=0.01)
    baseline = governor.decision(all_dormant=False)

    for _ in range(20):
        governor.observe_advance(
            1000,
            processing_seconds=0.0001,
            world_timing=False,
        )

    after_fast_forward = governor.decision(all_dormant=False)
    assert baseline.tick_rate == pytest.approx(80.0)
    assert after_fast_forward.tick_rate == baseline.tick_rate
    assert after_fast_forward.limiting_subsystem == "world"
