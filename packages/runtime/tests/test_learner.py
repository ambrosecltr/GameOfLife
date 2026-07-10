"""LearnerThread pacing: updates track train_ratio; backpressure skips, never banks."""

import threading
import time
from typing import cast

from gol_brains.base import Brain
from gol_runtime.scheduler import LearnerThread, Population
from gol_world.interface import Action, Observation


class CountingBrain(Brain):
    def __init__(self, ratio: float) -> None:
        self.ratio = ratio
        self.acts = 0
        self.learned = 0

    def act(self, obs: Observation) -> Action:  # pragma: no cover - learner never acts
        raise NotImplementedError

    def learn(self) -> dict[str, float] | None:
        self.learned += 1
        return {"ok": 1.0}

    def experience_count(self) -> int:
        return self.acts

    def target_train_ratio(self) -> float:
        return self.ratio


class FakePopulation:
    def __init__(self, brains: dict[str, Brain]) -> None:
        self.brains = brains
        self.locks = {rid: threading.Lock() for rid in brains}

    def learning_ids(self) -> list[str]:
        return list(self.brains)


def _learner(brains: dict[str, Brain]) -> LearnerThread:
    return LearnerThread(cast(Population, FakePopulation(brains)), idle_seconds=0.01)


def test_first_sight_carries_no_retroactive_debt() -> None:
    brain = CountingBrain(ratio=1.0)
    brain.acts = 5000  # lived before the learner was watching (resume)
    lt = _learner({"dreamer_000": brain})
    assert lt._accrue("dreamer_000", brain) == 0.0


def test_debt_accrues_at_train_ratio() -> None:
    brain = CountingBrain(ratio=0.5)
    lt = _learner({"dreamer_000": brain})
    lt._accrue("dreamer_000", brain)
    brain.acts = 10
    assert lt._accrue("dreamer_000", brain) == 5.0
    brain.acts = 12
    assert lt._accrue("dreamer_000", brain) == 6.0


def test_debt_is_capped_not_banked() -> None:
    """A world outrunning the learner sheds updates (skip), it never owes them."""
    brain = CountingBrain(ratio=1.0)
    lt = _learner({"dreamer_000": brain})
    lt._accrue("dreamer_000", brain)
    brain.acts = 100_000
    assert lt._accrue("dreamer_000", brain) == LearnerThread.MAX_DEBT


def test_workers_pay_debt_exactly_then_idle() -> None:
    brains = {f"dreamer_{i:03d}": CountingBrain(ratio=0.5) for i in range(3)}
    lt = _learner(dict(brains))
    lt.start()
    try:
        deadline = time.monotonic() + 2.0
        # Wait for every worker's first-sight baseline to land at acts=0.
        while len(lt._seen_acts) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        for brain in brains.values():
            brain.acts = 40  # each owes exactly 20 updates
        while (
            any(b.learned < 20 for b in brains.values()) and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        time.sleep(0.05)  # would overshoot here if debt math were wrong
    finally:
        lt.stop()
    # Each brain's worker paid its own debt concurrently, exactly, then idled.
    assert [b.learned for b in brains.values()] == [20, 20, 20]


class BlockingBrain(CountingBrain):
    """learn() blocks until released, so a test can act mid-update."""

    def __init__(self, ratio: float) -> None:
        super().__init__(ratio)
        self.entered = threading.Event()
        self.release = threading.Event()

    def learn(self) -> dict[str, float] | None:
        self.entered.set()
        self.release.wait(2.0)
        return super().learn()


class SnapshotBrain(BlockingBrain):
    def allows_concurrent_learning(self) -> bool:
        return True


def test_snapshot_brain_learns_without_taking_action_lock() -> None:
    brain = SnapshotBrain(ratio=1.0)
    pop = FakePopulation({"dreamer_008": brain})
    lt = LearnerThread(cast(Population, pop), idle_seconds=0.01)
    lt._accrue("dreamer_008", brain)
    brain.acts = 1
    pop.locks["dreamer_008"].acquire()
    worker = threading.Thread(target=lt._work, args=("dreamer_008",))
    worker.start()
    try:
        assert brain.entered.wait(2.0), "immutable inference must not wait for the action lock"
        lt._stop.set()
        brain.release.set()
    finally:
        pop.locks["dreamer_008"].release()
        worker.join(timeout=2.0)
    assert not worker.is_alive()


def test_worker_survives_prune_during_learn() -> None:
    """Body dies while learn() holds the GPU: the worker exits cleanly instead
    of dying on the pruned pacing key (KeyError seen live on beta_08)."""
    brain = BlockingBrain(ratio=1.0)
    pop = FakePopulation({"dreamer_007": brain})
    lt = LearnerThread(cast(Population, pop), idle_seconds=0.01)
    lt._accrue("dreamer_007", brain)  # first-sight baseline
    brain.acts = 10  # owes 10 updates -> worker enters learn()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            lt._work("dreamer_007")
        except BaseException as exc:  # noqa: BLE001 - the bug was an escaping KeyError
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    assert brain.entered.wait(2.0)
    # Agent dies mid-update; supervisor reaps the body and prunes pacing state.
    del pop.brains["dreamer_007"]
    lt._seen_acts.pop("dreamer_007", None)
    lt._owed.pop("dreamer_007", None)
    brain.release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert errors == []
    assert "dreamer_007" not in lt._owed  # not resurrected by post-learn bookkeeping


def test_dead_brain_pacing_state_pruned() -> None:
    brain = CountingBrain(ratio=1.0)
    pop = FakePopulation({"dreamer_000": brain})
    lt = LearnerThread(cast(Population, pop), idle_seconds=0.01)
    lt._accrue("dreamer_000", brain)
    del pop.brains["dreamer_000"]
    lt.start()
    try:
        deadline = time.monotonic() + 2.0
        while lt.rounds < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        lt.stop()
    assert "dreamer_000" not in lt._seen_acts
    assert "dreamer_000" not in lt._owed
