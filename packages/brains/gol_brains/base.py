"""The pluggable Brain interface.

Every robot has a Brain. Scripted brains and learning brains implement the
same five methods and live side by side in the same world — the scripted ones
are the control group and the debugging probes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gol_world.interface import Action, Observation


class Brain(ABC):
    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """Choose an action. Learning brains also record the transition here."""

    def learn(self) -> dict[str, float] | None:
        """One bounded learning update; None if there is nothing to learn.

        Called by the learner thread, never by the sim thread. Scripted brains
        keep the default no-op.
        """
        return None

    def experience_count(self) -> int:
        """Lifetime act-steps recorded. The learner thread paces updates
        against this; scripted brains stay at 0 and are never scheduled."""
        return 0

    def target_train_ratio(self) -> float:
        """Desired updates per recorded act-step (training.train_ratio).

        A target, not a promise: when the learner can't keep up it skips —
        the sim never waits — so the achieved ratio (logged as
        train_ratio_eff) may run below this.
        """
        return 0.0

    def introspect(self) -> dict[str, float]:
        """Live internals for the observability layer (curiosity, losses...)."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        """Serializable state, checkpointed at the same tick as the world."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore checkpointed state. Stateless brains ignore it."""
        del state  # nothing to restore by default

    def inherit(self, state: dict[str, Any]) -> None:
        """Warm-start a newborn from a donor brain's checkpointed state.

        Default is a plain copy plus a stream reset; learning brains may
        override to mutate heritable traits (temperament) on the way in.
        """
        self.load_state_dict(state)
        self.reset_stream()

    def reset_stream(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Called when the stream of experience breaks: lineage respawn into a
        new body, or waking from dormancy (the dormant gap is never observed).

        Weights and memory persist; only the live recurrent state resets.
        """
