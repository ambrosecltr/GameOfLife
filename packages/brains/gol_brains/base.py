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

        Runtime pacing changes wall-clock world speed to preserve this
        scientific budget. Dropping credit is an explicit runtime mode, never
        an implicit consequence of faster hardware.
        """
        return 0.0

    def pending_update_credit(self) -> float:
        """Checkpoint-coherent optimizer updates currently owed."""
        return 0.0

    def drop_update_credit(self, amount: float) -> None:
        """Explicitly discard owed updates in the configured drop mode."""
        if amount < 0.0:
            raise ValueError("dropped update credit cannot be negative")

    def allows_concurrent_learning(self) -> bool:
        """Whether learn() uses an immutable controller snapshot for act()."""
        return False

    def precision_mode(self) -> str:
        """Configured compute precision, for runtime and checkpoint telemetry."""
        return "not_applicable"

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
        """Called when the stream of experience breaks for good: lineage
        respawn into a new body (the previous body's end was never observed,
        and a newborn must not inherit a salience spike it didn't live).

        Weights and memory persist; only the live recurrent state resets.
        """

    def wake(self, dormant_steps: int = 0) -> None:
        """Called on the first act after a dormant spell, before that act.

        Default: the dormant gap is a stream break like any other (the
        legacy cut). Brains that price the blackout override this to keep
        the pre-collapse state as the predecessor of the wake observation —
        one visible transition carrying the gap's real energy/integrity
        delta — instead of severing the stream. `dormant_steps` is the number
        of perception/action opportunities missed while the body was dormant.
        """
        del dormant_steps
        self.reset_stream()

    def record_death(  # noqa: B027
        self, obs: Observation, dormant: bool = False, dormant_steps: int = 0
    ) -> None:
        """Called once by the runtime after the body died (integrity crossed
        the lethal floor and the world removed it).

        A dying body is never observable from inside — dormant bodies don't
        act, and the death tick removes the robot before sensing — so the
        runtime delivers the last observation it had for the body, with
        `dormant` saying whether the body was hibernating when it died.
        `dormant_steps` measures the unobserved interval when it was.
        Brains that learn a continuation/terminal signal override this to
        record the stream's real end; the default is a no-op (scripted
        brains, and learners that don't model death).
        """
        del obs, dormant, dormant_steps
