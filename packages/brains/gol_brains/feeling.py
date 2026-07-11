"""One shared definition of *feeling* — the interoceptive drives every brain
family computes from the same body.

The beta (world-model) track uses these as **rewards** an actor-critic maximizes;
the anima (plastic-valence) track uses the identical functions as a **neuromodulator**
that gates Hebbian plasticity (proposal 002). Keeping one copy is deliberate: the
viability barrier is subtle (round 012 / proposal 003) and two drifting copies would
be a silent comparability bug between the tracks.

These are pure functions of `proprio` plus scalar parameters — no module state, no
device assumptions beyond the input tensor's. Each works on any leading batch/time
shape; the drive/viability reduce over the last (feature) dimension.

Proprio layout: index 5 = energy, 6 = integrity, 14 = fatigue (stable across
OBS_VERSION 3→5; v4 appended senescence at 17, v5 appended in-water at 18 —
both leave the earlier indices untouched). All are "higher is better" except
fatigue (restedness = 1 - fatigue).
"""

from __future__ import annotations

import torch

# proprio feature indices (stable across OBS_VERSION 3→4)
ENERGY_IDX = 5
INTEGRITY_IDX = 6
FATIGUE_IDX = 14


def drive_level(
    proprio: torch.Tensor,
    setpoints: torch.Tensor,
    weights: torch.Tensor,
    pow_m: float,
    pow_n: float,
) -> torch.Tensor:
    """Keramati–Gutkin drive: convex distance from internal setpoints.

    Internal state comes straight from proprio — energy, integrity, and
    restedness (1 - fatigue), all "higher is better". Only deficits below
    setpoint count (surplus is not a drive), and the convex exponents
    (m > n) let the neediest variable dominate: a starving agent is not
    consoled by being well-rested.

    `setpoints`/`weights` are length-3 tensors over (energy, integrity,
    restedness) — passed in so a lineage can carry evolved weights.
    """
    x = torch.stack(
        [proprio[..., ENERGY_IDX], proprio[..., INTEGRITY_IDX], 1.0 - proprio[..., FATIGUE_IDX]],
        dim=-1,
    )
    # Clamp to the physical range: decoded proprio (imagined bodies) can stray
    # outside [0, 1].
    deficit = (setpoints - x.clamp(0.0, 1.0)).clamp(min=0.0)
    d = (weights * deficit.pow(pow_m)).sum(-1)
    return d.pow(1.0 / pow_n)


def viability(
    proprio: torch.Tensor,
    *,
    barrier_cap: float = 4.0,
    total_cap: float = 0.0,
    energy_lethal: float = 0.0,
    energy_safe: float = 0.25,
    integrity_lethal: float = 0.0,
    integrity_safe: float = 0.5,
    energy_weight: float = 1.0,
    integrity_weight: float = 1.0,
) -> torch.Tensor:
    """Log-barrier distance to the lethal floor, for the survival-critical
    state only (energy → dormancy, integrity → death; fatigue is not lethal,
    so restedness is a comfort drive, not a viability one). 0 at or above
    `safe`, rising toward the floor and capped per component at `barrier_cap`
    so it stays finite exactly at the boundary. A positive `total_cap` also
    caps their weighted sum. Unlike the convex comfort drive, the
    marginal cost of a lost unit grows without bound as the floor nears — the
    "a calorie when starving is worth everything" asymmetry.
    """

    def barrier(x: torch.Tensor, lethal: float, safe: float) -> torch.Tensor:
        frac = ((x.clamp(0.0, 1.0) - lethal) / (safe - lethal)).clamp(min=1e-6, max=1.0)
        return (-torch.log(frac)).clamp(max=barrier_cap)

    total = energy_weight * barrier(
        proprio[..., ENERGY_IDX], energy_lethal, energy_safe
    ) + integrity_weight * barrier(proprio[..., INTEGRITY_IDX], integrity_lethal, integrity_safe)
    return total.clamp(max=total_cap) if total_cap > 0.0 else total


def reduction(level: torch.Tensor, first: torch.Tensor | None = None) -> torch.Tensor:
    """HRRL reduction term for a drive/potential `level`: how much it fell from
    the previous step (movement toward safety/comfort is positive).

    The first step of a sequence has no predecessor, so its reduction is zero —
    and so does a stream-break step (`first`): without the mask a window
    spanning a respawn pays the newborn's full tank as a spurious "reduction"
    (measured on beta_09's dreamer_043; a real meal is +0.5 but a respawn read
    +3.9). Priced blackouts don't mark the wake, so their true cross-gap delta
    stays in.
    """
    red = torch.zeros_like(level)
    red[..., 1:] = level[..., :-1] - level[..., 1:]
    if first is not None:
        red = red * (1.0 - first)
    return red


def wellbeing(
    viability_level: torch.Tensor,
    comfort_drive: torch.Tensor,
    *,
    weight: float,
    barrier_cap: float,
    comfort_decay: float,
) -> torch.Tensor:
    """Positive valence of a regulated living body.

    Viability supplies the lethal-boundary geometry while comfort distinguishes
    merely non-lethal states from fed, intact, rested ones. The result is bounded
    in ``[0, weight]``: maximal at the bodily setpoint and zero at the capped
    lethal boundary.
    """
    if barrier_cap <= 0.0:
        raise ValueError("wellbeing barrier_cap must be positive")
    safe = (1.0 - viability_level / barrier_cap).clamp(0.0, 1.0)
    regulated = torch.exp(-comfort_decay * comfort_drive.clamp_min(0.0))
    return weight * safe * regulated


def acute_integrity_loss(
    proprio: torch.Tensor,
    damage_event: torch.Tensor,
    discontinuity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrity lost on transitions explicitly marked as acute damage.

    Chronic wear and dormant decay still lower bodily wellbeing, but do not
    become a constant pain signal. Stream starts and unconscious wake gaps have
    no experienced predecessor and therefore cannot manufacture pain.
    """
    loss = torch.zeros_like(damage_event)
    loss[..., 1:] = (
        proprio[..., :-1, INTEGRITY_IDX] - proprio[..., 1:, INTEGRITY_IDX]
    ).clamp_min(0.0)
    loss = loss * (damage_event > 0.5).to(loss.dtype)
    if discontinuity is not None:
        loss = loss * (1.0 - discontinuity)
    return loss
