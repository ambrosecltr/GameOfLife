"""anima: the plastic-valence brain family (proposal 002).

A backprop-free, world-model-free brain. A recurrent net whose fast weights
adapt online via a neuromodulated three-factor Hebbian rule, gated by an
evolved homeostatic valence `M` (the shared `gol_brains.feeling` signal, here a
neuromodulator rather than a reward). Nothing maximizes a return; feeling only
decides what sticks.
"""

from __future__ import annotations

from gol_brains.plastic.brain import PlasticBrain

__all__ = ["PlasticBrain"]
