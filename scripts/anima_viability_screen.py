"""Offline viability-gate screen for the anima track (proposal 002, calibration
step 3). Settles the reduction-gate-vs-standing-tax architecture bet BEFORE
spending M1 time, extending the round-012 `value_vs_energy.py` method.

anima's M is a *plasticity gate*, not a maximized reward. The proposal argues
the viability barrier should enter M as a REDUCTION gate (escape-death →
positive M → consolidate the escape), the MIRROR of beta_11's standing tax
(which anima rejects the *tax* form of because, with no value function, there is
no launchpad attractor to avoid). This screen replays a recorded life (a dreamer
blob's replay buffer — real proprio with deep near-death excursions) and
contrasts the two forms' viability contribution to M:

    reduction gate:  M_via = viability_gain · (V(t-1) − V(t))      [stream-masked]
    standing tax:    M_via = − standing_gain · V(t)

The discriminating question (proposal P5): on lived near-death *escapes*, does
the reduction form emit a large POSITIVE gate (consolidate what saved the agent)
while the tax form emits a NEGATIVE gate (uniformly suppress in-danger behaviour,
crediting nothing)? And are both forms quiet at satiety (V≈0)?

Usage:
    uv run python scripts/anima_viability_screen.py \
        saves/archive/beta_10_blobs_final/dreamer_042.pt
"""

from __future__ import annotations

import argparse
import functools
import pickle
from pathlib import Path

import numpy as np
import torch
from gol_brains import feeling


def load_recorded_life(blob_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Chronological proprio (T, PROPRIO_DIM) + stream-break markers (T,) from a
    dreamer blob's replay buffer. Cloud blobs pickle CUDA storages whose
    unpickling calls torch.load without map_location, so pin it to CPU (the
    conditioning_gym recipe)."""
    orig = torch.load
    torch.load = functools.partial(orig, map_location="cpu")
    try:
        state = pickle.loads(blob_path.read_bytes())
    finally:
        torch.load = orig
    buf = state["buffer"]
    proprio = np.asarray(buf["proprio"], dtype=np.float32)
    first = np.asarray(buf["first"], dtype=np.float32)
    return proprio, first


def screen(
    proprio: np.ndarray,
    first: np.ndarray,
    *,
    viability_gain: float,
    standing_gain: float,
    barrier_cap: float,
    energy_safe: float,
    integrity_safe: float,
    escape_eps: float,
    danger_thresh: float,
) -> None:
    pr = torch.from_numpy(proprio)
    fr = torch.from_numpy(first)

    # V and its comfort-drive counterpart via the shared feeling module, with
    # anima's founder floors (energy = recoverable dormancy, integrity = death).
    V = feeling.viability(
        pr,
        barrier_cap=barrier_cap,
        energy_safe=energy_safe,
        integrity_safe=integrity_safe,
    )
    V_int = feeling.viability(  # integrity-only: the LETHAL axis (012 method note)
        pr, barrier_cap=barrier_cap, energy_weight=0.0, integrity_safe=integrity_safe
    )
    red = feeling.reduction(V, fr)  # V(t-1) − V(t), stream-masked

    Vn = V.numpy()
    redn = red.numpy()

    # Two forms' viability contribution to the gate.
    m_reduction = viability_gain * redn
    m_tax = -standing_gain * Vn

    # Event masks along the life.
    in_danger = Vn > danger_thresh  # currently deep enough to matter
    was_danger = np.concatenate([[False], in_danger[:-1]])
    safe = Vn <= 1e-6
    was_safe = np.concatenate([[True], safe[:-1]])
    escape = (redn > escape_eps) & was_danger  # was near the floor, pulled away
    approach = (redn < -escape_eps) & (Vn > danger_thresh)  # sliding toward the floor
    settled = safe & was_safe  # SUSTAINED safety (not the escape step that lands in satiety)

    def stat(mask: np.ndarray, arr: np.ndarray) -> str:
        if mask.sum() == 0:
            return "   n=0"
        v = arr[mask]
        return f"n={int(mask.sum()):>6}  mean={v.mean():+.3f}  median={np.median(v):+.3f}"

    print(f"\nrecorded life: {len(proprio):,} steps  ({int(first.sum())} stream breaks)")
    # V is a SUM of two per-term barriers (energy + integrity), each capped at
    # barrier_cap, so its ceiling is (w_e + w_i)·cap — 2·cap at founder weights.
    print(
        f"viability V:   mean={Vn.mean():.3f}  p99={np.percentile(Vn, 99):.3f}  "
        f"max={Vn.max():.3f}   in-danger(V>{danger_thresh})={in_danger.mean():.1%}"
    )
    print(
        f"integrity-only V (lethal axis): mean={V_int.numpy().mean():.3f}  "
        f"max={V_int.numpy().max():.3f}"
    )

    print("\n── viability contribution to the gate M, by event ──")
    print(f"{'event':<22}{'REDUCTION gate (anima)':<34}{'STANDING TAX (beta form)'}")
    events = (
        ("near-death ESCAPE", escape),
        ("near-death APPROACH", approach),
        ("settled-safe (V≈0)", settled),
    )
    for name, mask in events:
        print(f"{name:<22}{stat(mask, m_reduction):<34}{stat(mask, m_tax)}")

    # Verdict.
    esc_red = m_reduction[escape].mean() if escape.sum() else 0.0
    esc_tax = m_tax[escape].mean() if escape.sum() else 0.0
    app_red = m_reduction[approach].mean() if approach.sum() else 0.0
    print("\n── verdict ──")
    print(
        f"On lived escapes the REDUCTION gate is {esc_red:+.3f} "
        f"({'POSITIVE → consolidates the escape' if esc_red > 0 else 'not positive'}); "
        f"the STANDING TAX is {esc_tax:+.3f} "
        f"({'negative → suppresses, credits nothing' if esc_tax < 0 else 'not negative'})."
    )
    print(
        f"On approaches the REDUCTION gate is {app_red:+.3f} "
        f"({'NEGATIVE → suppresses the endangering behaviour' if app_red < 0 else 'not negative'})."
    )
    settled_quiet = float(np.abs(m_reduction[settled]).mean()) if settled.any() else 0.0
    sane = bool(np.isfinite(Vn).all()) and settled_quiet < 0.05
    print(
        f"signal sane (finite everywhere, gate quiet when settled-safe: "
        f"mean|M_via|={settled_quiet:.4f}): {sane}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("blob", type=Path, help="dreamer blob .pt (a recorded life)")
    ap.add_argument("--viability-gain", type=float, default=3.0)
    ap.add_argument("--standing-gain", type=float, default=1.0, help="beta's operating point")
    ap.add_argument("--barrier-cap", type=float, default=4.0)
    ap.add_argument("--energy-safe", type=float, default=0.25)
    ap.add_argument("--integrity-safe", type=float, default=0.5)
    ap.add_argument("--escape-eps", type=float, default=0.05, help="min |ΔV| to count as an event")
    ap.add_argument(
        "--danger-thresh", type=float, default=0.3, help="V above which we count as near-floor"
    )
    args = ap.parse_args()

    proprio, first = load_recorded_life(args.blob)
    screen(
        proprio,
        first,
        viability_gain=args.viability_gain,
        standing_gain=args.standing_gain,
        barrier_cap=args.barrier_cap,
        energy_safe=args.energy_safe,
        integrity_safe=args.integrity_safe,
        escape_eps=args.escape_eps,
        danger_thresh=args.danger_thresh,
    )


if __name__ == "__main__":
    main()
