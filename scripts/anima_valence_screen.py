"""Offline level-valence screen for the anima track (anima_06 pre-registration).

anima_05 measured the reduction-based modulator dead on arrival: comfort M was
negative on 99.9% of act-steps and telescoped to a net-negative life return
(~-1.2), and the rectified viability gate was inert (m_viability ≡ 0.000 across
1.4M ticks). The diagnosis (see journal anima/006): the plastic brain has no
critic to integrate a stream of *changes* back into a *level*, so feeding it
`Δfeeling` teaches almost nothing — "being fed" earns zero because it is not a
transition. The fix is to read the feeling LEVEL directly:

    comfort (level):   M_comfort =  comfort_gain · (d_ref − d)      # fed>0, hungry<0
    viability (level):  M_via     = −via_level_gain · V             # standing danger

This screen REPLAYS the recorded anima_05 plastic lives (real energy/integrity
trajectories from metrics.ndjson, awake steps only — no feeling in dormancy) and
contrasts the two forms, sweeping the one free knob `d_ref` (the neutral hunger
level where reward crosses to punishment). It settles, before we spend anima_06:

  P1  sign/monotonicity: is level-M > 0 above the neutral energy, < 0 below,
      monotone in energy?
  P2  fed→positive, starving→negative: does a well-fed life integrate positive
      and a starving life negative (vs reduction, where every life telescopes
      negative)?
  P3  does fullness (hence FORAGING) earn the reward — i.e. is per-robot
      level-return positively correlated with mean energy AND with eat count,
      so idle camping cannot out-score foraging in this world?
  P4  viability activation: does the standing term fire (V>0 fraction, mean
      contribution) where the rectified gate was ≡0?

Usage:
    uv run python scripts/anima_valence_screen.py saves/anima_05
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from gol_brains import feeling

# anima_05 valence params (configs/brain/anima_05_plastic.yaml), base genes = 1.0.
COMFORT_GAIN = 3.0
POW_M, POW_N = 3.0, 2.0
SETPOINTS = torch.tensor([0.85, 1.0, 1.0])
WEIGHTS = torch.tensor([1.0, 1.0, 0.5])
VIA_CAP, E_SAFE, I_SAFE = 4.0, 0.25, 0.5


def load(save: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Counter[str]]:
    """Per-robot AWAKE (energy, integrity) trajectories (0..1) + eat counts."""
    rows = [json.loads(line) for line in (save / "metrics.ndjson").open()]
    en: dict[str, list[float]] = defaultdict(list)
    ig: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for rid, v in r.get("robots", {}).items():
            if v.get("brain") != "plastic" or v.get("dormant"):
                continue
            en[rid].append(v.get("energy", 0.0) / 100.0)
            ig[rid].append(v.get("integrity", 0.0) / 100.0)
    eats: Counter[str] = Counter()
    for line in (save / "events.ndjson").open():
        e = json.loads(line)
        if e.get("kind") == "eat" and "robot" in e:
            eats[e["robot"]] += 1
    energy = {k: np.asarray(v, np.float32) for k, v in en.items() if len(v) >= 8}
    integ = {k: np.asarray(ig[k], np.float32) for k in energy}
    return energy, integ, eats


def levels(energy: np.ndarray, integ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Comfort drive d and viability V along a trajectory, via shared feeling.*"""
    T = len(energy)
    pr = torch.zeros(T, 19)
    pr[:, feeling.ENERGY_IDX] = torch.from_numpy(energy)
    pr[:, feeling.INTEGRITY_IDX] = torch.from_numpy(integ)
    # fatigue (idx 14) unknown offline; 0 => fully rested (isolates the energy axis)
    d = feeling.drive_level(pr, SETPOINTS, WEIGHTS, POW_M, POW_N).numpy()
    V = feeling.viability(
        pr, barrier_cap=VIA_CAP, energy_safe=E_SAFE, integrity_safe=I_SAFE
    ).numpy()
    return d, V


def m_reduction(d: np.ndarray, V: np.ndarray, via_gain: float) -> np.ndarray:
    """Current anima_05 M: comfort drive-reduction + rectified viability escape."""
    mc = np.zeros_like(d)
    mc[1:] = COMFORT_GAIN * (d[:-1] - d[1:])
    redv = np.zeros_like(V)
    redv[1:] = np.maximum(V[:-1] - V[1:], 0.0)  # rectified
    return mc + via_gain * redv


def m_level(d: np.ndarray, V: np.ndarray, d_ref: float, via_gain: float) -> np.ndarray:
    """Proposed level M: satisfaction relative to a neutral hunger level, minus
    a standing danger term."""
    return COMFORT_GAIN * (d_ref - d) - via_gain * V


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def screen(save: Path, d_refs: list[float], via_gain: float) -> None:
    energy, integ, eats = load(save)
    print(f"\n{save.name}: {len(energy)} plastic lives with >=8 awake snapshots")

    # Precompute per-robot levels once.
    d_by, V_by = {}, {}
    for rid in energy:
        d_by[rid], V_by[rid] = levels(energy[rid], integ[rid])

    all_e = np.concatenate([energy[r] for r in energy])
    all_V = np.concatenate([V_by[r] for r in energy])
    print(
        f"awake energy: mean={all_e.mean()*100:.1f} median={np.median(all_e)*100:.1f}  "
        f"V>0 (in danger): {np.mean(all_V > 1e-6):.1%}   "
        f"[reduction viability was inert: m_via ≡ 0]"
    )

    # P1: pure monotonicity table at the first d_ref.
    print("\n── P1  level-M vs energy (comfort term only, d_ref sweep) ──")
    hdr = "  E    d      " + "  ".join(f"dref={r:.2f}" for r in d_refs)
    print(hdr)
    for e in [15, 25, 35, 45, 55, 65, 75, 85, 95]:
        pr = torch.tensor([[0.0] * 5 + [e / 100, 1.0] + [0.0] * 12])
        d = feeling.drive_level(pr, SETPOINTS, WEIGHTS, POW_M, POW_N).item()
        cells = "  ".join(f"{COMFORT_GAIN*(r-d):+6.3f} " for r in d_refs)
        print(f"  {e:3d}  {d:.3f}   {cells}")

    # Reduction baseline: per-robot life return + correlations.
    red_ret, red_e, red_eat = [], [], []
    for rid in energy:
        red_ret.append(float(m_reduction(d_by[rid], V_by[rid], via_gain).sum()))
        red_e.append(float(energy[rid].mean()))
        red_eat.append(eats.get(rid, 0))
    print("\n── reduction baseline (current anima_05) ──")
    print(
        f"life-return: mean={np.mean(red_ret):+.2f}  frac>0={np.mean(np.array(red_ret)>0):.0%}   "
        f"corr(return, meanE)={pearson(red_ret, red_e):+.2f}  "
        f"corr(return, eats)={pearson(red_ret, red_eat):+.2f}"
    )

    # P2-P4: the level sweep.
    print(f"\n── P2-P4  level valence, d_ref sweep (via_level_gain={via_gain:.1f}) ──")
    print(
        f"{'d_ref':>6}{'ret.mean':>10}{'frac>0':>8}{'corr(ret,E)':>12}"
        f"{'corr(ret,eat)':>14}{'fed>0?':>8}{'starve<0?':>10}"
    )
    e_arr = np.array([energy[r].mean() for r in energy])
    fed_ids = [r for r in energy if energy[r].mean() >= np.percentile(e_arr, 66)]
    starve_ids = [r for r in energy if energy[r].mean() <= np.percentile(e_arr, 33)]
    for dref in d_refs:
        ret, me, ea = [], [], []
        for rid in energy:
            ret.append(float(m_level(d_by[rid], V_by[rid], dref, via_gain).sum()))
            me.append(float(energy[rid].mean()))
            ea.append(eats.get(rid, 0))
        ret = np.array(ret)
        rmap = dict(zip(energy.keys(), ret, strict=True))
        fed = np.mean([rmap[r] for r in fed_ids])
        starve = np.mean([rmap[r] for r in starve_ids])
        print(
            f"{dref:>6.2f}{ret.mean():>10.1f}{np.mean(ret>0):>8.0%}"
            f"{pearson(list(ret), me):>12.2f}{pearson(list(ret), ea):>14.2f}"
            f"{'  '+('YES' if fed>0 else 'no '):>8}{'  '+('YES' if starve<0 else 'no '):>10}"
        )

    # P4: standing viability activation (independent of d_ref).
    via_contrib = np.concatenate([-via_gain * V_by[r] for r in energy])
    print(
        f"\n── P4  standing viability term: fires on {np.mean(all_V>1e-6):.1%} of awake steps,  "
        f"mean contribution={via_contrib.mean():+.3f}  (was ≡ 0 under rectified gate)"
    )
    print(
        "\nreading: pick the smallest d_ref where fed>0 and starving<0 with the "
        "strongest corr(ret,E)/corr(ret,eat) — that is the neutral point that rewards\n"
        "staying fed (foraging) without paying agents merely to exist.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("save", type=Path)
    ap.add_argument(
        "--d-refs", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )
    ap.add_argument("--via-level-gain", type=float, default=1.0)
    args = ap.parse_args()
    screen(args.save, args.d_refs, args.via_level_gain)


if __name__ == "__main__":
    main()
