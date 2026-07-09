"""A human-readable check-in report for anima (plastic-valence) runs.

Reads a save's metrics.ndjson + events.ndjson and prints the signals that
actually matter for these rounds — population hold, budding/reproduction,
eating-while-hungry, dormancy, mortality, plasticity health, gene drift under
selection, and (v4) whether felt finitude is doing anything. Tailored to the
finitude round (proposal 004) but degrades gracefully on older saves.

    uv run python scripts/anima_stats.py saves/anima_02
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(save: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(x) for x in (save / "metrics.ndjson").open()]
    evs = [json.loads(x) for x in (save / "events.ndjson").open()]
    mpath = save / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    return rows, evs, manifest


def _window(t: int, size: int) -> int:
    return t // size


def _fmt(x: float, n: int = 2) -> str:
    return f"{x:,.{n}f}" if abs(x) >= 1 else f"{x:.{n}f}"


def report(save: Path) -> None:
    rows, evs, manifest = _load(save)
    if not rows:
        print(f"{save}: no metrics yet")
        return
    T = rows[-1]["tick"]
    run_cfg = manifest.get("run_config", {})
    repro = run_cfg.get("reproduction", {})
    thrive_e = float(repro.get("thrive_energy", 75.0))
    thrive_i = float(repro.get("thrive_integrity", 70.0))
    min_age = int(repro.get("min_bud_age", 20000))
    caps: dict[str, int] = {}
    for e in run_cfg.get("population", {}).get("mix", []):
        b = e["brain"]
        kind = str(b.get("kind")) if isinstance(b, dict) else "plastic"
        caps[kind] = int(e.get("count", 0))
    win = 100_000 if T >= 200_000 else max(10_000, T // 4 or 1)

    r2b = {e["robot"]: e["brain"] for e in evs if e["kind"] == "spawn" and "robot" in e}
    last = rows[-1]
    robots = last.get("robots", {})
    pl_live = [v for v in robots.values() if v.get("brain") == "plastic"]
    fo_live = [v for v in robots.values() if v.get("brain") == "scripted_forager"]

    mode = run_cfg.get("reproduction", {}).get("mode", "respawn")
    print(f"\n{'='*66}\n  {save.name}  @ tick {T:,}  ({mode} mode)\n{'='*66}")

    # ---- population ----
    pcap = caps.get("plastic", 0)
    fcap = caps.get("scripted_forager", 0)
    floor = int(repro.get("floor", 0))
    flag = "  <-- near floor!" if floor and len(pl_live) <= floor + 1 else ""
    print("\nPOPULATION")
    print(f"  plastic {len(pl_live)}/{pcap}{flag}    forager {len(fo_live)}/{fcap}"
          f"    total {last['population']}")

    # ---- reproduction ----
    buds = [e for e in evs if e["kind"] == "bud"]
    spawns = [e for e in evs if e["kind"] == "spawn"]
    thriving = [
        v for v in pl_live
        if not v.get("dormant") and v.get("energy", 0) >= thrive_e
        and v.get("integrity", 0) >= thrive_i and v.get("age", 0) >= min_age
    ]
    print("\nREPRODUCTION")
    print(f"  bud events: {len(buds)}    spawns: {len(spawns)}"
          f"    thriving-now: {len(thriving)}/{len(pl_live)} plastic"
          f"  (need E>={thrive_e:.0f} I>={thrive_i:.0f} age>={min_age//1000}k, awake)")
    if buds:
        recent = sum(1 for e in buds if e["tick"] >= T - win)
        print(f"  buds in last {win//1000}k: {recent}")

    # ---- eating / hunger ----
    etl: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        for k, v in r.get("robots", {}).items():
            if v.get("brain") == "plastic":
                etl[k].append((r["tick"], v.get("energy", 0.0)))

    def energy_at(rid: str, t: int) -> float | None:
        tl = etl.get(rid)
        if not tl:
            return None
        ts = [x[0] for x in tl]
        return tl[min(int(np.searchsorted(ts, t)), len(tl) - 1)][1]

    eat_w: dict[int, Counter[str]] = defaultdict(Counter)
    hungry = sated = 0
    for e in evs:
        if e["kind"] != "eat" or "robot" not in e:
            continue
        b = r2b.get(e["robot"], "?")
        eat_w[_window(e["tick"], win)][b] += 1
        if b == "plastic":
            en = energy_at(e["robot"], e["tick"])
            if en is not None:
                if en < 40:
                    hungry += 1
                else:
                    sated += 1
    pl_series = [eat_w[w]["plastic"] for w in sorted(eat_w)]
    fo_series = [eat_w[w]["scripted_forager"] for w in sorted(eat_w)]
    dorm = np.mean([1.0 if v.get("dormant") else 0.0 for v in pl_live]) if pl_live else float("nan")
    print("\nSURVIVAL / BEHAVIOUR")
    print(f"  plastic eats/{win//1000}k: {pl_series}")
    print(f"  forager eats/{win//1000}k: {fo_series}")
    print(f"  eating-while-hungry (plastic): hungry(E<40)={hungry}  sated={sated}")
    print(f"  plastic dormant fraction (now): {_fmt(float(dorm))}")

    # ---- mortality ----
    lastage: dict[str, float] = {}
    lastled: dict[str, dict[str, float]] = {}
    lasteled: dict[str, dict[str, float]] = {}
    for r in rows:
        for k, v in r.get("robots", {}).items():
            if v.get("brain") == "plastic":
                lastage[k] = v.get("age", 0)
                lastled[k] = v.get("ledger", {})
                if "energy_ledger" in v:
                    lasteled[k] = v["energy_ledger"]
    alive = set(robots)
    dead_ages = [a for k, a in lastage.items() if k not in alive]
    cause: Counter[str] = Counter()
    for k in lastage:
        if k not in alive and lastled.get(k):
            cause[max(lastled[k], key=lastled[k].get)] += 1  # type: ignore[arg-type]
    print("\nMORTALITY (plastic)")
    if dead_ages:
        print(f"  deaths={len(dead_ages)}  age median={np.median(dead_ages):,.0f}"
              f"  p10={np.percentile(dead_ages,10):,.0f}  p90={np.percentile(dead_ages,90):,.0f}")
        print(f"  dominant cause (last ledger): {dict(cause)}")
    else:
        print("  no plastic deaths yet")

    # ---- plasticity + finitude ----
    def bmean(metric: str) -> float:
        vals = [v[metric] for r in rows if r["tick"] >= T - win
                for k, v in r["brains"].items() if "plastic" in k and metric in v]
        return float(np.mean(vals)) if vals else float("nan")

    # ---- energy budget (saves with the energy ledger, anima_04+) ----
    if lasteled:
        totals: dict[str, float] = defaultdict(float)
        for led in lasteled.values():
            for k, v in led.items():
                totals[k] += v
        ticks_obs = sum(lastage.values()) or 1.0
        spend = {k: v for k, v in totals.items() if k not in ("eaten", "solar") and v > 0}
        tot_spend = sum(spend.values()) or 1.0
        meals = sum(eat_w[w]["plastic"] for w in eat_w)
        print("\nENERGY BUDGET (plastic, lifetime totals across observed robots)")
        print("  spend: " + "  ".join(
            f"{k}={v/tot_spend*100:.0f}%" for k, v in sorted(spend.items(), key=lambda x: -x[1])
        ))
        print(f"  spend rate: {tot_spend/ticks_obs:.4f}/robot-tick (dormant time included)")
        print(f"  income: eaten={totals.get('eaten', 0):,.0f}  solar={totals.get('solar', 0):,.0f}"
              + (f"  banked/meal={totals.get('eaten', 0)/meals:.1f}" if meals else ""))

    print("\nPLASTICITY & VALENCE (recent mean)")
    print(f"  w_fast_norm={_fmt(bmean('w_fast_norm'),4)}"
          f"  m_viability={_fmt(bmean('m_viability'),4)}"
          f"  life_return_via={_fmt(bmean('life_return_via'))}")
    # felt finitude: robot age vs the world's senescence half-life (proprio ch.17)
    econ = manifest.get("world_config", {}).get("economy", {})
    hl = float(econ.get("senescence_halflife", 0) or 0)
    if hl > 0 and pl_live:
        ages = [v.get("age", 0) for v in pl_live]
        felt = float(np.mean([1.0 - 0.5 ** (a / hl) for a in ages]))
        print(f"\nFINITUDE (v4)\n  mean age(plastic)={np.mean(ages):,.0f}"
              f"  felt senescence≈{_fmt(felt,3)}  (halflife {hl:,.0f})")

    # ---- gene drift ----
    genes = ["gene_viability_gain", "gene_comfort_gain", "gene_restlessness", "gene_alpha",
             "gene_integrity_weight", "gene_via_integrity_weight"]

    def gmean(lo: int, hi: int, g: str) -> float:
        vals = [v[g] for r in rows if lo <= r["tick"] < hi
                for k, v in r["brains"].items() if "plastic" in k and g in v]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nGENE DRIFT (founder <100k -> recent, selection signal)")
    for g in genes:
        a, b = gmean(0, 100_000, g), gmean(T - 100_000, T + 1, g)
        if not (np.isnan(a) or np.isnan(b)):
            print(f"  {g.replace('gene_',''):20s} {a:.3f} -> {b:.3f}  ({(b/a-1)*100:+.1f}%)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("save", type=Path, help="save directory (e.g. saves/anima_02)")
    report(ap.parse_args().save)


if __name__ == "__main__":
    main()
