"""Rerun viewer layout: an overview, a health board, and one dashboard per mind.

Built programmatically so the layout always matches the population: tabs appear
and disappear with the learning agents they belong to. Charts scroll in a
trailing window instead of compressing the whole run into one screen.
"""

from __future__ import annotations

import rerun.blueprint as rrb

TRAIL_TICKS = 48_000  # ~2 sim-days visible; scrub or zoom out for more

# Brain introspection metrics, grouped into readable charts.
CURIOSITY_METRICS = ("curiosity_scaled", "pred_error_depth", "pred_error_class")
VALUE_METRICS = ("value", "reward_homeostasis")
LOSS_METRICS = ("loss_model", "loss_actor", "loss_critic", "kl", "entropy")


def _trailing(window: int = TRAIL_TICKS) -> list[rrb.VisibleTimeRange]:
    return [
        rrb.VisibleTimeRange(
            "tick",
            start=rrb.TimeRangeBoundary.cursor_relative(seq=-window),
            end=rrb.TimeRangeBoundary.cursor_relative(),
        )
    ]


def _series(name: str, origin: str | None = None, contents: list[str] | None = None,
            y_max: float | None = None) -> rrb.TimeSeriesView:
    axis = rrb.ScalarAxis(range=(0.0, y_max)) if y_max is not None else None
    return rrb.TimeSeriesView(
        name=name,
        origin=origin if origin is not None else "/",
        contents=contents if contents is not None else "$origin/**",
        axis_y=axis,
        time_ranges=_trailing(),
    )


def _overview_tab() -> rrb.Horizontal:
    return rrb.Horizontal(
        rrb.Spatial3DView(name="World", origin="world"),
        rrb.Vertical(
            _series("Energy", origin="charts/energy", y_max=105.0),
            _series(
                "Fatigue vs daylight",
                contents=["+ charts/fatigue/**", "+ charts/light_level"],
                y_max=1.05,
            ),
            _series("Food supply", contents=["+ charts/ripe_bushes", "+ charts/toxic_bushes"]),
            rrb.TextLogView(name="Events", origin="events"),
        ),
        column_shares=[7, 3],
        name="Overview",
    )


def _health_tab() -> rrb.Grid:
    return rrb.Grid(
        _series("Energy", origin="charts/energy", y_max=105.0),
        _series("Integrity", origin="charts/integrity", y_max=105.0),
        _series(
            "Fatigue vs daylight",
            contents=["+ charts/fatigue/**", "+ charts/light_level"],
            y_max=1.05,
        ),
        _series("Population", contents=["+ charts/population"]),
        rrb.Spatial2DView(name="Visit heatmap", origin="charts/visit_heatmap"),
        grid_columns=2,
        name="Health",
    )


def _mind_tab(rid: str) -> rrb.Vertical:
    """One learning agent: its view of the world, vitals, and brain internals."""
    brains = f"charts/brains/{rid}"
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name=f"{rid} in the world",
                origin="world",
                contents=["+ world/**", "- world/rays/**", f"+ world/rays/{rid}"],
                # Chase-cam: the view's eye follows this dreamer's body around.
                eye_controls=rrb.EyeControls3D(
                    kind="Orbital",
                    tracking_entity=f"world/robots/{rid}",
                ),
            ),
            rrb.Vertical(
                _series("Energy", contents=[f"+ charts/energy/{rid}"], y_max=105.0),
                _series("Integrity", contents=[f"+ charts/integrity/{rid}"], y_max=105.0),
                _series("Fatigue", contents=[f"+ charts/fatigue/{rid}"], y_max=1.05),
                rrb.TextLogView(name="Life events", origin=f"events/{rid}"),
            ),
            column_shares=[5, 2],
        ),
        rrb.Horizontal(
            _series("Curiosity & prediction error",
                    contents=[f"+ {brains}/{m}" for m in CURIOSITY_METRICS]),
            _series("Value & reward", contents=[f"+ {brains}/{m}" for m in VALUE_METRICS]),
            _series("Losses", contents=[f"+ {brains}/{m}" for m in LOSS_METRICS]),
        ),
        row_shares=[3, 2],
        name=rid,
    )


def build_blueprint(dreamer_ids: list[str]) -> rrb.Blueprint:
    tabs = [_overview_tab(), _health_tab()] + [_mind_tab(rid) for rid in sorted(dreamer_ids)]
    return rrb.Blueprint(rrb.Tabs(*tabs, active_tab=0), collapse_panels=True)
