from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from funnelshift.run import FunnelRun
from helpers import ev


def test_users_never_increase_along_the_funnel(run14):
    summary = run14.summary()
    for _, group in summary.groupby(["cohort_week", "segment"]):
        users = group.sort_values("step_no")["users_reached"].tolist()
        assert users == sorted(users, reverse=True)


def test_conversion_stays_between_zero_and_one(run14):
    summary = run14.summary()
    assert summary["cum_conv"].between(0, 1).all()
    assert summary["step_conv"].dropna().between(0, 1).all()
    first = summary[summary["step_no"] == 1]
    assert (first["cum_conv"] == 1.0).all()
    assert first["step_conv"].isna().all()


def test_cumulative_conversion_is_the_product_of_step_conversions(run14):
    summary = run14.summary()
    summary = summary[
        summary["users_reached"]
        .groupby([summary["cohort_week"], summary["segment"]])
        .transform("min")
        > 0
    ]
    product = (
        summary.assign(step_conv=summary["step_conv"].fillna(1.0))
        .groupby(["cohort_week", "segment"])["step_conv"]
        .cumprod()
    )
    assert (product - summary["cum_conv"]).abs().max() < 1e-12


def test_every_cohort_week_and_segment_cell_is_reported(run14, data):
    summary = run14.summary()
    cells = summary.groupby(["cohort_week", "segment"]).ngroups
    assert cells == 20 * 12
    entered = summary[summary["step_no"] == 1].groupby("cohort_week")["users_reached"].sum()
    expected = data.truth.groupby("cohort_week").size()
    assert entered.to_dict() == expected.to_dict()


def test_cohorts_are_reported_only_after_their_window_has_elapsed(events_path, trial):
    run = FunnelRun.from_parquet(events_path, trial, as_of=datetime(2025, 11, 2))
    status = run.cohort_status().set_index("cohort_week")
    assert not status.loc["2025-W42", "complete"]  # week ends 20 Oct; 14 days later is 3 Nov
    assert status.loc["2025-W41", "complete"]  # week ends 13 Oct; 14 days later is 27 Oct
    assert "2025-W42" not in set(run.summary()["cohort_week"])
    assert max(run.summary()["cohort_week"]) == "2025-W41"


def test_default_as_of_is_the_latest_event_and_covers_every_generated_cohort(run14):
    assert run14.complete_weeks() == {f"2025-W{w}" for w in range(23, 43)}
    assert run14.cohort_status()["complete"].all()


def test_time_between_steps_median_and_p90():
    from funnelshift.definitions import parse_funnel

    funnel = parse_funnel("name: f\nsteps:\n  - event: signup\n  - event: workspace_created\n")
    rows = []
    for i, gap in enumerate([60, 120, 180, 240, 300]):
        rows += [
            ev(f"s{i}", f"u{i}", "signup"),
            ev(f"w{i}", f"u{i}", "workspace_created", seconds=gap),
        ]
    run = FunnelRun.from_rows(rows, funnel, as_of=datetime(2030, 1, 1))
    step2 = run.summary().query("step_no == 2").iloc[0]
    assert step2["users_reached"] == 5
    assert step2["median_secs"] == pytest.approx(180.0)
    assert step2["p90_secs"] == pytest.approx(276.0)  # 240 + 0.6 * (300 - 240)
    assert pd.isna(run.summary().query("step_no == 1").iloc[0]["median_secs"])
