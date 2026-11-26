from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from funnelshift import synth
from funnelshift.render import read_sql
from funnelshift.report import verdict
from funnelshift.run import connect
from funnelshift.weeks import expand_weeks

W_9_12 = expand_weeks("2025-W31..2025-W34")
W_13_16 = expand_weeks("2025-W35..2025-W38")
W_17_20 = expand_weeks("2025-W39..2025-W42")


def toy_decomposition(rows: list[tuple[str, int, int, int]]):
    """Run sql/30_mix_rate.sql on a two-step funnel.

    rows: (segment, period, entering users, converting users); period 0 is cohort week
    2025-W01, period 1 is 2025-W02.
    """
    con = connect()
    con.execute(
        "CREATE TABLE funnel_step_defs AS SELECT * FROM (VALUES (1, 'a'), (2, 'b')) "
        "AS t(step_no, step_name)"
    )
    con.execute(
        "CREATE TABLE funnel_summary (cohort_week VARCHAR, segment VARCHAR, step_no INTEGER, "
        "step_name VARCHAR, users_reached BIGINT)"
    )
    for segment, period, entered, converted in rows:
        week = f"2025-W0{period + 1}"
        con.execute("INSERT INTO funnel_summary VALUES (?, ?, 1, 'a', ?)", [week, segment, entered])
        con.execute(
            "INSERT INTO funnel_summary VALUES (?, ?, 2, 'b', ?)", [week, segment, converted]
        )
    con.execute(
        "CREATE TABLE mixrate_weeks AS SELECT * FROM "
        "(VALUES ('2025-W01', 0), ('2025-W02', 1)) AS t(week, period)"
    )
    con.execute(read_sql("mix_rate"))
    total = con.execute("SELECT * FROM mix_rate_total").df().iloc[0]
    cells = con.execute("SELECT * FROM mix_rate_cells").df().set_index("segment")
    steps = con.execute("SELECT * FROM mix_rate_steps").df()
    return total, cells, steps


def test_toy_shift_with_constant_rates_is_all_mix():
    # weights 0.60/0.25/0.15 -> 0.45/0.25/0.30 with rates 0.30/0.25/0.20 held constant
    rows = [
        ("A", 0, 2400, 720), ("B", 0, 1000, 250), ("C", 0, 600, 120),
        ("A", 1, 1800, 540), ("B", 1, 1000, 250), ("C", 1, 1200, 240),
    ]  # fmt: skip
    total, cells, _ = toy_decomposition(rows)
    assert total["conv_base"] == pytest.approx(0.2725, abs=1e-12)
    assert total["conv_test"] == pytest.approx(0.2575, abs=1e-12)
    assert total["delta"] * 100 == pytest.approx(-1.50, abs=1e-9)
    assert total["mix"] * 100 == pytest.approx(-1.50, abs=1e-9)
    assert total["rate"] * 100 == pytest.approx(0.00, abs=1e-9)
    assert cells.loc["A", "mix"] == pytest.approx(-0.15 * 0.30)
    assert cells.loc["C", "mix"] == pytest.approx(0.15 * 0.20)


def test_toy_rate_change_with_constant_mix_is_all_rate():
    rows = [
        ("A", 0, 1000, 300), ("B", 0, 1000, 200),
        ("A", 1, 1000, 250), ("B", 1, 1000, 200),
    ]  # fmt: skip
    total, _, _ = toy_decomposition(rows)
    assert total["mix"] == pytest.approx(0.0, abs=1e-12)
    assert total["rate"] == pytest.approx(-0.025, abs=1e-12)


def test_toy_cell_present_in_one_period_only_is_pure_mix():
    rows = [
        ("A", 0, 1000, 300),
        ("A", 1, 800, 240), ("B", 1, 200, 20),
    ]  # fmt: skip
    total, cells, _ = toy_decomposition(rows)
    assert cells.loc["B", "rate"] == 0.0  # no base-period rate to compare with
    assert cells.loc["B", "mix"] == pytest.approx(0.2 * 0.10)
    assert total["mix"] + total["rate"] == pytest.approx(total["delta"], abs=1e-15)


def test_sql_matches_an_independent_python_decomposition_on_random_counts():
    rng = np.random.default_rng(11)
    for _ in range(25):
        k = int(rng.integers(2, 7))
        entered = rng.integers(50, 5000, size=(2, k))
        converted = (entered * rng.uniform(0.02, 0.6, size=(2, k))).astype(int)
        rows = [
            (f"s{j}", p, int(entered[p, j]), int(converted[p, j])) for p in (0, 1) for j in range(k)
        ]
        total, _, _ = toy_decomposition(rows)
        w = entered / entered.sum(axis=1, keepdims=True)
        c = converted / entered
        mix = float(((w[1] - w[0]) * (c[0] + c[1]) / 2).sum())
        rate = float(((c[1] - c[0]) * (w[0] + w[1]) / 2).sum())
        assert total["mix"] == pytest.approx(mix, abs=1e-12)
        assert total["rate"] == pytest.approx(rate, abs=1e-12)
        assert total["delta"] == pytest.approx(mix + rate, abs=1e-12)


def test_step_level_decomposition_sums_to_the_change_in_each_conditional_rate():
    rows = [
        ("A", 0, 1000, 400), ("B", 0, 3000, 600),
        ("A", 1, 2000, 700), ("B", 1, 2000, 500),
    ]  # fmt: skip
    _, _, steps = toy_decomposition(rows)
    step2 = steps[steps["step_no"] == 2]
    r0 = (400 + 600) / (1000 + 3000)
    r1 = (700 + 500) / (2000 + 2000)
    assert (step2["mix"] + step2["rate"]).sum() == pytest.approx(r1 - r0, abs=1e-12)


@pytest.mark.parametrize(
    ("base", "test"), [(W_9_12, W_13_16), (W_13_16, W_17_20), (W_9_12, W_17_20)]
)
def test_mix_plus_rate_equals_the_observed_change_exactly(run14, base, test):
    result = run14.mix_rate(base, test)
    assert abs(result.mix + result.rate - result.delta) < 1e-12
    for _, group in result.steps.groupby("step_no"):
        observed = _conditional_change(run14, group["step_no"].iloc[0], base, test)
        assert abs((group["mix"] + group["rate"]).sum() - observed) < 1e-12


def _conditional_change(run, step_no, base, test) -> float:
    s = run.summary()

    def cond(weeks):
        part = s[s["cohort_week"].isin(weeks)]
        users = part.groupby("step_no")["users_reached"].sum()
        return users[step_no] / users[step_no - 1]

    return cond(test) - cond(base)


def test_campaign_weeks_are_a_mix_effect_not_a_regression(run14):
    result = run14.mix_rate(W_9_12, W_13_16)
    expected = synth.expected_mix_effect(W_9_12, W_13_16)
    assert abs(result.mix - expected) < 0.010
    assert abs(result.rate) < 0.012
    assert -0.031 < expected < -0.027  # "about -2.9pp"
    assert verdict(result.mix, result.rate) == "mix_driven"
    # the campaign cell is the one whose weight moved
    cells = result.cells.set_index("segment")
    assert cells.loc["android|paid_social", "w1"] > 2 * cells.loc["android|paid_social", "w0"]


def test_release_weeks_are_a_rate_effect_localised_to_android_project_created(run14):
    result = run14.mix_rate(W_13_16, W_17_20)
    worst = result.steps.sort_values("rate").iloc[0]
    assert worst["segment"].startswith("android|")
    assert worst["step_name"] == "project_created"
    assert verdict(result.mix, result.rate) == "rate_driven"
    # every android cell loses at that step, no non-android cell moves comparably
    android = result.steps[
        (result.steps["step_name"] == "project_created")
        & result.steps["segment"].str.startswith("android|")
    ]
    assert (android["rate"] < 0).all()
    others = result.steps[~result.steps["segment"].str.startswith("android|")]
    assert others["rate"].min() > worst["rate"] / 3


def test_android_conditional_rate_falls_by_about_017_at_project_created(run14):
    summary = run14.summary()
    summary = summary[summary["segment"].str.startswith("android|")]

    def conditional(weeks):
        part = summary[summary["cohort_week"].isin(weeks)]
        users = part.groupby("step_no")["users_reached"].sum()
        return users[4] / users[3]

    drop = conditional(W_13_16) - conditional(W_17_20)
    assert drop == pytest.approx(0.17, abs=0.05)


def test_requested_weeks_without_complete_cohorts_are_reported_or_rejected(run14):
    result = run14.mix_rate(W_9_12, W_13_16 + ["2025-W43"])
    assert result.unavailable_weeks == ["2025-W43"]
    with pytest.raises(ValueError, match="no complete cohorts in the test period"):
        run14.mix_rate(W_9_12, ["2026-W01"])
    with pytest.raises(ValueError, match="periods overlap"):
        run14.mix_rate(W_9_12, W_9_12[:1])


def test_verdict_thresholds():
    assert verdict(-0.029, 0.002) == "mix_driven"
    assert verdict(-0.02, 0.01) == "mix_driven"  # exactly twice: inclusive
    assert verdict(-0.0199, 0.01) == "mixed"
    assert verdict(0.001, -0.015) == "rate_driven"
    assert verdict(0.0, 0.0) == "no_change"


def test_segments_can_be_split_by_a_single_column(events_path, trial):
    from funnelshift.run import FunnelRun

    run = FunnelRun.from_parquet(events_path, trial, segments=["platform"])
    result = run.mix_rate(W_9_12, W_13_16)
    assert sorted(result.cells["segment"]) == ["android", "ios", "web"]
    assert abs(result.mix + result.rate - result.delta) < 1e-12
    assert isinstance(result.cells, pd.DataFrame)


def test_centred_mix_sums_to_the_same_total_and_signs_by_cell_effect():
    # weight moves from the high-rate cell A to the low-rate cell B
    rows = [
        ("A", 0, 1000, 400), ("B", 0, 1000, 100),
        ("A", 1, 500, 200), ("B", 1, 1500, 150),
    ]  # fmt: skip
    total, cells, steps = toy_decomposition(rows)
    assert cells["mix_c"].sum() == pytest.approx(total["mix"], abs=1e-12)
    cbar = (total["conv_base"] + total["conv_test"]) / 2
    assert cells.loc["B", "mix_c"] == pytest.approx(0.25 * ((0.1 + 0.1) / 2 - cbar))
    # losing the high-rate cell and gaining the low-rate cell both pull the average down
    assert cells.loc["A", "mix_c"] < cells.loc["B", "mix_c"] < 0
    assert cells.loc["B", "mix"] > 0  # the uncentred term has the opposite sign
    step_sum = steps.groupby("step_no")["mix_c"].sum().iloc[0]
    assert step_sum == pytest.approx(total["mix"], abs=1e-12)


def test_campaign_cell_has_the_largest_negative_mix_contribution(run14):
    result = run14.mix_rate(W_9_12, W_13_16)
    cells = result.cells.sort_values("mix_c")
    assert cells.iloc[0]["segment"] == "android|paid_social"
    assert cells.iloc[0]["mix_c"] < -0.015
    assert cells["mix_c"].sum() == pytest.approx(result.mix, abs=1e-12)
    assert cells.iloc[1]["mix_c"] > -0.005


def test_periods_are_the_exact_week_lists_not_the_range_between_their_ends(run14):
    base = ["2025-W31", "2025-W33"]  # W32 is not in the period
    result = run14.mix_rate(base, W_13_16)
    s = run14.summary()
    entering = s[(s["step_no"] == 1) & s["cohort_week"].isin(base)]["users_reached"].sum()
    contiguous = s[(s["step_no"] == 1) & s["cohort_week"].isin(W_9_12[:3])]["users_reached"].sum()
    assert result.users_base == entering
    assert result.users_base < contiguous
    last = s[(s["step_no"] == 5) & s["cohort_week"].isin(base)]["users_reached"].sum()
    assert result.conv_base == pytest.approx(last / entering, abs=1e-15)
    assert result.base_weeks == base
    assert abs(result.mix + result.rate - result.delta) < 1e-12


def test_duplicate_weeks_in_a_period_are_counted_once(run14):
    once = run14.mix_rate(W_9_12, W_13_16)
    twice = run14.mix_rate(W_9_12 + W_9_12, W_13_16)
    assert twice.users_base == once.users_base
    assert twice.delta == once.delta
