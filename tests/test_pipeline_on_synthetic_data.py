"""The SQL pipeline recovers the generator's ground truth on the full seeded dataset."""

from __future__ import annotations

import pandas as pd

from funnelshift.weeks import expand_weeks


def test_furthest_step_equals_ground_truth_for_every_user(run14, data):
    sql = run14.furthest()
    truth = data.truth.set_index("user_id")["depth_14"].to_dict()
    assert sql == truth


def test_furthest_step_equals_ground_truth_with_a_21_day_window(run21, data):
    sql = run21.furthest()
    truth = data.truth.set_index("user_id")["depth_21"].to_dict()
    assert sql == truth


def test_cohort_week_and_segment_come_from_the_first_signup_in_utc(run14, data):
    steps = run14.steps_table().set_index("user_id")
    truth = data.truth.set_index("user_id").reindex(steps.index)
    assert (steps["cohort_week"] == truth["cohort_week"]).all()
    assert (steps["segment"] == truth["platform"] + "|" + truth["source"]).all()


def test_clean_step_removes_exactly_the_planted_rows(run14, data):
    stats = run14.stats
    assert stats.raw_rows == len(data.events)
    assert stats.duplicate_rows_dropped == data.meta["duplicate_events"]
    assert stats.before_signup_rows_dropped == data.meta["pre_signup_events"]
    assert stats.no_signup_rows_dropped == 0
    assert stats.kept_rows == len(data.events) - stats.duplicate_rows_dropped - (
        stats.before_signup_rows_dropped
    )


def test_identical_timestamp_users_are_counted_through_both_steps(run14, data):
    steps = run14.steps_table().set_index("user_id")
    same = pd.Series(False, index=steps.index)
    for k in range(2, 6):
        same |= steps[f"ts_{k}"].notna() & (steps[f"ts_{k}"] == steps[f"ts_{k - 1}"])
    assert int(same.sum()) == data.meta["tie_users"]
    planted = data.truth[data.truth["tie_step"] > 0].set_index("user_id")
    reached = steps.loc[planted.index, "furthest_step"]
    assert (reached >= planted["tie_step"]).all()
    assert set(steps.index[same]) == set(planted.index)


def test_double_signup_users_are_one_user_in_their_first_week(run14, data):
    doubles = data.truth.loc[data.truth["double_signup"], "user_id"]
    steps = run14.steps_table().set_index("user_id")
    assert len(doubles) == data.meta["double_signups"]
    truth = data.truth.set_index("user_id")
    assert (steps.loc[doubles, "cohort_week"] == truth.loc[doubles, "cohort_week"]).all()
    assert steps.index.is_unique


def test_subscription_on_day_15_to_20_is_not_counted_at_14_days(run14, data):
    late = data.truth.loc[data.truth["late_converter"], "user_id"]
    steps = run14.steps_table().set_index("user_id")
    assert (steps.loc[late, "furthest_step"] == 4).all()
    assert len(late) == data.meta["late_converters"]


def test_21_day_window_adds_exactly_the_late_converters(run14, run21, data):
    weeks = expand_weeks("2025-W31..2025-W34")

    def converters(run):
        s = run.summary()
        return int(s[(s["step_no"] == 5) & s["cohort_week"].isin(weeks)]["users_reached"].sum())

    extra = converters(run21) - converters(run14)
    assert extra == data.late_converters(weeks)
    assert extra > 0


def test_a_21_day_window_leaves_out_the_last_cohort_week_whose_window_is_still_open(run14, run21):
    # the data stops 21 days after the last signup, so 2025-W42 (ends 20 Oct) has no full
    # 21-day window while its 14-day window is complete
    assert "2025-W42" in run14.complete_weeks()
    assert "2025-W42" not in run21.complete_weeks()
    assert "2025-W41" in run21.complete_weeks()
