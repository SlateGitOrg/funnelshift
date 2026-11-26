"""A hand-written fixture: every user's furthest step is decided by hand."""

from __future__ import annotations

from datetime import datetime

import pytest

from funnelshift.definitions import parse_funnel
from funnelshift.oracle import furthest_steps
from funnelshift.run import FunnelRun
from helpers import ev

FAR_FUTURE = datetime(2030, 1, 1)


def fixture_rows() -> list[tuple]:
    rows: list[tuple] = []
    # u_complete: the whole path
    rows += [
        ev("c1", "u_complete", "signup"),
        ev("c2", "u_complete", "workspace_created", minutes=5),
        ev("c3", "u_complete", "teammate_invited", hours=1),
        ev("c4", "u_complete", "project_created", days=1),
        ev("c5", "u_complete", "subscribed", days=3, plan="team"),
    ]
    # u_skip: no teammate_invited, so project_created and subscribed cannot count
    rows += [
        ev("s1", "u_skip", "signup"),
        ev("s2", "u_skip", "workspace_created", hours=1),
        ev("s3", "u_skip", "project_created", days=1),
        ev("s4", "u_skip", "subscribed", days=2),
    ]
    # u_order: teammate_invited before workspace_created and again after it; the only
    # project_created sits between the two invites, so the path stops at step 3
    rows += [
        ev("o1", "u_order", "signup"),
        ev("o2", "u_order", "teammate_invited", minutes=1),
        ev("o3", "u_order", "project_created", minutes=5),
        ev("o4", "u_order", "workspace_created", minutes=10),
        ev("o5", "u_order", "teammate_invited", hours=2),
    ]
    # u_tie: workspace_created and teammate_invited at the same instant, written in two
    # different local zones; ties are in order, so the path continues
    rows += [
        ev("t1", "u_tie", "signup"),
        ev("t2", "u_tie", "workspace_created", minutes=10, tz_minutes=330),
        ev("t3", "u_tie", "teammate_invited", minutes=10, tz_minutes=0),
        ev("t4", "u_tie", "project_created", hours=1, tz_minutes=-300),
    ]
    # u_day15: everything on time, but subscribed on day 15
    rows += [
        ev("d1", "u_day15", "signup"),
        ev("d2", "u_day15", "workspace_created", days=1),
        ev("d3", "u_day15", "teammate_invited", days=2),
        ev("d4", "u_day15", "project_created", days=3),
        ev("d5", "u_day15", "subscribed", days=15),
    ]
    # u_dups: three events delivered twice
    rows += [
        ev("p1", "u_dups", "signup"),
        ev("p1", "u_dups", "signup"),
        ev("p2", "u_dups", "workspace_created", hours=1),
        ev("p2", "u_dups", "workspace_created", hours=1),
        ev("p3", "u_dups", "teammate_invited", hours=2),
        ev("p3", "u_dups", "teammate_invited", hours=2),
    ]
    # u_pre: three events before signup are dropped; only signup and one later step remain
    rows += [
        ev("b1", "u_pre", "workspace_created", days=-2),
        ev("b2", "u_pre", "project_created", days=-1),
        ev("b3", "u_pre", "subscribed", days=-1),
        ev("b4", "u_pre", "signup"),
        ev("b5", "u_pre", "workspace_created", days=1),
    ]
    # u_double: signs up on day 0 and day 10; the window runs from the first signup, so
    # the day-20 subscription is out of a 14-day window
    rows += [
        ev("w1", "u_double", "signup"),
        ev("w2", "u_double", "signup", days=10),
        ev("w3", "u_double", "workspace_created", days=11),
        ev("w4", "u_double", "teammate_invited", days=12),
        ev("w5", "u_double", "project_created", days=13),
        ev("w6", "u_double", "subscribed", days=20),
    ]
    return rows


EXPECTED_14 = {
    "u_complete": 5,
    "u_skip": 2,
    "u_order": 3,
    "u_tie": 4,
    "u_day15": 4,
    "u_dups": 3,
    "u_pre": 2,
    "u_double": 4,
}
EXPECTED_21 = {**EXPECTED_14, "u_day15": 5, "u_double": 5}


def test_hand_written_furthest_steps_14_day_window(trial):
    run = FunnelRun.from_rows(fixture_rows(), trial, as_of=FAR_FUTURE)
    assert run.furthest() == EXPECTED_14


def test_hand_written_furthest_steps_21_day_window(trial_21):
    run = FunnelRun.from_rows(fixture_rows(), trial_21, as_of=FAR_FUTURE)
    assert run.furthest() == EXPECTED_21


def test_oracle_agrees_with_the_hand_written_answers(trial, trial_21):
    assert furthest_steps(fixture_rows(), trial) == EXPECTED_14
    assert furthest_steps(fixture_rows(), trial_21) == EXPECTED_21


def test_clean_step_accounts_for_every_dropped_row(trial):
    run = FunnelRun.from_rows(fixture_rows(), trial, as_of=FAR_FUTURE)
    stats = run.stats
    assert stats.raw_rows == len(fixture_rows())
    assert stats.duplicate_rows_dropped == 3
    assert stats.before_signup_rows_dropped == 3
    assert stats.no_signup_rows_dropped == 0
    assert stats.kept_rows == stats.raw_rows - 6


def test_events_of_users_without_signup_are_dropped(trial):
    rows = fixture_rows() + [ev("n1", "u_ghost", "workspace_created")]
    run = FunnelRun.from_rows(rows, trial, as_of=FAR_FUTURE)
    assert "u_ghost" not in run.furthest()
    assert run.stats.no_signup_rows_dropped == 1


@pytest.mark.parametrize(
    ("extra_seconds", "expected"),
    [(0, 5), (1, 4)],
    ids=["exactly 14 days is inside", "one second later is outside"],
)
def test_window_is_closed_at_the_upper_end(trial, extra_seconds, expected):
    rows = [
        ev("a1", "u", "signup"),
        ev("a2", "u", "workspace_created", hours=1),
        ev("a3", "u", "teammate_invited", hours=2),
        ev("a4", "u", "project_created", hours=3),
        ev("a5", "u", "subscribed", days=14, seconds=extra_seconds),
    ]
    assert FunnelRun.from_rows(rows, trial, as_of=FAR_FUTURE).furthest() == {"u": expected}
    assert furthest_steps(rows, trial) == {"u": expected}


def test_timestamps_are_normalised_to_utc(trial):
    # 23:30 on Sunday 8 June in UTC-5 is 04:30 UTC on Monday 9 June: the next ISO week
    rows = [
        ev("z1", "u_tz", "signup", days=6, hours=19, minutes=30, tz_minutes=-300),
    ]
    run = FunnelRun.from_rows(rows, trial, as_of=FAR_FUTURE)
    row = run.steps_table().iloc[0]
    assert row["cohort_week"] == "2025-W24"
    assert str(row["ts_1"]) == "2025-06-09 04:30:00"


def test_property_filter_restricts_the_last_step():
    funnel = parse_funnel(
        "name: f\nsteps:\n  - event: signup\n  - event: subscribed\n    where:\n      plan: team\n"
    )
    rows = [
        ev("1", "u_team", "signup"),
        ev("2", "u_team", "subscribed", days=1, plan="team"),
        ev("3", "u_biz", "signup"),
        ev("4", "u_biz", "subscribed", days=1, plan="business"),
        ev("5", "u_mixed", "signup"),
        ev("6", "u_mixed", "subscribed", days=1, plan="business"),
        ev("7", "u_mixed", "subscribed", days=2, plan="team"),
    ]
    run = FunnelRun.from_rows(rows, funnel, as_of=FAR_FUTURE)
    assert run.furthest() == {"u_team": 2, "u_biz": 1, "u_mixed": 2}
    assert furthest_steps(rows, funnel) == run.furthest()
