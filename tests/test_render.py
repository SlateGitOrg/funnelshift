from __future__ import annotations

import pytest

from funnelshift.definitions import parse_funnel
from funnelshift.render import render_steps, sql_str


def test_one_cte_per_step_with_ordering_and_window_rules(trial):
    sql = render_steps(trial)
    for k in range(1, 6):
        assert f"s{k} AS (" in sql
    # step k is no earlier than step k-1 (ties allowed) and no later than anchor + window
    assert "e.ts >= p.ts_4" in sql
    assert "e.ts <= p.ts_1 + INTERVAL '14 days'" in sql
    assert "e.ts > p.ts" not in sql


def test_window_days_are_rendered(trial_21):
    assert "INTERVAL '21 days'" in render_steps(trial_21)
    assert "INTERVAL '14 days'" not in render_steps(trial_21)


def test_property_filters_become_equality_predicates():
    funnel = parse_funnel(
        "name: f\nsteps:\n  - event: signup\n  - event: subscribed\n    where:\n      plan: team\n"
    )
    sql = render_steps(funnel)
    assert "AND e.plan = 'team'" in sql


def test_segments_override_changes_the_key(trial):
    sql = render_steps(trial, ["platform"])
    assert "concat_ws('|', coalesce(u.platform, 'unknown'))" in sql
    with pytest.raises(ValueError, match="not a valid identifier"):
        render_steps(trial, ["platform; DROP TABLE users"])
    with pytest.raises(ValueError, match="unknown segment column"):
        render_steps(trial, ["country"])


def test_sql_str_doubles_quotes():
    assert sql_str("it's") == "'it''s'"
