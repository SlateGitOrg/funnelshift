"""Differential test: the SQL's furthest step equals a brute-force oracle on random logs.

Logs are built to be adversarial: timestamps sit on a half-day grid so ties and exact window
boundaries are common, users re-sign-up, events repeat, appear out of order, before signup,
beyond the window and as exact duplicates, and timestamps are written in assorted UTC offsets.
"""

from __future__ import annotations

from datetime import datetime

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import ROOT
from funnelshift.definitions import Funnel, load_funnel, parse_funnel
from funnelshift.oracle import furthest_steps
from funnelshift.run import FunnelRun
from helpers import ev

STEPS = ("signup", "workspace_created", "teammate_invited", "project_created", "subscribed")
NAMES = STEPS + ("page_view", "help_opened")
TZ = (0, -300, 60, 330, 540)
HALF_DAY_HOURS = 12
FAR_FUTURE = datetime(2030, 1, 1)

TRIAL = load_funnel(ROOT / "funnels" / "trial.yaml")
TEAM = load_funnel(ROOT / "funnels" / "trial_team_plan.yaml")
SHORT = parse_funnel(
    "name: short\nwindow_days: 3\nsteps:\n  - event: signup\n  - event: project_created\n"
    "  - event: subscribed\n"
)
FUNNELS: list[Funnel] = [TRIAL, TEAM, SHORT]

event_spec = st.tuples(
    st.integers(0, len(NAMES) - 1),  # event name
    st.integers(-4, 40),  # half-days from the user's reference time
    st.integers(0, len(TZ) - 1),  # local offset
    st.integers(0, 2),  # plan: none, team, business
)
user_spec = st.tuples(
    st.integers(0, 5),  # planted path depth
    st.lists(st.sampled_from([0, 0, 0, 1, 2, 4, 10, 28, 29]), min_size=4, max_size=4),
    st.lists(event_spec, max_size=5),  # extra events (noise, repeats, out of order)
    st.integers(0, 6),  # signups: 0 (none) .. 1 (usual) .. 2 (re-signup)
    st.integers(0, 6),  # reference offset in half-days
)


@st.composite
def logs(draw):
    users = draw(st.lists(user_spec, min_size=1, max_size=25))
    rows: list[tuple] = []

    def add(user: str, name: str, half_days: int, tz: int, plan: int) -> None:
        plan_value = (None, "team", "business")[plan] if name == "subscribed" else None
        rows.append(
            ev(
                f"e{len(rows)}",
                user,
                name,
                hours=half_days * HALF_DAY_HOURS,
                tz_minutes=TZ[tz],
                plan=plan_value,
            )
        )

    for i, (depth, gaps, extras, signups, ref) in enumerate(users):
        user = f"u{i}"
        if signups:
            add(user, "signup", ref, 0, 0)
        if signups == 6:
            add(user, "signup", ref + gaps[0] + 1, 1, 0)
        at = ref
        for k in range(1, depth):
            at += gaps[k - 1] if k - 1 < len(gaps) else 0
            add(user, STEPS[k], at, k % len(TZ), 1 + (k % 2))
        for name_i, half_days, tz, plan in extras:
            add(user, NAMES[name_i], ref + half_days, tz, plan)
    copies = draw(st.lists(st.integers(0, max(len(rows) - 1, 0)), max_size=6))
    rows += [rows[i] for i in copies if rows]
    window = draw(st.sampled_from([1, 3, 7, 14]))
    funnel = draw(st.sampled_from(FUNNELS)).model_copy(update={"window_days": window})
    return rows, funnel


@settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(logs())
def test_sql_matches_brute_force_oracle(case):
    rows, funnel = case
    sql_result = FunnelRun.from_rows(rows, funnel, as_of=FAR_FUTURE).furthest()
    assert sql_result == furthest_steps(rows, funnel)


def test_differential_strategy_reaches_deep_paths_and_boundaries():
    """Guard against a generator that only produces shallow logs."""
    seen_depths: set[int] = set()

    @settings(max_examples=60, deadline=None, derandomize=True, database=None)
    @given(logs())
    def collect(case):
        rows, funnel = case
        seen_depths.update(furthest_steps(rows, funnel).values())

    collect()
    assert {1, 2, 3, 4, 5}.issubset(seen_depths)
