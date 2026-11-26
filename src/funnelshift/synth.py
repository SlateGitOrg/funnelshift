"""Seeded synthetic product events for a project-management SaaS trial funnel.

The generator draws every user's path from per-cell conditional step rates, so each cell's
expected end-to-end conversion is known analytically, and it records ground truth for every
user. Injected patterns the analysis has to recover:

* from week 13 (2025-W35) a paid_social campaign triples paid_social x android signups; that
  cell's step rates do not change and are low, so total conversion falls by a pure mix effect
  (`expected_mix_effect` gives it exactly);
* from week 17 (2025-W39, app release 5.2) android's teammate_invited -> project_created
  conditional rate falls from 0.62 to 0.45 in every android cell;
* messy data: 1.5% duplicate events, 2% of users with two consecutive steps at one timestamp,
  4% of users subscribing 15-20 days after signup, 1% with events before their signup and
  0.5% who sign up twice.

Weeks are numbered 1..20 from Monday 2 June 2025 (ISO 2025-W23 to 2025-W42), all in UTC.
Raw timestamps carry the user's local UTC offset, as a real client feed would.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .run import RAW_COLUMNS, connect
from .weeks import week_label

FIRST_MONDAY = date(2025, 6, 2)
N_WEEKS = 20
N_USERS = 60_000
CAMPAIGN_WEEK = 13  # first week of the paid_social x android campaign (2025-W35)
RELEASE_WEEK = 17  # first week of android app release 5.2 (2025-W39)
CAMPAIGN_MULTIPLIER = 3.0
WINDOW_DAYS = 14
PATH_CAP_DAYS = 13.0  # in-window paths finish before day 13, so the 14-day window is clean
TAIL_DAYS = 21  # events continue this long after the last signup
LATE_DAYS = (15.0, 20.0)

PLATFORMS = ("web", "ios", "android")
SOURCES = ("organic", "paid_social", "referral", "partner")
STEP_EVENTS = ("signup", "workspace_created", "teammate_invited", "project_created", "subscribed")
NOISE_EVENTS = ("page_view", "help_opened")
PAGES = ("dashboard", "board", "settings", "pricing")
PLANS = ("team", "business")
PLAN_TEAM_SHARE = 0.7

#: Share of baseline signups per (platform, source) cell. Sums to 1.
BASE_SHARE: dict[tuple[str, str], float] = {
    ("web", "organic"): 0.16,
    ("web", "paid_social"): 0.06,
    ("web", "referral"): 0.07,
    ("web", "partner"): 0.05,
    ("ios", "organic"): 0.14,
    ("ios", "paid_social"): 0.06,
    ("ios", "referral"): 0.06,
    ("ios", "partner"): 0.04,
    ("android", "organic"): 0.09,
    ("android", "paid_social"): 0.17,
    ("android", "referral"): 0.05,
    ("android", "partner"): 0.05,
}
CELLS: tuple[tuple[str, str], ...] = tuple(BASE_SHARE)
CAMPAIGN_CELL = ("android", "paid_social")

#: Conditional rates (workspace, invite, project, subscribe) by source; android overrides
#: the project rate; the campaign cell has its own low rates.
SOURCE_RATES: dict[str, tuple[float, float, float, float]] = {
    "organic": (0.86, 0.58, 0.64, 0.62),
    "paid_social": (0.80, 0.50, 0.60, 0.54),
    "referral": (0.90, 0.66, 0.66, 0.66),
    "partner": (0.84, 0.55, 0.62, 0.58),
}
PLATFORM_INVITE_FACTOR = {"web": 1.0, "ios": 0.95, "android": 0.90}
CAMPAIGN_RATES = (0.60, 0.30, 0.62, 0.45)
ANDROID_PROJECT_RATE = 0.62
ANDROID_PROJECT_RATE_AFTER_RELEASE = 0.45

#: Delay before each step: (median seconds, lognormal sigma).
DELAYS = (
    (6 * 60.0, 1.0),
    (5 * 3600.0, 1.1),
    (20 * 3600.0, 1.0),
    (2.5 * 86400.0, 0.8),
)

LATE_SHARE = 0.04  # of all users; drawn from those who reach project_created and do not subscribe
DUPLICATE_SHARE = 0.015
TIE_SHARE = 0.02
PRE_SIGNUP_SHARE = 0.01
DOUBLE_SIGNUP_SHARE = 0.005
DECOY_SHARE = 0.04
REPEAT_SHARE = {2: 0.15, 3: 0.35, 4: 0.30}
PAGE_VIEWS_PER_USER = 2.15
HELP_PER_USER = 0.6

#: Local UTC offsets (minutes) and their shares.
TZ_OFFSETS = ((0, 0.30), (-300, 0.22), (-480, 0.12), (60, 0.18), (330, 0.10), (540, 0.08))

_US = 10**6
_DAY_US = 86_400 * _US


def _epoch_us(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * _US


def week_number(label: str) -> int:
    """1-based week number (1 = 2025-W23) of an ISO week label."""
    from .weeks import parse_week

    n = (parse_week(label) - FIRST_MONDAY).days // 7 + 1
    if not 1 <= n <= N_WEEKS:
        raise ValueError(f"{label} is outside the generated weeks (2025-W23..2025-W42)")
    return n


def cell_rates(cell: tuple[str, str], week: int) -> tuple[float, float, float, float]:
    """Conditional step rates for a signup cell in a given week (1-based)."""
    platform, source = cell
    if cell == CAMPAIGN_CELL:
        rates = list(CAMPAIGN_RATES)
    else:
        rates = list(SOURCE_RATES[source])
        rates[1] *= PLATFORM_INVITE_FACTOR[platform]
    if platform == "android":
        rates[2] = (
            ANDROID_PROJECT_RATE if week < RELEASE_WEEK else ANDROID_PROJECT_RATE_AFTER_RELEASE
        )
    return (rates[0], rates[1], rates[2], rates[3])


def expected_cell_conversion(cell: tuple[str, str], week: int) -> float:
    """Analytic end-to-end conversion inside the 14-day window for a cell and week."""
    r = cell_rates(cell, week)
    return r[0] * r[1] * r[2] * r[3]


def cell_probabilities(week: int) -> dict[tuple[str, str], float]:
    """Probability that a signup in `week` belongs to each cell (campaign triples one cell)."""
    raw = dict(BASE_SHARE)
    if week >= CAMPAIGN_WEEK:
        raw[CAMPAIGN_CELL] *= CAMPAIGN_MULTIPLIER
    total = sum(raw.values())
    return {cell: v / total for cell, v in raw.items()}


def weekly_totals(n_users: int = N_USERS) -> list[int]:
    """Signups per week: the campaign adds traffic, and the totals sum to `n_users` exactly."""
    intensity = [
        1.0 + (CAMPAIGN_MULTIPLIER - 1.0) * BASE_SHARE[CAMPAIGN_CELL] * (w >= CAMPAIGN_WEEK)
        for w in range(1, N_WEEKS + 1)
    ]
    scale = n_users / sum(intensity)
    totals = [int(i * scale) for i in intensity]
    for k in range(n_users - sum(totals)):  # largest-remainder style top-up, from the last week
        totals[-1 - (k % N_WEEKS)] += 1
    return totals


def _period_weeks(weeks: list[str] | list[int]) -> list[int]:
    return [w if isinstance(w, int) else week_number(w) for w in weeks]


def expected_cell_weights_and_rates(
    weeks: list[str] | list[int], n_users: int = N_USERS
) -> dict[tuple[str, str], tuple[float, float]]:
    """Expected (entry weight, end-to-end conversion) per cell over a set of weeks."""
    numbers = _period_weeks(weeks)
    totals = weekly_totals(n_users)
    users = {cell: 0.0 for cell in CELLS}
    converters = {cell: 0.0 for cell in CELLS}
    for w in numbers:
        probs = cell_probabilities(w)
        for cell in CELLS:
            n = totals[w - 1] * probs[cell]
            users[cell] += n
            converters[cell] += n * expected_cell_conversion(cell, w)
    grand = sum(users.values())
    return {c: (users[c] / grand, converters[c] / users[c]) for c in CELLS}


def expected_mix_effect(
    base_weeks: list[str] | list[int], test_weeks: list[str] | list[int], n_users: int = N_USERS
) -> float:
    """Mix effect (a fraction, not pp) the generator is built to produce between two periods.

    Uses expected cell weights and the analytic cell conversions with the same symmetric
    decomposition the SQL applies: sum((w1 - w0) * (c0 + c1) / 2).
    """
    base = expected_cell_weights_and_rates(base_weeks, n_users)
    test = expected_cell_weights_and_rates(test_weeks, n_users)
    return sum((test[c][0] - base[c][0]) * (base[c][1] + test[c][1]) / 2 for c in CELLS)


def expected_conversion(weeks: list[str] | list[int], n_users: int = N_USERS) -> float:
    """Expected total end-to-end conversion (14-day window) over a set of weeks."""
    cells = expected_cell_weights_and_rates(weeks, n_users)
    return sum(w * c for w, c in cells.values())


@dataclass
class SynthData:
    """Generated events plus the ground truth needed to check an analysis."""

    events: pd.DataFrame  # columns RAW_COLUMNS
    truth: pd.DataFrame  # one row per user
    meta: dict[str, object] = field(default_factory=dict)

    def late_converters(self, weeks: list[str]) -> int:
        t = self.truth
        return int(t.loc[t["cohort_week"].isin(weeks), "late_converter"].sum())


def _truncated_lognormal(
    rng: np.random.Generator, median: float, sigma: float, cum_us: np.ndarray
) -> np.ndarray:
    """Delays in microseconds such that `cum + delay` stays within the path cap."""
    cap_us = PATH_CAP_DAYS * _DAY_US
    n = len(cum_us)
    delay = rng.lognormal(np.log(median), sigma, n) * _US
    for _ in range(60):
        bad = cum_us + delay > cap_us
        if not bad.any():
            break
        delay[bad] = rng.lognormal(np.log(median), sigma, int(bad.sum())) * _US
    bad = cum_us + delay > cap_us
    if bad.any():
        delay[bad] = rng.uniform(0.1, 0.9, int(bad.sum())) * (cap_us - cum_us[bad])
    return np.maximum(np.round(delay), 1.0)


def _pick(rng: np.random.Generator, pool: np.ndarray, count: float) -> np.ndarray:
    """Exactly round(count) distinct members of `pool`, so planted counts are exact."""
    return rng.choice(pool, size=min(round(count), len(pool)), replace=False)


def generate(seed: int = 7, n_users: int = N_USERS) -> SynthData:
    """Generate the event log and ground truth. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    totals = weekly_totals(n_users)

    # -- who signs up, when ----------------------------------------------------------------
    week_idx_parts, cell_idx_parts = [], []
    for w in range(1, N_WEEKS + 1):
        probs = cell_probabilities(w)
        counts = rng.multinomial(totals[w - 1], [probs[c] for c in CELLS])
        week_idx_parts.append(np.full(totals[w - 1], w - 1))
        cell_idx_parts.append(np.repeat(np.arange(len(CELLS)), counts))
    week_idx = np.concatenate(week_idx_parts)
    cell_idx = np.concatenate(cell_idx_parts)
    n = len(week_idx)
    first_us = _epoch_us(FIRST_MONDAY)
    signup_us = first_us + week_idx * 7 * _DAY_US + (rng.random(n) * 7 * _DAY_US).astype(np.int64)
    user_ids = np.array([f"u{i:06d}" for i in rng.permutation(n)])
    tz_minutes = rng.choice(
        [m for m, _ in TZ_OFFSETS], size=n, p=[p for _, p in TZ_OFFSETS]
    ).astype(np.int64)

    platforms = np.array([c[0] for c in CELLS])[cell_idx]
    sources = np.array([c[1] for c in CELLS])[cell_idx]

    # -- how far each user goes ------------------------------------------------------------
    rates = np.empty((n, 4))
    for w in range(1, N_WEEKS + 1):
        for ci, cell in enumerate(CELLS):
            mask = (week_idx == w - 1) & (cell_idx == ci)
            rates[mask] = cell_rates(cell, w)
    passed = rng.random((n, 4)) < rates
    reach = np.cumprod(passed, axis=1).astype(bool)  # reach[:, t] -> step t + 2 reached
    depth14 = 1 + reach.sum(axis=1)

    # offsets of each step from signup, microseconds (valid where reached)
    offset = np.zeros((n, 5), dtype=np.int64)
    cum = np.zeros(n)
    for t, (median, sigma) in enumerate(DELAYS):
        cum = cum + _truncated_lognormal(rng, median, sigma, cum)
        offset[:, t + 1] = cum.astype(np.int64)

    # two consecutive steps logged at the same instant
    tie_step = np.zeros(n, dtype=np.int64)
    tie_users = _pick(rng, np.where(depth14 >= 2)[0], TIE_SHARE * n)
    tie_step[tie_users] = rng.integers(2, depth14[tie_users] + 1)
    for j in range(2, 6):
        rows = tie_step == j
        offset[rows, j - 1] = offset[rows, j - 2]

    # subscribed 15-20 days after signup
    late = np.zeros(n, dtype=bool)
    late[_pick(rng, np.where(depth14 == 4)[0], LATE_SHARE * n)] = True
    late_offset = (rng.uniform(*LATE_DAYS, n) * _DAY_US).astype(np.int64)
    depth21 = np.where(late, 5, depth14)

    double_signup = np.zeros(n, dtype=bool)
    double_signup[_pick(rng, np.arange(n), DOUBLE_SIGNUP_SHARE * n)] = True
    pre_count = np.zeros(n, dtype=np.int64)
    pre_users = _pick(rng, np.arange(n), PRE_SIGNUP_SHARE * n)
    pre_count[pre_users] = rng.integers(1, 4, len(pre_users))

    # -- events ----------------------------------------------------------------------------
    parts: list[tuple[np.ndarray, str | np.ndarray, np.ndarray]] = []  # (user idx, name, ts)

    def add(users: np.ndarray, name: str | np.ndarray, ts: np.ndarray) -> None:
        if len(users):
            parts.append((users, name, ts))

    all_users = np.arange(n)
    add(all_users, "signup", signup_us)
    for k in range(2, 6):
        who = np.where(depth14 >= k)[0]
        add(who, STEP_EVENTS[k - 1], signup_us[who] + offset[who, k - 1])
    late_who = np.where(late)[0]
    add(late_who, "subscribed", signup_us[late_who] + late_offset[late_who])

    dbl = np.where(double_signup)[0]
    add(dbl, "signup", signup_us[dbl] + (rng.uniform(1, 10, len(dbl)) * _DAY_US).astype(np.int64))

    tail_us = TAIL_DAYS * _DAY_US
    for k, share in REPEAT_SHARE.items():
        who = np.where((depth14 >= k) & (rng.random(n) < share))[0]
        for extra in (0, 1):  # a first repeat, then sometimes a second
            again = who[rng.random(len(who)) < 0.6] if extra else who
            gap = (rng.exponential(1.0, len(again)) * _DAY_US).astype(np.int64) + _US
            ts = np.minimum(
                signup_us[again] + offset[again, k - 1] + gap, signup_us[again] + tail_us - 1
            )
            add(again, STEP_EVENTS[k - 1], ts)

    # decoys: a later step logged before the step that precedes it; never a valid path
    for j in (3, 4, 5):
        gap = offset[:, j - 2] - offset[:, j - 3]
        who = np.where((depth14 == j - 1) & (gap > 0) & (rng.random(n) < DECOY_SHARE))[0]
        frac = rng.random(len(who))
        ts = signup_us[who] + offset[who, j - 3] + np.floor(frac * gap[who]).astype(np.int64)
        add(who, STEP_EVENTS[j - 1], ts)

    # noise events within 21 days of signup
    for name, mean in (("page_view", PAGE_VIEWS_PER_USER), ("help_opened", HELP_PER_USER)):
        counts = rng.poisson(mean, n)
        who = np.repeat(all_users, counts)
        delay = (rng.exponential(1.5, len(who)) * _DAY_US).astype(np.int64) + _US
        ts = signup_us[who] + np.minimum(delay, tail_us - 1)
        add(who, name, ts)

    # events before the user's signup
    pre_who = np.repeat(all_users, pre_count)
    pre_names = rng.choice(np.array(STEP_EVENTS[1:] + NOISE_EVENTS), size=len(pre_who))
    pre_ts = signup_us[pre_who] - (rng.uniform(1 / 1440, 5, len(pre_who)) * _DAY_US).astype(
        np.int64
    )
    add(pre_who, pre_names, pre_ts)

    user_col = np.concatenate([p[0] for p in parts])
    name_col = np.concatenate(
        [
            np.full(len(p[0]), p[1], dtype=object) if isinstance(p[1], str) else p[1].astype(object)
            for p in parts
        ]
    )
    ts_col = np.concatenate([p[2] for p in parts])

    order = np.argsort(ts_col, kind="stable")
    user_col, name_col, ts_col = user_col[order], name_col[order], ts_col[order]
    n_base = len(ts_col)
    event_ids = np.array([f"ev{i:08d}" for i in range(n_base)])
    base_plan = np.where(
        (name_col == "subscribed") & (rng.random(n_base) < PLAN_TEAM_SHARE), PLANS[0], PLANS[1]
    )
    base_plan = np.where(name_col == "subscribed", base_plan, None)
    base_page = np.where(name_col == "page_view", rng.choice(np.array(PAGES), size=n_base), None)

    # exact duplicates of randomly chosen events (same event_id)
    n_dup = round(DUPLICATE_SHARE * n_base)
    dup_pick = rng.choice(n_base, size=n_dup, replace=False)
    pick = np.concatenate([np.arange(n_base), dup_pick])
    pick = pick[np.argsort(ts_col[pick], kind="stable")]

    user_col, name_col, ts_col = user_col[pick], name_col[pick], ts_col[pick]
    event_ids, plan_col, page_col = event_ids[pick], base_plan[pick], base_page[pick]
    n_events = len(pick)

    local = pd.to_datetime(ts_col + tz_minutes[user_col] * 60 * _US, unit="us")
    sign = np.where(tz_minutes[user_col] < 0, "-", "+")
    abs_min = np.abs(tz_minutes[user_col])
    offset_text = pd.Series(sign).str.cat(
        pd.Series(np.char.zfill((abs_min // 60).astype(str), 2))
        + ":"
        + pd.Series(np.char.zfill((abs_min % 60).astype(str), 2))
    )
    event_time = pd.Series(local).dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + offset_text

    events = pd.DataFrame(
        {
            "event_id": event_ids,
            "user_id": user_ids[user_col],
            "event_name": name_col.astype(str),
            "event_time": event_time.to_numpy(),
            "platform": platforms[user_col],
            "source": sources[user_col],
            "plan": plan_col,
            "page": page_col,
        }
    )[list(RAW_COLUMNS)]

    truth = pd.DataFrame(
        {
            "user_id": user_ids,
            "cohort_week": [
                week_label(FIRST_MONDAY + timedelta(days=7 * int(w))) for w in week_idx
            ],
            "platform": platforms,
            "source": sources,
            "depth_14": depth14,
            "depth_21": depth21,
            "late_converter": late,
            "tie_step": tie_step,
            "double_signup": double_signup,
            "pre_signup_events": pre_count,
        }
    )
    meta: dict[str, object] = {
        "seed": seed,
        "users": int(n),
        "events": int(n_events),
        "duplicate_events": int(n_dup),
        "pre_signup_events": int(pre_count.sum()),
        "tie_users": int((tie_step > 0).sum()),
        "late_converters": int(late.sum()),
        "double_signups": int(double_signup.sum()),
        "first_week": week_label(FIRST_MONDAY),
        "last_week": week_label(FIRST_MONDAY + timedelta(days=7 * (N_WEEKS - 1))),
    }
    return SynthData(events=events, truth=truth, meta=meta)


def write_dataset(data: SynthData, directory: str | Path) -> Path:
    """Write events.parquet (via DuckDB), truth.parquet and meta.json; return the events path."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    con = connect()
    events_path = out / "events.parquet"
    for frame, name in ((data.events, "events"), (data.truth, "truth")):
        con.register("frame", frame)
        con.execute(
            f"COPY (SELECT * FROM frame) TO '{out / (name + '.parquet')}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.unregister("frame")
    (out / "meta.json").write_text(json.dumps(data.meta, indent=2) + "\n", encoding="utf-8")
    con.close()
    return events_path
