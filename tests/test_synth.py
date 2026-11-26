from __future__ import annotations

import math

import pandas as pd
import pytest

from funnelshift import synth
from funnelshift.definitions import SIGNUP_EVENT
from funnelshift.weeks import expand_weeks


def test_scale_matches_the_documented_dataset(data):
    assert data.meta["users"] == 60_000
    assert data.truth["user_id"].nunique() == 60_000
    assert 345_000 <= len(data.events) <= 380_000
    assert set(data.events["event_name"]) == set(synth.STEP_EVENTS) | set(synth.NOISE_EVENTS)
    assert set(data.truth["platform"]) == set(synth.PLATFORMS)
    assert set(data.truth["source"]) == set(synth.SOURCES)


def test_twenty_iso_weeks_from_monday_2_june(data):
    weeks = sorted(data.truth["cohort_week"].unique())
    assert weeks == expand_weeks("2025-W23..2025-W42")
    counts = data.truth.groupby("cohort_week").size().to_dict()
    assert list(counts.values()) == synth.weekly_totals()


def test_planted_counts_are_exact(data):
    meta = data.meta
    n = 60_000
    assert meta["double_signups"] == round(0.005 * n)
    assert meta["tie_users"] == round(0.02 * n)
    assert meta["late_converters"] == round(0.04 * n)
    assert meta["pre_signup_events"] == int(data.truth["pre_signup_events"].sum())
    assert (data.truth["pre_signup_events"] > 0).sum() == round(0.01 * n)
    base_events = meta["events"] - meta["duplicate_events"]
    assert meta["duplicate_events"] == round(0.015 * base_events)
    dupes = data.events["event_id"].duplicated().sum()
    assert dupes == meta["duplicate_events"]


def test_duplicates_are_exact_copies(data):
    dup_ids = data.events.loc[data.events["event_id"].duplicated(), "event_id"]
    rows = data.events[data.events["event_id"].isin(dup_ids)].drop_duplicates()
    assert len(rows) == dup_ids.nunique()


def test_generation_is_deterministic_and_seed_sensitive():
    a = synth.generate(seed=3, n_users=1_200)
    b = synth.generate(seed=3, n_users=1_200)
    c = synth.generate(seed=4, n_users=1_200)
    pd.testing.assert_frame_equal(a.events, b.events)
    pd.testing.assert_frame_equal(a.truth, b.truth)
    assert not a.events.equals(c.events)


def test_raw_timestamps_carry_local_offsets(data):
    offsets = data.events["event_time"].str[-6:].value_counts()
    assert len(offsets) >= 5
    assert "+00:00" in offsets.index and "+05:30" in offsets.index and "-05:00" in offsets.index


def test_campaign_triples_the_paid_social_android_signups(data):
    t = data.truth
    early = t[t["cohort_week"] < "2025-W35"]
    late = t[t["cohort_week"] >= "2025-W35"]

    def share(frame):
        return ((frame["platform"] == "android") & (frame["source"] == "paid_social")).mean()

    p = synth.BASE_SHARE[synth.CAMPAIGN_CELL]
    assert share(early) == pytest.approx(p, abs=0.008)
    assert share(late) == pytest.approx(3 * p / (1 + 2 * p), abs=0.01)


def test_expected_mix_effect_is_about_minus_2_9_points():
    base, test = expand_weeks("2025-W31..2025-W34"), expand_weeks("2025-W35..2025-W38")
    mix = synth.expected_mix_effect(base, test)
    assert mix * 100 == pytest.approx(-2.9, abs=0.2)
    # the same figure from week numbers, and nothing changes between two pre-campaign periods
    assert synth.expected_mix_effect([9, 10, 11, 12], [13, 14, 15, 16]) == pytest.approx(mix)
    assert synth.expected_mix_effect([1, 2, 3, 4], [5, 6, 7, 8]) == pytest.approx(0.0, abs=1e-12)


def test_campaign_cell_converts_at_about_5_percent_and_the_rest_near_18():
    c = synth.expected_cell_conversion
    assert c(synth.CAMPAIGN_CELL, 1) == pytest.approx(0.05, abs=0.005)
    rest = [cell for cell in synth.CELLS if cell != synth.CAMPAIGN_CELL]
    probs = synth.cell_probabilities(1)
    average = sum(probs[x] * c(x, 1) for x in rest) / sum(probs[x] for x in rest)
    assert average == pytest.approx(0.18, abs=0.02)


def test_observed_conversion_tracks_the_analytic_value_per_cell(data):
    t = data.truth[data.truth["cohort_week"] < "2025-W35"]  # before the release too
    for (platform, source), group in t.groupby(["platform", "source"]):
        expected = synth.expected_cell_conversion((platform, source), 1)
        observed = (group["depth_14"] == 5).mean()
        se = math.sqrt(expected * (1 - expected) / len(group))
        assert abs(observed - expected) < 4 * se, (platform, source)


def test_android_release_changes_only_the_project_created_rate():
    before = synth.cell_rates(("android", "organic"), 16)
    after = synth.cell_rates(("android", "organic"), 17)
    assert before[2] == 0.62 and after[2] == 0.45
    assert before[:2] + before[3:] == after[:2] + after[3:]
    assert synth.cell_rates(("web", "organic"), 16) == synth.cell_rates(("web", "organic"), 17)


def test_week_number_rejects_weeks_outside_the_data():
    assert synth.week_number("2025-W23") == 1 and synth.week_number("2025-W42") == 20
    with pytest.raises(ValueError, match="outside the generated weeks"):
        synth.week_number("2025-W43")


def test_signup_events_exist_for_every_user(data):
    signups = data.events[data.events["event_name"] == SIGNUP_EVENT]
    assert signups["user_id"].nunique() == 60_000
