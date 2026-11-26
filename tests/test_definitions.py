from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from funnelshift import render, run
from funnelshift.definitions import SEGMENT_COLUMNS, load_funnel, parse_funnel

BASE = """
name: f
steps:
  - event: signup
  - event: workspace_created
"""


def test_shipped_definitions_load(trial):
    assert trial.step_names == [
        "signup",
        "workspace_created",
        "teammate_invited",
        "project_created",
        "subscribed",
    ]
    assert trial.window_days == 14
    assert trial.segments == ["platform", "source"]
    team = load_funnel(Path(__file__).parents[1] / "funnels" / "trial_team_plan.yaml")
    assert team.steps[-1].where == {"plan": "team"}


def test_window_defaults_to_14_days():
    assert parse_funnel(BASE).window_days == 14
    assert parse_funnel(BASE).segments == list(SEGMENT_COLUMNS)


def test_repeated_event_names_rejected():
    text = BASE + "  - event: signup\n"
    with pytest.raises(ValidationError, match="may not repeat"):
        parse_funnel(text)


def test_unknown_property_rejected():
    text = BASE + "    where:\n      colour: red\n"
    with pytest.raises(ValidationError, match="unknown event properties"):
        parse_funnel(text)


@pytest.mark.parametrize("bad", ["Signup", "1signup", "sign-up", "sign up", "", "a;b"])
def test_identifiers_must_match_pattern(bad):
    text = BASE.replace("workspace_created", repr(bad))
    with pytest.raises(ValidationError):
        parse_funnel(text)


def test_bad_funnel_name_rejected():
    with pytest.raises(ValidationError):
        parse_funnel(BASE.replace("name: f", "name: 'x y'"))


def test_unknown_segment_rejected():
    with pytest.raises(ValidationError, match="unknown segment column"):
        parse_funnel(BASE + "segments: [country]\n")


def test_single_step_and_bad_window_rejected():
    with pytest.raises(ValidationError):
        parse_funnel("name: f\nsteps:\n  - event: signup\n")
    with pytest.raises(ValidationError):
        parse_funnel(BASE + "window_days: 0\n")


def test_filter_values_are_restricted():
    text = BASE + "    where:\n      plan: \"x' OR '1'='1\"\n"
    with pytest.raises(ValidationError):
        parse_funnel(text)


def test_injection_in_event_name_rejected_and_no_sql_rendered(monkeypatch):
    calls: list[object] = []

    def spy(*args, **kwargs):
        calls.append(args)
        return ""

    monkeypatch.setattr(render, "render_steps", spy)
    monkeypatch.setattr(run, "render_steps", spy)
    text = BASE.replace("workspace_created", '"x\'); DROP TABLE events; --"')
    with pytest.raises(ValidationError):
        parse_funnel(text)
    assert calls == []


def test_render_refuses_unvalidated_input():
    with pytest.raises(TypeError):
        render.render_steps({"name": "f"})  # type: ignore[arg-type]
