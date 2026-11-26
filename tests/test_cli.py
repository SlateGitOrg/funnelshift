from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from funnelshift import render, run
from funnelshift.cli import app

ROOT = Path(__file__).resolve().parents[1]
TRIAL = str(ROOT / "funnels" / "trial.yaml")
runner = CliRunner()
ENV = {"COLUMNS": "140"}


def invoke(*args: str):
    return runner.invoke(app, list(args), env=ENV)


def test_funnel_prints_users_and_conversion_per_step_by_platform(events_path, run14):
    result = invoke(
        "funnel", "--def", TRIAL, "--weeks", "2025-W31..2025-W34", "--data", str(events_path)
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "cohort weeks 2025-W31..2025-W34" in out
    for step in ("1. signup", "5. subscribed"):
        assert step in out
    for platform in ("web", "ios", "android", "all"):
        assert f"{platform} users" in out
    summary = run14.summary()
    window = summary[summary["cohort_week"].between("2025-W31", "2025-W34")]
    total = int(window[window["step_no"] == 1]["users_reached"].sum())
    assert f"{total:,}" in out
    assert "100.00%" in out


def test_funnel_reports_weeks_whose_window_has_not_elapsed(events_path):
    result = invoke(
        "funnel", "--def", TRIAL, "--weeks", "2025-W41..2025-W44", "--data", str(events_path)
    )
    assert result.exit_code == 0, result.output
    assert "left out (window not yet elapsed): 2025-W43, 2025-W44" in result.output


def test_compare_writes_the_report_with_a_mix_driven_verdict(events_path, tmp_path):
    out = tmp_path / "reports" / "compare.md"
    result = invoke(
        "compare", "--def", TRIAL,
        "--base", "2025-W31..2025-W34", "--test", "2025-W35..2025-W38",
        "--by", "platform,source", "--data", str(events_path), "--out", str(out),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "Verdict: **mix_driven**" in text
    assert "| Total change |" in text and "| Mix (who signs up) |" in text
    assert "android / paid_social" in text
    top = text.split("## Top 5 step x segment cells")[1]
    assert len([line for line in top.splitlines() if line.startswith("| ") and "/" in line]) == 5


def test_compare_names_the_android_step_when_the_release_is_the_cause(events_path, tmp_path):
    out = tmp_path / "compare.md"
    result = invoke(
        "compare", "--def", TRIAL,
        "--base", "2025-W35..2025-W38", "--test", "2025-W39..2025-W42",
        "--data", str(events_path), "--out", str(out),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "Verdict: **rate_driven**" in text
    first_row = text.split(
        "Conditional rate base > test | Rate contribution (pp) |\n|---|---|---|---|\n"
    )[1]
    assert first_row.startswith("| teammate_invited > project_created | android /")


def test_explain_prints_the_rendered_sql():
    result = invoke("explain", "--def", TRIAL)
    assert result.exit_code == 0
    assert "CREATE OR REPLACE TABLE funnel_steps" in result.output
    assert "INTERVAL '14 days'" in result.output
    assert "clean_events" in result.output
    full = invoke("explain", "--def", TRIAL, "--all")
    assert "CREATE OR REPLACE TABLE mix_rate_total" in full.output
    assert "CREATE OR REPLACE TABLE funnel_summary" in full.output
    assert "CREATE OR REPLACE TABLE clean_events" in full.output


@pytest.mark.parametrize("command", ["funnel", "compare", "explain"])
def test_injected_event_name_is_rejected_before_any_sql_is_rendered(
    command, tmp_path, monkeypatch, events_path
):
    calls: list[object] = []

    def spy(*args, **kwargs):
        calls.append(args)
        return "SELECT 1"

    monkeypatch.setattr(render, "render_steps", spy)
    monkeypatch.setattr(run, "render_steps", spy)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        'name: evil\nsteps:\n  - event: signup\n  - event: "x\'); DROP TABLE events; --"\n',
        encoding="utf-8",
    )
    args = {
        "funnel": ["funnel", "--def", str(bad), "--weeks", "2025-W31", "--data", str(events_path)],
        "compare": ["compare", "--def", str(bad), "--base", "2025-W31", "--test", "2025-W32",
                    "--data", str(events_path), "--out", str(tmp_path / "r.md")],
        "explain": ["explain", "--def", str(bad)],
    }[command]  # fmt: skip
    result = invoke(*args)
    assert result.exit_code == 2
    assert "invalid funnel definition" in result.output
    assert calls == []
    assert not (tmp_path / "r.md").exists()


def test_bad_arguments_exit_with_a_clear_error(events_path):
    result = invoke("funnel", "--def", TRIAL, "--weeks", "2025-31", "--data", str(events_path))
    assert result.exit_code == 2 and "not an ISO week label" in result.output
    result = invoke("funnel", "--def", TRIAL, "--weeks", "2025-W31", "--data", "missing.parquet")
    assert result.exit_code == 2 and "run `funnelshift generate` first" in result.output
    result = invoke(
        "compare", "--def", TRIAL, "--base", "2025-W31", "--test", "2025-W32",
        "--by", "country", "--data", str(events_path),
    )  # fmt: skip
    assert result.exit_code == 2 and "unknown segment column" in result.output


def test_generate_writes_the_dataset(tmp_path):
    out = tmp_path / "d"
    result = invoke("generate", "--out", str(out), "--seed", "5", "--users", "900")
    assert result.exit_code == 0, result.output
    assert (out / "events.parquet").is_file() and (out / "truth.parquet").is_file()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["users"] == 900 and meta["seed"] == 5
    assert "planted duplicate_events" in result.output
