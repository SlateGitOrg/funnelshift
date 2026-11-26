"""Command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.table import Table

from . import render, report, synth
from .definitions import Funnel, load_funnel, validate_segments
from .run import FunnelRun
from .weeks import expand_weeks

app = typer.Typer(
    help="Funnel conversion in DuckDB SQL, checked against a brute-force oracle.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_DATA = Path("data/events.parquet")

DefOption = Annotated[Path, typer.Option("--def", help="Funnel definition YAML.")]
DataOption = Annotated[Path, typer.Option("--data", help="Events Parquet file.")]


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


def _load(path: Path) -> Funnel:
    try:
        return load_funnel(path)
    except FileNotFoundError:
        raise _fail(f"funnel definition not found: {path}") from None
    except ValidationError as exc:
        raise _fail(f"invalid funnel definition {path}:\n{exc}") from None
    except ValueError as exc:
        raise _fail(f"invalid funnel definition {path}: {exc}") from None


def _weeks(spec: str) -> list[str]:
    try:
        return expand_weeks(spec)
    except ValueError as exc:
        raise _fail(str(exc)) from None


def _segments(text: str) -> list[str]:
    try:
        return validate_segments([part.strip() for part in text.split(",") if part.strip()])
    except ValueError as exc:
        raise _fail(str(exc)) from None


def _run(data: Path, funnel: Funnel, segments: list[str]) -> FunnelRun:
    if not data.is_file():
        raise _fail(f"events file not found: {data} (run `funnelshift generate` first)")
    return FunnelRun.from_parquet(data, funnel, segments)


@app.command()
def generate(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data"),
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
    users: Annotated[int, typer.Option(help="Number of signups.")] = synth.N_USERS,
) -> None:
    """Write synthetic events, ground truth and metadata."""
    data = synth.generate(seed=seed, n_users=users)
    path = synth.write_dataset(data, out)
    typer.echo(f"wrote {path} ({data.meta['events']:,} events, {data.meta['users']:,} users)")
    typer.echo(f"wrote {out / 'truth.parquet'} and {out / 'meta.json'}")
    for key in ("duplicate_events", "pre_signup_events", "tie_users", "late_converters"):
        typer.echo(f"  planted {key}: {data.meta[key]:,}")
    typer.echo(f"  planted double_signups: {data.meta['double_signups']:,}")


@app.command()
def funnel(
    definition: DefOption,
    weeks: Annotated[str, typer.Option(help="Cohort weeks, e.g. 2025-W31..2025-W34.")],
    data: DataOption = DEFAULT_DATA,
    by: Annotated[str, typer.Option(help="Segment column to split the table by.")] = "platform",
) -> None:
    """Users and conversion per step for a range of cohort weeks."""
    spec = _load(definition)
    wanted = _weeks(weeks)
    columns = _segments(by)
    if len(columns) != 1:
        raise _fail("--by takes one column for the funnel table")
    run = _run(data, spec, columns)
    available = run.complete_weeks()
    used = [w for w in wanted if w in available]
    skipped = [w for w in wanted if w not in available]
    if not used:
        raise _fail("none of the requested cohort weeks is complete")
    table = report.period_table(run.summary(), used, columns, columns[0])
    _print_funnel(spec, table, columns[0], used, skipped)


def _print_funnel(spec: Funnel, table, by: str, used: list[str], skipped: list[str]) -> None:
    console = Console(highlight=False)
    groups = [g for g in table[by].unique() if g != "all"] + ["all"]
    title = (
        f"{spec.name}: cohort weeks {used[0]}..{used[-1]}, {spec.window_days}-day window; "
        "conv = cumulative from step 1"
    )
    view = Table(title=title, box=box.SIMPLE_HEAD, title_justify="left")
    view.add_column("step")
    for group in groups:
        view.add_column(f"{group} users", justify="right")
        view.add_column("conv", justify="right")
    for step_no, name in enumerate(spec.step_names, start=1):
        cells = [f"{step_no}. {name}"]
        for group in groups:
            row = table[(table[by] == group) & (table["step_no"] == step_no)].iloc[0]
            cells += [f"{int(row.users_reached):,}", report.pct(row.cum_conv)]
        view.add_row(*cells)
    console.print(view)
    if skipped:
        console.print(f"left out (window not yet elapsed): {', '.join(skipped)}")


@app.command()
def compare(
    definition: DefOption,
    base: Annotated[str, typer.Option(help="Base cohort weeks.")],
    test: Annotated[str, typer.Option(help="Test cohort weeks.")],
    by: Annotated[str, typer.Option(help="Segment columns, comma separated.")] = "platform,source",
    data: DataOption = DEFAULT_DATA,
    out: Annotated[Path, typer.Option(help="Report path.")] = Path("reports/compare.md"),
) -> None:
    """Split the change in conversion between two periods into mix and rate effects."""
    spec = _load(definition)
    base_weeks, test_weeks = _weeks(base), _weeks(test)
    columns = _segments(by)
    run = _run(data, spec, columns)
    try:
        result = run.mix_rate(base_weeks, test_weeks)
    except ValueError as exc:
        raise _fail(str(exc)) from None
    text = report.compare_markdown(spec, result, columns)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    typer.echo(text)
    typer.echo(f"wrote {out}")


@app.command()
def explain(
    definition: DefOption,
    all_stages: Annotated[
        bool, typer.Option("--all", help="Also print the other SQL stages.")
    ] = False,
) -> None:
    """Print the SQL rendered for a funnel definition."""
    spec = _load(definition)
    if all_stages:
        typer.echo(f"-- sql/{render.SQL_FILES['clean']}\n{render.read_sql('clean')}")
    typer.echo(f"-- sql/{render.SQL_FILES['steps']} rendered for {spec.name}")
    typer.echo(render.render_steps(spec))
    if all_stages:
        typer.echo(f"-- sql/{render.SQL_FILES['summary']}\n{render.read_sql('summary')}")
        typer.echo(f"-- sql/{render.SQL_FILES['mix_rate']}\n{render.read_sql('mix_rate')}")
