"""Render funnel SQL from a validated definition."""

from __future__ import annotations

from pathlib import Path

import jinja2

from .definitions import Funnel, validate_segments

SQL_FILES = {
    "clean": "00_clean_events.sql",
    "steps": "10_funnel_steps.sql.j2",
    "summary": "20_funnel_summary.sql",
    "mix_rate": "30_mix_rate.sql",
}


def sql_dir() -> Path:
    """Find the SQL directory: next to the package when installed, else the repo root."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "sql", here.parents[1] / "sql"):
        if (candidate / SQL_FILES["clean"]).is_file():
            return candidate
    raise FileNotFoundError("cannot locate the sql/ directory")


def read_sql(stage: str) -> str:
    """Return the text of a static SQL stage (``clean``, ``summary`` or ``mix_rate``)."""
    return (sql_dir() / SQL_FILES[stage]).read_text(encoding="utf-8")


def sql_str(value: object) -> str:
    """Quote a value as a SQL string literal (single quotes doubled)."""
    text = str(value)
    if "\x00" in text:
        raise ValueError("NUL byte in SQL literal")
    return "'" + text.replace("'", "''") + "'"


def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(sql_dir())),
        undefined=jinja2.StrictUndefined,
        autoescape=False,  # noqa: S701 - this renders SQL text, not HTML
        trim_blocks=False,
        keep_trailing_newline=True,
    )
    env.filters["sql_str"] = sql_str
    return env


def render_steps(funnel: Funnel, segments: list[str] | None = None) -> str:
    """Render the step-matching SQL for ``funnel``.

    ``segments`` overrides the funnel's own segment columns (used by ``compare --by``).
    Only an already-validated ``Funnel`` is accepted, so no unchecked text reaches the template.
    """
    if not isinstance(funnel, Funnel):
        raise TypeError("render_steps requires a validated Funnel")
    chosen = validate_segments(list(segments if segments is not None else funnel.segments))
    template = _environment().get_template(SQL_FILES["steps"])
    return template.render(steps=funnel.steps, window_days=funnel.window_days, segments=chosen)
