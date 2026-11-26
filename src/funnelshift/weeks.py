"""ISO week labels (``2025-W31``) and inclusive week ranges (``2025-W31..2025-W34``)."""

from __future__ import annotations

import re
from datetime import date, timedelta

_WEEK = re.compile(r"^(\d{4})-W(\d{2})$")


def parse_week(label: str) -> date:
    """Return the Monday of an ISO week label."""
    match = _WEEK.match(label.strip())
    if match is None:
        raise ValueError(f"not an ISO week label like 2025-W31: {label!r}")
    year, week = int(match.group(1)), int(match.group(2))
    try:
        return date.fromisocalendar(year, week, 1)
    except ValueError as exc:
        raise ValueError(f"invalid ISO week: {label!r}") from exc


def week_label(day: date) -> str:
    iso = day.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def expand_weeks(spec: str) -> list[str]:
    """Expand ``A..B`` (inclusive) or a single week into a list of ISO week labels."""
    if ".." in spec:
        first, last = (part.strip() for part in spec.split("..", 1))
    else:
        first = last = spec.strip()
    start, end = parse_week(first), parse_week(last)
    if end < start:
        raise ValueError(f"week range runs backwards: {spec!r}")
    labels = []
    day = start
    while day <= end:
        labels.append(week_label(day))
        day += timedelta(days=7)
    return labels
