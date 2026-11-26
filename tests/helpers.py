"""Shared builders for hand-written event fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

T0 = datetime(2025, 6, 2, 9, 0, 0, tzinfo=UTC)  # a Monday, 2025-W23


def ev(
    event_id: str,
    user: str,
    name: str,
    *,
    days: float = 0,
    hours: float = 0,
    minutes: float = 0,
    seconds: float = 0,
    tz_minutes: int = 0,
    platform: str | None = "web",
    source: str | None = "organic",
    plan: str | None = None,
    page: str | None = None,
    base: datetime = T0,
) -> tuple:
    """A raw event tuple; the instant is `base` plus the offsets, written in a local zone."""
    instant = base + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    local = instant.astimezone(timezone(timedelta(minutes=tz_minutes)))
    return (event_id, user, name, local.isoformat(), platform, source, plan, page)
