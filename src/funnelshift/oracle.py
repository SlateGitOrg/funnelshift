"""Brute-force reference for the furthest funnel step per user.

It shares no logic with the SQL. It reads raw event tuples, applies the cleaning rules in plain
Python, then tries every ordered subsequence of a user's qualifying events, deepest first, and
accepts the first one that is a valid funnel path. That is exponential in the number of events
per user, which is fine for the small logs it is used on and is the point: nothing is assumed
about which match to take at each step.

A subsequence of ``k`` events is a valid path when

* the events can be labelled with steps 1..k, one each (event names are unique per step, so
  the labelling is forced), and each event satisfies its step's property filters;
* timestamps do not decrease from step to step (equal timestamps are in order);
* the step 1 event is one of the user's earliest qualifying step 1 events, which fixes the
  window start; and
* every event is at most ``window_days`` after the step 1 event.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import combinations

from .definitions import SIGNUP_EVENT, Funnel
from .run import RAW_COLUMNS


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    name: str
    ts: datetime  # timezone-aware UTC
    props: tuple[tuple[str, str | None], ...]

    def prop(self, key: str) -> str | None:
        return dict(self.props).get(key)


def parse_event(row: Sequence[object]) -> Event:
    """Build an ``Event`` from a raw tuple in ``RAW_COLUMNS`` order."""
    record = dict(zip(RAW_COLUMNS, row, strict=True))
    text = str(record["event_time"])
    ts = datetime.fromisoformat(text)
    ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    props = tuple((k, record[k]) for k in ("platform", "source", "plan", "page"))
    return Event(
        str(record["event_id"]),
        str(record["user_id"]),
        str(record["event_name"]),
        ts,
        props,  # type: ignore[arg-type]
    )


def clean(events: Iterable[Event]) -> dict[str, list[Event]]:
    """Drop duplicate ids, users without a signup, and events before the first signup."""
    seen: dict[str, Event] = {}
    for event in events:
        kept = seen.get(event.event_id)
        if kept is None or (event.ts, event.user_id, event.name) < (
            kept.ts,
            kept.user_id,
            kept.name,
        ):
            seen[event.event_id] = event
    by_user: dict[str, list[Event]] = defaultdict(list)
    for event in seen.values():
        by_user[event.user_id].append(event)
    cleaned: dict[str, list[Event]] = {}
    for user, user_events in by_user.items():
        signups = [e.ts for e in user_events if e.name == SIGNUP_EVENT]
        if not signups:
            continue
        first = min(signups)
        cleaned[user] = [e for e in user_events if e.ts >= first]
    return cleaned


def _qualifies(event: Event, funnel: Funnel, index: int) -> bool:
    step = funnel.steps[index]
    return event.name == step.event and all(event.prop(k) == v for k, v in step.where.items())


def _valid_path(path: Sequence[Event], funnel: Funnel, anchor: datetime) -> bool:
    """Is this set of events a valid path through the first ``len(path)`` steps?"""
    k = len(path)
    # Label by step: each of the first k steps must be matched by exactly one event.
    ordered: list[Event | None] = [None] * k
    for event in path:
        slots = [i for i in range(k) if _qualifies(event, funnel, i)]
        if len(slots) != 1 or ordered[slots[0]] is not None:
            return False
        ordered[slots[0]] = event
    chain = [e for e in ordered if e is not None]
    if len(chain) != k:
        return False
    if chain[0].ts != anchor:
        return False
    deadline = anchor + timedelta(days=funnel.window_days)
    if any(e.ts > deadline for e in chain):
        return False
    return all(chain[i].ts <= chain[i + 1].ts for i in range(k - 1))


def furthest_step(user_events: Sequence[Event], funnel: Funnel) -> int:
    """Deepest valid path length for one user's (already cleaned) events."""
    step_one = [e.ts for e in user_events if _qualifies(e, funnel, 0)]
    if not step_one:
        return 0
    anchor = min(step_one)
    relevant = [
        e for e in user_events if any(_qualifies(e, funnel, i) for i in range(len(funnel.steps)))
    ]
    for k in range(len(funnel.steps), 0, -1):
        for path in combinations(relevant, k):
            if _valid_path(path, funnel, anchor):
                return k
    return 0


def furthest_steps(rows: Iterable[Sequence[object]], funnel: Funnel) -> dict[str, int]:
    """Furthest step for every user that has a signup, from raw event tuples."""
    cleaned = clean(parse_event(row) for row in rows)
    return {user: furthest_step(events, funnel) for user, events in cleaned.items()}
