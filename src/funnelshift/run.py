"""Execute the funnel SQL in DuckDB over Parquet event files."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from .definitions import Funnel, validate_segments
from .render import read_sql, render_steps

#: Column order of the raw event feed.
RAW_COLUMNS: tuple[str, ...] = (
    "event_id",
    "user_id",
    "event_name",
    "event_time",
    "platform",
    "source",
    "plan",
    "page",
)

_RAW_DDL = (
    "CREATE OR REPLACE TABLE raw_events ("
    "event_id VARCHAR, user_id VARCHAR, event_name VARCHAR, event_time VARCHAR, "
    "platform VARCHAR, source VARCHAR, plan VARCHAR, page VARCHAR)"
)


def connect() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection with the session time zone fixed to UTC."""
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone = 'UTC'")
    return con


def load_parquet(con: duckdb.DuckDBPyConnection, path: str | Path) -> None:
    """Load an event Parquet file as ``raw_events``."""
    con.execute(_RAW_DDL)
    con.execute(
        f"INSERT INTO raw_events SELECT {', '.join(RAW_COLUMNS)} FROM read_parquet(?)",
        [str(path)],
    )


def load_rows(con: duckdb.DuckDBPyConnection, rows: Iterable[Sequence[object]]) -> None:
    """Load event tuples (in ``RAW_COLUMNS`` order) as ``raw_events``."""
    con.execute(_RAW_DDL)
    batch = [list(r) for r in rows]
    if batch:
        placeholders = ", ".join("?" for _ in RAW_COLUMNS)
        con.executemany(f"INSERT INTO raw_events VALUES ({placeholders})", batch)


@dataclass(frozen=True)
class CleanStats:
    raw_rows: int
    duplicate_rows_dropped: int
    no_signup_rows_dropped: int
    before_signup_rows_dropped: int
    kept_rows: int


@dataclass(frozen=True)
class MixRate:
    """Result of the shift-share decomposition between two periods."""

    base_weeks: list[str]
    test_weeks: list[str]
    unavailable_weeks: list[str]
    conv_base: float
    conv_test: float
    delta: float
    mix: float
    rate: float
    users_base: int
    users_test: int
    cells: pd.DataFrame
    steps: pd.DataFrame


class FunnelRun:
    """One funnel executed over one event set: clean, match steps, summarise, decompose."""

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        funnel: Funnel,
        segments: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> None:
        self.con = con
        self.funnel = funnel
        self.segments = validate_segments(list(segments or funnel.segments))
        self.stats = self._clean()
        self._run_steps()
        self.as_of = as_of if as_of is not None else self._latest_event()
        self._run_summary()

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        funnel: Funnel,
        segments: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> FunnelRun:
        con = connect()
        load_parquet(con, path)
        return cls(con, funnel, segments, as_of)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Sequence[object]],
        funnel: Funnel,
        segments: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> FunnelRun:
        con = connect()
        load_rows(con, rows)
        return cls(con, funnel, segments, as_of)

    # -- pipeline stages -------------------------------------------------------------------

    def _clean(self) -> CleanStats:
        self.con.execute(read_sql("clean"))
        row = self.con.execute("SELECT * FROM clean_stats").fetchone()
        assert row is not None
        return CleanStats(*(int(v) for v in row))

    def _run_steps(self) -> None:
        self.con.execute(render_steps(self.funnel, self.segments))

    def _latest_event(self) -> datetime:
        row = self.con.execute("SELECT max(ts) FROM clean_events").fetchone()
        if row is None or row[0] is None:
            raise ValueError("no events to analyse")
        return row[0]

    def _run_summary(self) -> None:
        self.con.execute(
            "CREATE OR REPLACE TABLE funnel_params AS "
            "SELECT CAST(? AS TIMESTAMP) AS as_of, CAST(? AS INTEGER) AS window_days",
            [self.as_of, self.funnel.window_days],
        )
        self.con.execute(read_sql("summary"))

    # -- results ---------------------------------------------------------------------------

    def furthest(self) -> dict[str, int]:
        """Furthest step reached per user (users with a signup only)."""
        rows = self.con.execute("SELECT user_id, furthest_step FROM funnel_steps").fetchall()
        return {user: int(step) for user, step in rows}

    def steps_table(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM funnel_steps ORDER BY user_id").df()

    def summary(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM funnel_summary").df()

    def cohort_status(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM funnel_cohort_status ORDER BY cohort_week").df()

    def complete_weeks(self) -> set[str]:
        rows = self.con.execute(
            "SELECT cohort_week FROM funnel_cohort_status WHERE complete"
        ).fetchall()
        return {r[0] for r in rows}

    def mix_rate(self, base_weeks: Sequence[str], test_weeks: Sequence[str]) -> MixRate:
        """Decompose the change in end-to-end conversion from ``base_weeks`` to ``test_weeks``.

        Only cohorts whose full window has elapsed take part; requested weeks that are not
        complete (or have no signups) are listed in ``unavailable_weeks``.
        """
        base, test = list(dict.fromkeys(base_weeks)), list(dict.fromkeys(test_weeks))
        if not base or not test:
            raise ValueError("both periods need at least one week")
        if set(base) & set(test):
            raise ValueError(f"periods overlap: {sorted(set(base) & set(test))}")
        available = self.complete_weeks()
        base_ok = [w for w in base if w in available]
        test_ok = [w for w in test if w in available]
        for name, ok in (("base", base_ok), ("test", test_ok)):
            if not ok:
                raise ValueError(f"no complete cohorts in the {name} period")
        unavailable = [w for w in base + test if w not in available]
        # Periods are the exact week lists, so a gap in a list stays a gap.
        self.con.execute("CREATE OR REPLACE TABLE mixrate_weeks (week VARCHAR, period INTEGER)")
        self.con.executemany(
            "INSERT INTO mixrate_weeks VALUES (?, ?)",
            [[w, 0] for w in base_ok] + [[w, 1] for w in test_ok],
        )
        self.con.execute(read_sql("mix_rate"))
        total = self.con.execute("SELECT * FROM mix_rate_total").fetchone()
        assert total is not None
        conv_base, conv_test, delta, mix, rate, users_base, users_test = total
        return MixRate(
            base_weeks=base_ok,
            test_weeks=test_ok,
            unavailable_weeks=unavailable,
            conv_base=float(conv_base),
            conv_test=float(conv_test),
            delta=float(delta),
            mix=float(mix),
            rate=float(rate),
            users_base=int(users_base),
            users_test=int(users_test),
            cells=self.con.execute("SELECT * FROM mix_rate_cells").df(),
            steps=self.con.execute("SELECT * FROM mix_rate_steps").df(),
        )
