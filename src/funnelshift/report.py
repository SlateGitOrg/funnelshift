"""Turn funnel summaries and decompositions into tables and the compare report."""

from __future__ import annotations

import pandas as pd

from .definitions import Funnel
from .run import MixRate

VERDICT_RATIO = 2.0
TOP_CELLS = 5


def verdict(mix: float, rate: float) -> str:
    """`mix_driven` when |mix| >= 2 x |rate|, `rate_driven` when |rate| >= 2 x |mix|."""
    if mix == 0 and rate == 0:
        return "no_change"
    if abs(mix) >= VERDICT_RATIO * abs(rate):
        return "mix_driven"
    if abs(rate) >= VERDICT_RATIO * abs(mix):
        return "rate_driven"
    return "mixed"


def pp(value: float) -> str:
    """A fraction as signed percentage points, e.g. -0.0285 -> '-2.85'."""
    text = f"{value * 100:+.2f}"
    return "+0.00" if text == "-0.00" else text


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def split_segment(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add one column per segment attribute by splitting the `segment` key."""
    parts = frame["segment"].str.split("|", expand=True)
    out = frame.copy()
    for i, column in enumerate(columns):
        out[column] = parts[i]
    return out


def top_step_cells(result: MixRate, columns: list[str], n: int = TOP_CELLS) -> pd.DataFrame:
    """The `n` step x segment cells with the largest absolute rate contribution."""
    steps = split_segment(result.steps, columns)
    steps["abs_rate"] = steps["rate"].abs()
    return steps.sort_values(
        ["abs_rate", "step_no", "segment"], ascending=[False, True, True]
    ).head(n)


def period_table(
    summary: pd.DataFrame, weeks: list[str], columns: list[str], by: str
) -> pd.DataFrame:
    """Users and conversion per step for each value of `by`, plus an `all` group.

    Counts are summed over the chosen cohort weeks and every other segment attribute, and
    conversion is recomputed from those sums.
    """
    frame = split_segment(summary[summary["cohort_week"].isin(weeks)], columns)
    if frame.empty:
        raise ValueError("no complete cohorts in the requested weeks")
    grouped = frame.groupby([by, "step_no", "step_name"], as_index=False)["users_reached"].sum()
    overall = frame.groupby(["step_no", "step_name"], as_index=False)["users_reached"].sum()
    overall[by] = "all"
    table = pd.concat([grouped, overall], ignore_index=True).sort_values([by, "step_no"])
    first = table[table["step_no"] == 1].set_index(by)["users_reached"]
    table["cum_conv"] = table["users_reached"] / table[by].map(first)
    table["step_conv"] = table.groupby(by)["users_reached"].transform(lambda s: s / s.shift(1))
    return table.reset_index(drop=True)


def compare_markdown(funnel: Funnel, result: MixRate, by: list[str]) -> str:
    """The `reports/compare.md` text."""
    label_base = f"{result.base_weeks[0]}..{result.base_weeks[-1]}"
    label_test = f"{result.test_weeks[0]}..{result.test_weeks[-1]}"
    check = abs(result.delta - result.mix - result.rate)
    lines = [
        f"# Conversion change: {label_base} vs {label_test}",
        "",
        f"Funnel `{funnel.name}`: {' > '.join(funnel.step_names)}, "
        f"{funnel.window_days}-day window from signup. Cells are split by {', '.join(by)}.",
        "",
        "| | Base | Test |",
        "|---|---|---|",
        f"| Cohort weeks | {label_base} | {label_test} |",
        f"| Users | {result.users_base:,} | {result.users_test:,} |",
        f"| End-to-end conversion | {pct(result.conv_base)} | {pct(result.conv_test)} |",
        "",
        "| Effect | Percentage points |",
        "|---|---|",
        f"| Total change | {pp(result.delta)} |",
        f"| Mix (who signs up) | {pp(result.mix)} |",
        f"| Rate (how each cell converts) | {pp(result.rate)} |",
        "",
        f"Mix + rate - total = {check:.1e} (midpoint weights leave no residual).",
        "",
        f"Verdict: **{verdict(result.mix, result.rate)}** "
        f"(`mix_driven` when |mix| >= 2 x |rate|, `rate_driven` when |rate| >= 2 x |mix|, "
        "otherwise `mixed`).",
    ]
    if result.unavailable_weeks:
        lines += [
            "",
            "Left out because their window has not fully elapsed: "
            + ", ".join(result.unavailable_weeks)
            + ".",
        ]

    cells = split_segment(result.cells, by)
    cells["abs_mix"] = cells["mix_c"].abs()
    lines += [
        "",
        "## End-to-end effect by cell",
        "",
        "Mix is centred on the overall midpoint conversion: (w1 - w0) x ((c0 + c1) / 2 - C). "
        "A cell that gains weight at below-average conversion is negative. Both columns "
        "sum to the totals above.",
        "",
        "| Cell | Weight base > test | Conversion base > test | Mix (pp) | Rate (pp) |",
        "|---|---|---|---|---|",
    ]
    for row in cells.sort_values("abs_mix", ascending=False).itertuples():
        lines.append(
            f"| {row.segment.replace('|', ' / ')} | {pct(row.w0)} > {pct(row.w1)} | "
            f"{pct(row.c0)} > {pct(row.c1)} | {pp(row.mix_c)} | {pp(row.rate)} |"
        )

    top = top_step_cells(result, by)
    lines += [
        "",
        f"## Top {len(top)} step x segment cells by rate contribution",
        "",
        "Rate contribution is the cell's share of the change in that step's conditional "
        "rate (users at the step / users at the previous step), in percentage points.",
        "",
        "| Step | Cell | Conditional rate base > test | Rate contribution (pp) |",
        "|---|---|---|---|",
    ]
    for row in top.itertuples():
        prev = funnel.step_names[row.step_no - 2]
        lines.append(
            f"| {prev} > {row.step_name} | {row.segment.replace('|', ' / ')} | "
            f"{pct(row.r0)} > {pct(row.r1)} | {pp(row.rate)} |"
        )
    return "\n".join(lines) + "\n"
