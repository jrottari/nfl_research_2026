"""Chart-source tabs derived from the running History log and this week's
Summary. A Sheets chart needs each series in its own column, but History is
long-format (one row per league per week) and Summary is long-format too
(one row per league per lineup type) — so each function here pivots to wide
first, pushes the reshaped table to its own tab, then charts that tab.
"""

from __future__ import annotations

import gspread
import pandas as pd

from .charts import replace_chart
from .dashboard import HISTORY_TAB, get_or_create_worksheet, push_dataframe


def read_history(spreadsheet: gspread.Spreadsheet) -> pd.DataFrame:
    """The full accumulated History tab (not just this run's new rows).

    Parsed manually rather than via ``get_all_records()`` — that worksheet
    was created wider than its actual data (``get_or_create_worksheet``'s
    default column count), so the trailing blank header cells register as
    duplicate empty-string headers and gspread raises.
    """
    ws = get_or_create_worksheet(spreadsheet, HISTORY_TAB, rows=2000)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    header, *rows = values
    n_cols = len(header)
    while n_cols > 0 and not header[n_cols - 1]:
        n_cols -= 1
    header = header[:n_cols]
    rows = [r[:n_cols] + [""] * (n_cols - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=header)


def _pivot_metric(history: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if history.empty or value_col not in history.columns:
        return pd.DataFrame()
    history = history.copy()
    history[value_col] = pd.to_numeric(history[value_col], errors="coerce")
    history["week"] = pd.to_numeric(history["week"], errors="coerce")
    pivot = history.pivot_table(index="week", columns="league", values=value_col, aggfunc="mean")
    return pivot.reset_index().sort_values("week")


def push_trend_chart(
    spreadsheet: gspread.Spreadsheet,
    history: pd.DataFrame,
    value_col: str,
    tab_name: str,
    title: str,
    reverse_y: bool = False,
) -> int:
    """Pivot ``value_col`` to wide (week x league) on ``tab_name`` and chart
    it as one line per league. Returns the number of week-rows charted.
    """
    wide = _pivot_metric(history, value_col)
    push_dataframe(spreadsheet, tab_name, wide)
    if wide.empty:
        return 0

    ws = spreadsheet.worksheet(tab_name)
    n_leagues = wide.shape[1] - 1
    replace_chart(
        spreadsheet, ws.id, "LINE", title,
        n_data_rows=len(wide) + 1, domain_col=0, series_cols=list(range(1, n_leagues + 1)),
        reverse_y=reverse_y,
    )
    return len(wide)


def push_summary_charts(spreadsheet: gspread.Spreadsheet, summary_df: pd.DataFrame) -> None:
    """This week's lineup comparison across leagues, split into two charts
    since rank (lower=better) and points (higher=better) don't share a
    scale: 'Chart Source - Rank' (ecr_consensus/ecr_ceiling) and
    'Chart Source - Points' (xgboost_points).
    """
    if summary_df.empty:
        return

    rank_wide = (
        summary_df[summary_df["lineup"].isin(["ecr_consensus", "ecr_ceiling"])]
        .pivot(index="league", columns="lineup", values="avg_metric")
        .reset_index()
    )
    push_dataframe(spreadsheet, "Chart Source - Rank", rank_wide)
    if not rank_wide.empty:
        ws = spreadsheet.worksheet("Chart Source - Rank")
        n_series = rank_wide.shape[1] - 1
        replace_chart(
            spreadsheet, ws.id, "COLUMN", "Avg Starting ECR Rank by League (lower = better)",
            n_data_rows=len(rank_wide) + 1, domain_col=0, series_cols=list(range(1, n_series + 1)),
        )

    points_wide = (
        summary_df[summary_df["lineup"] == "xgboost_points"][["league", "avg_metric"]]
        .rename(columns={"avg_metric": "xgboost_avg_points"})
    )
    push_dataframe(spreadsheet, "Chart Source - Points", points_wide)
    if not points_wide.empty:
        ws = spreadsheet.worksheet("Chart Source - Points")
        replace_chart(
            spreadsheet, ws.id, "COLUMN", "Avg XGBoost Projected Points by League",
            n_data_rows=len(points_wide) + 1, domain_col=0, series_cols=[1],
        )
