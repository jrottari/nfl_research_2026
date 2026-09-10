"""Cross-league "Player Overview" tab: one row per unique rostered player,
a handful of at-a-glance stat columns (ECR, XGBoost projection, last game),
one checkbox column per league (currently started — blank if the player
isn't on that roster at all), and two live-formula summary columns
(``num_leagues_rostered``, ``num_leagues_starting``) that recompute inside
Sheets itself as checkboxes are toggled. This lets you try a different
lineup configuration by hand, in the Sheet, with no script rerun needed to
see the effect — only to re-sync against what Sleeper actually has started.

Two more tabs key off those same checkboxes, both pure Sheets formulas so
they too update live with no script rerun:
  - "Correlation Score" (``push_correlation_score``) — raw/max/normalized
    score for how consistently you start players you own in multiple
    leagues, normalized against the max achievable given your actual
    rosters. See that function's docstring for the exact definition. Kept
    on its own tab rather than as a panel on this one: this sheet has a
    Sheets "Table" object on it (set up outside this script) whose header
    row silently discards any formula written into it and replaces it with
    a default "Column N" name — confirmed by testing against a throwaway
    tab. Row 1 here is that Table's header row, so no formula can live in
    it; a separate, Table-free tab sidesteps the issue entirely.
  - "ECR Range by League" (``push_ecr_range_summary``) — each league's
    average best/consensus/worst-case ECR among its currently started
    players.
"""

from __future__ import annotations

import gspread
import numpy as np
import pandas as pd
from gspread.utils import ValueInputOption

from ..sleeper.rosters import LeagueRoster
from .charts import replace_chart
from .dashboard import get_or_create_worksheet, push_dataframe

PLAYER_OVERVIEW_TAB = "Player Overview"
LEADERBOARD_TAB = "Player Overview - Top Starts"
ECR_RANGE_TAB = "ECR Range by League"
CORRELATION_TAB = "Correlation Score"

# (output column name, source column in roster_board.build_full_roster_board's output)
STAT_COLS = [
    ("team", "team"),
    ("ecr_consensus_rank", "rank"),
    ("ecr_best_rank", "best"),
    ("ecr_worst_rank", "worst"),
    ("xgboost_proj_points", "proj_points"),
    ("xgboost_ceiling_points", "ceiling"),
    ("last_game_points", "last_game_points"),
    ("last_game", "last_game"),
]
# Column letters (within STAT_COLS) used by the live AVERAGEIF formulas in
# push_ecr_range_summary — kept as names, not hardcoded indices, so adding/
# reordering STAT_COLS can't silently point the range formulas at the wrong
# column.
_ECR_BAND_COLS = ["ecr_best_rank", "ecr_consensus_rank", "ecr_worst_rank"]


def _clean_value(val):
    """Round floats to avoid float32->float64 display artifacts (e.g.
    XGBoost's raw output showing as 11.80000019 instead of 11.8). Covers
    numpy float32/float64 too, not just Python float — itertuples() yields
    numpy scalars for numeric columns.
    """
    if pd.isna(val):
        return ""
    if isinstance(val, (float, np.floating)):
        return round(float(val), 2)
    return val


def _col_letter(idx: int) -> str:
    """1-indexed column number -> spreadsheet column letters (1 -> A, 27 -> AA)."""
    letters = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _dedupe_league_names(leagues: list[LeagueRoster]) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for league in leagues:
        seen[league.league_name] = seen.get(league.league_name, 0) + 1
        n = seen[league.league_name]
        names.append(league.league_name if n == 1 else f"{league.league_name} ({n})")
    return names


def _build_stats_lookup(
    roster_boards: list[pd.DataFrame],
) -> dict[tuple[str, str], dict]:
    """First non-empty board's values win for each (player_name, position) —
    a player's ECR/projection are effectively league-independent (mild
    exception: proj_points under non-PPR scoring adjustment), so any one
    league's board is a fine representative source.
    """
    lookup: dict[tuple[str, str], dict] = {}
    for board in roster_boards:
        if board.empty:
            continue
        for row in board.itertuples():
            key = (row.player_name, row.position)
            if key not in lookup:
                lookup[key] = {out: getattr(row, src, pd.NA) for out, src in STAT_COLS}
    return lookup


def build_player_overview_frame(
    leagues: list[LeagueRoster], roster_boards: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[str]]:
    """Returns ``(frame, league_columns)``. ``frame`` has ``player_name``,
    ``position``, the ``STAT_COLS`` (ECR/projection/last-game), then one
    column per league holding ``True``/``False`` (currently started/benched)
    or ``pd.NA`` if the player isn't rostered in that league at all.
    """
    league_cols = _dedupe_league_names(leagues)
    stats_lookup = _build_stats_lookup(roster_boards)

    player_positions: dict[tuple[str, str], None] = {}
    per_league_started: list[dict[tuple[str, str], bool]] = []
    for league in leagues:
        started = {
            (row.player_name, row.position): bool(row.is_starter)
            for row in league.players.itertuples()
        }
        per_league_started.append(started)
        for key in started:
            player_positions[key] = None

    stat_col_names = [out for out, _src in STAT_COLS]
    rows = []
    for player_name, position in sorted(player_positions):
        row = {"player_name": player_name, "position": position}
        row.update(stats_lookup.get((player_name, position), dict.fromkeys(stat_col_names, pd.NA)))
        for col_name, started in zip(league_cols, per_league_started, strict=True):
            row[col_name] = started.get((player_name, position), pd.NA)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["player_name", "position", *stat_col_names, *league_cols])
    return frame, league_cols


def push_player_overview(
    spreadsheet: gspread.Spreadsheet, frame: pd.DataFrame, league_cols: list[str]
) -> int:
    """Push/refresh the Player Overview tab. Returns the number of player rows.

    ``frame``/``league_cols`` come from ``build_player_overview_frame`` —
    built once by the caller so it can also feed ``push_leaderboard_chart``
    without recomputing.
    """
    n_rows = len(frame)
    n_league_cols = len(league_cols)
    n_stat_cols = len(STAT_COLS)

    first_league_col = 3 + n_stat_cols
    last_league_col = 2 + n_stat_cols + n_league_cols
    start_letter, end_letter = _col_letter(first_league_col), _col_letter(last_league_col)

    needed_cols = last_league_col + 2  # + num_leagues_rostered/num_leagues_starting
    needed_rows = n_rows + 10
    ws = get_or_create_worksheet(
        spreadsheet, PLAYER_OVERVIEW_TAB, rows=needed_rows, cols=needed_cols
    )
    # Exact-size (not just grow) so a shrinking league/roster count purges any
    # leftover columns rather than leaving stray blank ones the sheet's Table
    # object (see module docstring) would auto-stamp with "Column N" labels.
    if ws.col_count != needed_cols or ws.row_count < needed_rows:
        ws.resize(rows=max(ws.row_count, needed_rows), cols=needed_cols)
    ws.clear()

    stat_col_names = [out for out, _src in STAT_COLS]
    header = ["player_name", "position", *stat_col_names, *league_cols,
              "num_leagues_rostered", "num_leagues_starting"]
    values = [header]
    checkbox_requests = []
    for i in range(n_rows):
        row = frame.iloc[i]
        excel_row = i + 2
        stat_vals = [_clean_value(row[c]) for c in stat_col_names]
        league_vals = []
        for j, col_name in enumerate(league_cols):
            val = row[col_name]
            if pd.isna(val):
                league_vals.append("")
            else:
                league_vals.append(bool(val))
                checkbox_requests.append({
                    "setDataValidation": {
                        "range": {
                            "sheetId": ws.id,
                            "startRowIndex": excel_row - 1,
                            "endRowIndex": excel_row,
                            "startColumnIndex": first_league_col - 1 + j,
                            "endColumnIndex": first_league_col + j,
                        },
                        "rule": {"condition": {"type": "BOOLEAN"}, "strict": True},
                    }
                })
        rostered_formula = f"=COUNTA({start_letter}{excel_row}:{end_letter}{excel_row})"
        starting_formula = f"=COUNTIF({start_letter}{excel_row}:{end_letter}{excel_row},TRUE)"
        values.append([row["player_name"], row["position"], *stat_vals, *league_vals,
                       rostered_formula, starting_formula])

    ws.update(values=values, range_name="A1", value_input_option=ValueInputOption.user_entered)

    if checkbox_requests:
        spreadsheet.batch_update({"requests": checkbox_requests})

    return n_rows


def push_correlation_score(
    spreadsheet: gspread.Spreadsheet, frame: pd.DataFrame, league_cols: list[str]
) -> None:
    """Push the "Correlation Score" tab: raw / max-possible / normalized
    score for how consistently you start players you own in multiple
    leagues. Pure formulas against Player Overview's already-live
    ``num_leagues_rostered``/``num_leagues_starting`` columns, so this
    updates the moment you toggle a checkbox there — no script rerun.

    Definition (per project spec): only players rostered in 2+ leagues can
    contribute; only start-vs-bench matters, not points. For a player
    started in k of the n leagues that roster them, the number of leagues
    *simultaneously* starting them is C(k,2) = k(k-1)/2 — summed across all
    players, that's the raw correlation score. The max possible value of
    that same sum — if every multi-league player were started everywhere
    they're rostered — replaces k with n: C(n,2) = n(n-1)/2. Both reduce to
    0 for n<2 or k<2 automatically, no extra guarding needed. Normalized =
    raw / max.

    ``frame``/``league_cols`` must be the same objects already passed to
    ``push_player_overview`` for this run, so the column arithmetic here
    (which recomputes where num_leagues_rostered/starting landed) matches.
    """
    n_rows = len(frame)
    last_league_col = 2 + len(STAT_COLS) + len(league_cols)

    ws = get_or_create_worksheet(spreadsheet, CORRELATION_TAB, rows=6, cols=2)
    ws.clear()
    if n_rows == 0:
        return

    rostered_col = _col_letter(last_league_col + 1)
    starting_col = _col_letter(last_league_col + 2)
    rostered_range = f"'{PLAYER_OVERVIEW_TAB}'!{rostered_col}2:{rostered_col}{n_rows + 1}"
    starting_range = f"'{PLAYER_OVERVIEW_TAB}'!{starting_col}2:{starting_col}{n_rows + 1}"

    values = [
        ["metric", "value"],
        ["Raw Correlation Score", f"=SUMPRODUCT({starting_range}*({starting_range}-1))/2"],
        ["Max Possible Correlation", f"=SUMPRODUCT({rostered_range}*({rostered_range}-1))/2"],
        ["Normalized Correlation Score", "=IFERROR(B2/B3,0)"],
    ]
    ws.update(values=values, range_name="A1", value_input_option=ValueInputOption.user_entered)

    spreadsheet.batch_update({"requests": [{
        "repeatCell": {
            "range": {"sheetId": ws.id, "startRowIndex": 3, "endRowIndex": 4,
                      "startColumnIndex": 1, "endColumnIndex": 2},
            "cell": {"userEnteredFormat": {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}}},
            "fields": "userEnteredFormat.numberFormat",
        }
    }]})


def build_leaderboard(
    frame: pd.DataFrame, league_cols: list[str], top_n: int = 20
) -> pd.DataFrame:
    """Which players are locks across the most leagues, and how much bench
    depth they're carrying (rostered-but-never-started gap). Computed
    straight from ``frame`` in Python — mirrors the Sheet's own
    COUNTA/COUNTIF formulas exactly, just without a network round-trip.
    """
    starts = (frame[league_cols] == True).sum(axis=1)  # noqa: E712 (NaN-safe; pd.NA == True -> False)
    rostered = frame[league_cols].notna().sum(axis=1)
    board = pd.DataFrame({
        "player_name": frame["player_name"],
        "position": frame["position"],
        "num_leagues_starting": starts,
        "num_leagues_rostered": rostered,
    })
    return (
        board.sort_values(["num_leagues_starting", "num_leagues_rostered"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def push_leaderboard_chart(
    spreadsheet: gspread.Spreadsheet, frame: pd.DataFrame, league_cols: list[str], top_n: int = 20
) -> int:
    """Push the leaderboard tab and a bar chart of it. Returns row count."""
    board = build_leaderboard(frame, league_cols, top_n=top_n)
    push_dataframe(spreadsheet, LEADERBOARD_TAB, board)
    if board.empty:
        return 0

    ws = spreadsheet.worksheet(LEADERBOARD_TAB)
    replace_chart(
        spreadsheet, ws.id, "COLUMN", "Most-Started / Most-Rostered Players",
        n_data_rows=len(board) + 1, domain_col=0, series_cols=[2, 3],
    )
    return len(board)


def push_ecr_range_summary(
    spreadsheet: gspread.Spreadsheet, frame: pd.DataFrame, league_cols: list[str]
) -> int:
    """Push the "ECR Range by League" tab + grouped bar chart: for each
    league, the average best-case / consensus / worst-case ECR (FantasyPros'
    own per-player expert-disagreement range) among that league's
    *currently started* players on the Player Overview tab.

    Pure ``AVERAGEIF`` cross-sheet formulas against Player Overview's
    checkbox + stat columns — no Python computation of the actual averages —
    so both this table and its chart move live as checkboxes are toggled
    there, same as the correlation panel. Remember lower ECR = better (the
    Sheets API has no axis-reversal field — see ``charts.replace_chart``).

    ``frame``/``league_cols`` must be the exact objects already passed to
    ``push_player_overview`` for the same run, so row order lines up.
    """
    n_rows = len(frame)
    ws = get_or_create_worksheet(spreadsheet, ECR_RANGE_TAB, rows=len(league_cols) + 10, cols=8)
    ws.clear()
    if n_rows == 0 or not league_cols:
        return 0

    first_league_col = 3 + len(STAT_COLS)
    stat_col_letter = {
        name: _col_letter(3 + i) for i, (name, _src) in enumerate(STAT_COLS)
    }

    header = ["league", "avg_best_ecr", "avg_consensus_ecr", "avg_worst_ecr"]
    values = [header]
    for j, league in enumerate(league_cols):
        checkbox_col = _col_letter(first_league_col + j)
        checkbox_range = f"'{PLAYER_OVERVIEW_TAB}'!{checkbox_col}2:{checkbox_col}{n_rows + 1}"
        row = [league]
        for band_name in _ECR_BAND_COLS:
            stat_col = stat_col_letter[band_name]
            stat_range = f"'{PLAYER_OVERVIEW_TAB}'!{stat_col}2:{stat_col}{n_rows + 1}"
            row.append(f'=IFERROR(AVERAGEIF({checkbox_range},TRUE,{stat_range}),"")')
        values.append(row)

    ws.update(values=values, range_name="A1", value_input_option=ValueInputOption.user_entered)

    replace_chart(
        spreadsheet, ws.id, "COLUMN",
        "Starting Lineup ECR Range by League (lower = better)",
        n_data_rows=len(league_cols) + 1, domain_col=0, series_cols=[1, 2, 3],
    )
    return len(league_cols)
