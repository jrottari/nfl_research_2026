"""Shared league-board builder used by both ``scripts/ecr_lineup.py`` (prints
to the console) and ``scripts/update_dashboard.py`` (pushes to Sheets), so
the two never drift out of sync on how a lineup gets built.
"""

from __future__ import annotations

import pandas as pd

from ..sleeper.rosters import LeagueRoster
from .match import attach_ecr
from .optimize import add_normalized_value, modeled_slots, optimize_ecr_lineup


def build_league_board(
    league: LeagueRoster, ecr: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    """Returns ``(starters_df, bench_df, empty_slots, unmatched_df)``."""
    roster_df = league.players
    if roster_df.empty:
        return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame()

    startable = roster_df[~roster_df["is_ir"] & ~roster_df["is_taxi"]].reset_index(drop=True)
    board = attach_ecr(startable, ecr)
    board = add_normalized_value(board, mode=mode)

    unmatched = board[board["objective_value"].isna() & board["position"].notna()]
    slots = modeled_slots(league.roster_positions)
    starters_df, bench_df, empty_slots = optimize_ecr_lineup(board, slots)
    return starters_df, bench_df, empty_slots, unmatched
