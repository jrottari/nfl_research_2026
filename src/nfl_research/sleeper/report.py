"""Shared league-board builder for the Ridge/XGBoost model-based optimizer.

Used by ``scripts/sleeper_lineup.py`` (console) and, alongside
``nfl_research.ecr.report``, by ``scripts/three_lineups.py`` /
``scripts/update_dashboard.py`` — so every caller builds a model-based
lineup the same way.
"""

from __future__ import annotations

import pandas as pd

from ..weekly.projection import WeeklyModelBundle, project_players
from .optimize import (
    MODELED_POSITIONS,
    SLOT_ELIGIBILITY,
    adjust_for_scoring,
    modeled_slots,
    optimize_lineup,
)
from .rosters import LeagueRoster


def _fill_unmodeled_slot(slot: str, roster_df: pd.DataFrame) -> dict | None:
    """K/DEF/IDP slots aren't projected — carry over the actual current
    Sleeper starter for that slot, or any eligible rostered player if none is
    currently started, or leave it empty."""
    eligible = SLOT_ELIGIBILITY.get(slot, {slot})
    startable = roster_df["position"].isin(eligible) & ~roster_df["is_ir"] & ~roster_df["is_taxi"]
    pool = roster_df[startable]
    if pool.empty:
        return None
    started = pool[pool["is_starter"]]
    pick = started.iloc[0] if not started.empty else pool.iloc[0]
    row = pick.to_dict()
    row["slot"] = slot
    return row


def build_model_league_board(
    league: LeagueRoster,
    bundle: WeeklyModelBundle,
    season: int,
    week: int,
    objective_col: str = "proj_points",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Returns ``(lineup_df, bench_df, empty_slots)``. ``lineup_df`` includes
    both model-optimized starters (QB/RB/WR/TE) and unmodeled-slot
    pass-through rows (K/DEF/IDP, carried over from the actual current
    roster — see ``_fill_unmodeled_slot``).
    """
    roster_df = league.players
    if roster_df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    candidates = set(
        roster_df.loc[
            roster_df["position"].isin(MODELED_POSITIONS)
            & roster_df["gsis_id"].notna()
            & ~roster_df["is_ir"]
            & ~roster_df["is_taxi"],
            "gsis_id",
        ]
    )
    if not candidates:
        return pd.DataFrame(), roster_df, []

    projected = project_players(candidates, bundle, season=season, week=week)
    if projected.empty:
        return pd.DataFrame(), roster_df, []

    projected = adjust_for_scoring(projected, league.scoring_settings)
    proj_cols = ["player_id", "proj_points", "floor", "ceiling", "explosiveness_score", "risk_tier"]
    board = roster_df.merge(
        projected[proj_cols], left_on="gsis_id", right_on="player_id", how="left"
    )

    startable = board[
        board["position"].isin(MODELED_POSITIONS)
        & board["proj_points"].notna()
        & ~board["is_ir"]
        & ~board["is_taxi"]
    ].reset_index(drop=True)

    m_slots = modeled_slots(league.roster_positions)
    starters_df, bench_df, empty_slots = optimize_lineup(
        startable, m_slots, objective_col=objective_col
    )

    unmodeled = [s for s in league.roster_positions if s not in m_slots]
    unmodeled_rows = [r for s in unmodeled if (r := _fill_unmodeled_slot(s, roster_df)) is not None]

    lineup = pd.concat([starters_df, pd.DataFrame(unmodeled_rows)], ignore_index=True, sort=False)
    return lineup, bench_df, empty_slots
