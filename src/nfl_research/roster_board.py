"""Full-roster view: every player on a Sleeper roster (starters and bench
alike, IR/taxi included) with both FantasyPros ECR and the XGBoost weekly
projection attached side by side.

Distinct from ``ecr.report.build_league_board`` /
``sleeper.report.build_model_league_board``, which restrict to eligible
players and solve a specific lineup. This module answers a different
question — "what does the data say about my whole roster" — for
``scripts/update_dashboard.py``'s roster tabs.
"""

from __future__ import annotations

import pandas as pd

from .ecr.match import attach_ecr
from .sleeper.optimize import MODELED_POSITIONS, adjust_for_scoring
from .sleeper.rosters import LeagueRoster
from .weekly.projection import WeeklyModelBundle, project_players

ECR_COLS = ["rank", "ecr", "best", "worst", "sd", "pos_rank"]
PROJECTION_COLS = ["proj_points", "floor", "ceiling", "explosiveness_score", "risk_tier"]
LAST_GAME_COLS = ["last_game", "last_game_points"]


def _attach_last_game(board: pd.DataFrame, bundle: WeeklyModelBundle) -> pd.DataFrame:
    """Each player's most recent *played* game (any season in ``bundle.weekly``,
    so this naturally falls back to last season late in the offseason / week 1
    before any current-season games exist) and its actual PPR points.
    """
    weekly = bundle.weekly
    if weekly.empty or "gsis_id" not in board.columns:
        board["last_game"] = pd.NA
        board["last_game_points"] = pd.NA
        return board

    last = (
        weekly.sort_values(["player_id", "season", "week"])
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id", "season", "week", "fantasy_points_ppr"]]
    )
    last["last_game"] = last["season"].astype(str) + " wk" + last["week"].astype(str)
    last = last.rename(columns={"fantasy_points_ppr": "last_game_points"})

    board = board.merge(
        last[["player_id", "last_game", "last_game_points"]],
        left_on="gsis_id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")
    return board


def build_full_roster_board(
    league: LeagueRoster, ecr: pd.DataFrame, bundle: WeeklyModelBundle, season: int, week: int
) -> pd.DataFrame:
    """Every rostered player with ECR + projection + last-game-actual columns
    attached (NaN where a player has no ECR match, isn't in the modeled
    QB/RB/WR/TE set with enough history — e.g. K/DST/IDP never get
    proj_points — or has never played a tracked game).
    """
    roster_df = league.players
    if roster_df.empty:
        return pd.DataFrame()

    board = attach_ecr(roster_df, ecr)

    candidates = set(
        board.loc[board["position"].isin(MODELED_POSITIONS) & board["gsis_id"].notna(), "gsis_id"]
    )
    if candidates:
        projected = project_players(candidates, bundle, season=season, week=week)
        if not projected.empty:
            projected = adjust_for_scoring(projected, league.scoring_settings)
            board = board.merge(
                projected[["player_id", *PROJECTION_COLS]],
                left_on="gsis_id", right_on="player_id", how="left",
            ).drop(columns=["player_id"], errors="ignore")

    for col in PROJECTION_COLS:
        if col not in board.columns:
            board[col] = pd.NA

    board = _attach_last_game(board, bundle)

    keep = ["player_name", "position", "team", "is_starter", "is_ir", "is_taxi",
            *ECR_COLS, *PROJECTION_COLS, *LAST_GAME_COLS]
    return board[[c for c in keep if c in board.columns]]
