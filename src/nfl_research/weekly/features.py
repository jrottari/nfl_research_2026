"""Feature engineering for within-season weekly game forecasting.

Each row in the output panel represents a (player_id, season, week) where
the player was active. All features are computed from *prior* games only —
no look-ahead.

Key feature groups
------------------
Recent form       ppr_lag1, ppr_ma3, ppr_ma5
Season role       targets_ma3, carries_ma3, target_share_ma3, ppr_season_avg
Context           week, games_played, pos_code
Opponent          opp_ppr_allowed_avg (defense vs position, current season)
Prior season      prior_season_ppg, prior_season_games
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..forecasting.features import POSITION_CODES


def _rolling_within(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over the last ``window`` *previous* games (shift(1) applied)."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def build_weekly_panel(
    weekly: pd.DataFrame,
    defense_df: pd.DataFrame | None = None,
    prior_season_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a model-ready weekly panel with lag / rolling features.

    Args:
        weekly:          Long-format weekly game data (output of ``load_multi_season_weekly``).
        defense_df:      Output of ``defense_vs_position``; merged in when provided.
        prior_season_df: Season-level data for the year before each game, used to
                         add prior-year PPG/games as features.

    Returns:
        Panel where each row is a playable game with rolling features derived
        from prior games. Rows with no prior game history are dropped.
        Target column: ``target`` (this game's PPR points).
    """
    df = weekly.copy()
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    g = df.groupby(["player_id", "season"], sort=False)

    # --- Rolling PPR features ------------------------------------------------
    df["ppr_lag1"]       = g["fantasy_points_ppr"].shift(1)
    df["ppr_ma3"]        = g["fantasy_points_ppr"].apply(lambda s: _rolling_within(s, 3)).values
    df["ppr_ma5"]        = g["fantasy_points_ppr"].apply(lambda s: _rolling_within(s, 5)).values
    df["ppr_season_avg"] = g["fantasy_points_ppr"].apply(lambda s: _rolling_within(s, 17)).values

    # --- Rolling volume features ---------------------------------------------
    if "targets" in df.columns:
        df["targets_ma3"] = g["targets"].apply(lambda s: _rolling_within(s, 3)).values
    else:
        df["targets_ma3"] = 0.0

    if "carries" in df.columns:
        df["carries_ma3"] = g["carries"].apply(lambda s: _rolling_within(s, 3)).values
    else:
        df["carries_ma3"] = 0.0

    if "receptions" in df.columns:
        df["receptions_ma3"] = g["receptions"].apply(lambda s: _rolling_within(s, 3)).values
    else:
        df["receptions_ma3"] = 0.0

    if "target_share" in df.columns:
        df["target_share_ma3"] = g["target_share"].apply(lambda s: _rolling_within(s, 3)).values
    else:
        df["target_share_ma3"] = float("nan")

    # --- Momentum / trend ----------------------------------------------------
    df["ppr_trend"] = df["ppr_lag1"] - df["ppr_ma3"]   # last game vs recent average

    # --- Context features ----------------------------------------------------
    df["games_played"] = g.cumcount()  # 0 = first game (no history yet)
    df["pos_code"]     = df["position"].map(POSITION_CODES).fillna(-1).astype(int)
    df["week_norm"]    = (df["week"] - 1) / 16.0  # 0..1 normalised week number

    # --- Opponent matchup ----------------------------------------------------
    if defense_df is not None and not defense_df.empty and "opponent" in df.columns:
        df = df.merge(
            defense_df.rename(columns={"def_team": "opponent"}),
            on=["opponent", "season", "week", "position"],
            how="left",
        )
    else:
        df["opp_ppr_allowed_avg"] = float("nan")

    # --- Prior-season features -----------------------------------------------
    if prior_season_df is not None and not prior_season_df.empty:
        ps = prior_season_df[["player_id", "season", "fantasy_points_ppr", "games", "ppg_ppr"]].copy()
        ps = ps.rename(columns={
            "season": "prior_season",
            "fantasy_points_ppr": "prior_season_total",
            "games": "prior_season_games",
            "ppg_ppr": "prior_season_ppg",
        })
        # Join on player + (this_season - 1 == prior_season)
        df["prior_season_key"] = df["season"].astype(int) - 1
        df = df.merge(
            ps.rename(columns={"prior_season": "prior_season_key"}),
            on=["player_id", "prior_season_key"],
            how="left",
        )
        df = df.drop(columns=["prior_season_key"])
    else:
        df["prior_season_ppg"]   = float("nan")
        df["prior_season_games"] = float("nan")

    # --- Target column -------------------------------------------------------
    df = df.rename(columns={"fantasy_points_ppr": "target"})

    # Drop rows with no prior game history
    panel = df[df["games_played"] >= 1].copy()
    return panel.reset_index(drop=True)


_WEEKLY_FEATURE_COLS = [
    "ppr_lag1",
    "ppr_ma3",
    "ppr_ma5",
    "ppr_season_avg",
    "targets_ma3",
    "carries_ma3",
    "receptions_ma3",
    "target_share_ma3",
    "ppr_trend",
    "games_played",
    "week_norm",
    "pos_code",
    "opp_ppr_allowed_avg",
    "prior_season_ppg",
    "prior_season_games",
]


def weekly_feature_cols() -> list[str]:
    return list(_WEEKLY_FEATURE_COLS)


def weekly_feature_matrix(panel: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    cols = [c for c in _WEEKLY_FEATURE_COLS if c in panel.columns]
    return panel[cols].fillna(fill_value)


def make_upcoming_row(
    player_id: str,
    weekly_history: pd.DataFrame,
    upcoming_week: int,
    upcoming_season: int,
    opponent: str | None = None,
    defense_df: pd.DataFrame | None = None,
    prior_season_ppg: float | None = None,
    prior_season_games: float | None = None,
) -> pd.DataFrame | None:
    """Build a single feature row for an upcoming game with no actuals yet.

    ``weekly_history`` should be the player's game log for the current season
    (weeks before ``upcoming_week``), already standardized and coerced.
    """
    hist = weekly_history[
        (weekly_history["player_id"] == player_id) &
        (weekly_history["season"] == upcoming_season) &
        (weekly_history["week"] < upcoming_week)
    ].sort_values("week")

    if hist.empty:
        return None

    ppr_col = "fantasy_points_ppr" if "fantasy_points_ppr" in hist.columns else "target"

    def safe_ma(col: str, n: int) -> float:
        vals = hist[col].dropna().values if col in hist.columns else np.array([])
        return float(np.mean(vals[-n:])) if len(vals) > 0 else 0.0

    ppr_vals = hist[ppr_col].values
    last = float(ppr_vals[-1]) if len(ppr_vals) else 0.0
    ma3  = float(np.mean(ppr_vals[-3:])) if len(ppr_vals) else 0.0
    ma5  = float(np.mean(ppr_vals[-5:])) if len(ppr_vals) else 0.0
    avg  = float(np.mean(ppr_vals))      if len(ppr_vals) else 0.0

    row: dict = {
        "player_id":         player_id,
        "player_name":       hist["player_name"].iloc[-1] if "player_name" in hist.columns else player_id,
        "position":          hist["position"].iloc[-1]    if "position"    in hist.columns else "UNK",
        "season":            upcoming_season,
        "week":              upcoming_week,
        "ppr_lag1":          last,
        "ppr_ma3":           ma3,
        "ppr_ma5":           ma5,
        "ppr_season_avg":    avg,
        "targets_ma3":       safe_ma("targets", 3),
        "carries_ma3":       safe_ma("carries", 3),
        "receptions_ma3":    safe_ma("receptions", 3),
        "target_share_ma3":  safe_ma("target_share", 3),
        "ppr_trend":         last - ma3,
        "games_played":      len(hist),
        "week_norm":         (upcoming_week - 1) / 16.0,
        "pos_code":          POSITION_CODES.get(
            hist["position"].iloc[-1] if "position" in hist.columns else "UNK", -1
        ),
        "prior_season_ppg":   prior_season_ppg   if prior_season_ppg   is not None else 0.0,
        "prior_season_games": prior_season_games if prior_season_games is not None else 0.0,
    }

    # Opponent matchup
    if defense_df is not None and opponent is not None and not defense_df.empty:
        pos = row["position"]
        opp_row = defense_df[
            (defense_df["def_team"] == opponent) &
            (defense_df["season"] == upcoming_season) &
            (defense_df["week"] == upcoming_week) &
            (defense_df["position"] == pos)
        ]
        row["opp_ppr_allowed_avg"] = float(opp_row["opp_ppr_allowed_avg"].values[0]) if not opp_row.empty else 0.0
    else:
        row["opp_ppr_allowed_avg"] = 0.0

    return pd.DataFrame([row])
