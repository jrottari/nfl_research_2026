"""Derived metrics: scoring formats, per-game rates, efficiency, consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings


def safe_div(num, den) -> pd.Series:
    """Element-wise divide returning NaN (not inf) where the denominator is 0."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return num / den


def games_played(weekly: pd.DataFrame) -> pd.DataFrame:
    """Games, active games, team, and mean target share, per player.

    ``games`` counts weeks the player appeared in the box score. ``games_active``
    only counts weeks with a real opportunity (carry, target or pass attempt),
    which is the fairer denominator for someone who dressed but never touched
    the ball.
    """
    df = weekly.copy()
    df["had_opportunity"] = (df["carries"] + df["targets"] + df["attempts"]) > 0
    df = df.sort_values(["player_id", "week"])

    agg = {
        "games": ("week", "nunique"),
        "games_active": ("had_opportunity", "sum"),
        "first_week": ("week", "min"),
        "last_week": ("week", "max"),
    }
    if "team" in df.columns:
        agg["team"] = ("team", "last")
        agg["teams"] = ("team", lambda s: "/".join(pd.unique(s)))
    if "target_share" in df.columns:
        # Season sums of a per-game share are meaningless; take the mean.
        agg["target_share"] = ("target_share", "mean")

    out = df.groupby("player_id").agg(**agg).reset_index()
    out["games"] = out["games"].astype(int)
    out["games_active"] = out["games_active"].astype(int)
    if "teams" in out.columns:
        out["multi_team"] = out["teams"].str.contains("/")
    return out


def add_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Standard / half-PPR / full-PPR totals and per-game rates.

    nflverse ``fantasy_points`` is standard (no reception bonus) and
    ``fantasy_points_ppr`` is full PPR, so half-PPR is exactly the midpoint.
    """
    out = df.copy()
    out["points_std"] = out["fantasy_points"]
    out["points_ppr"] = out["fantasy_points_ppr"]
    out["points_half"] = (out["points_std"] + out["points_ppr"]) / 2

    out["ppg_ppr"] = safe_div(out["points_ppr"], out["games"])
    out["ppg_half"] = safe_div(out["points_half"], out["games"])
    out["ppg_std"] = safe_div(out["points_std"], out["games"])
    return out


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Touches, opportunities, yardage and scoring totals."""
    out = df.copy()
    out["touches"] = out["carries"] + out["receptions"]
    out["opportunities"] = out["carries"] + out["targets"]
    out["touch_pg"] = safe_div(out["touches"], out["games"])
    out["opp_pg"] = safe_div(out["opportunities"], out["games"])

    out["scrim_yards"] = out["rushing_yards"] + out["receiving_yards"]
    out["total_yards"] = out["scrim_yards"] + out["passing_yards"]
    out["total_tds"] = (
        out["passing_tds"] + out["rushing_tds"]
        + out["receiving_tds"] + out["special_teams_tds"]
    )
    out["fumbles_lost"] = (
        out["sack_fumbles_lost"] + out["rushing_fumbles_lost"]
        + out["receiving_fumbles_lost"]
    )
    out["turnovers"] = out["interceptions"] + out["fumbles_lost"]
    out["first_downs"] = (
        out["passing_first_downs"] + out["rushing_first_downs"]
        + out["receiving_first_downs"]
    )
    out["two_pt"] = out["passing_2pt"] + out["rushing_2pt"] + out["receiving_2pt"]

    out["scrim_yds_pg"] = safe_div(out["scrim_yards"], out["games"])
    out["total_yds_pg"] = safe_div(out["total_yards"], out["games"])
    out["tds_pg"] = safe_div(out["total_tds"], out["games"])
    return out


def add_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Rate stats: points per touch/opportunity, yards per attempt, catch rate."""
    out = df.copy()
    out["pts_per_touch"] = safe_div(out["points_ppr"], out["touches"])
    out["pts_per_opp"] = safe_div(out["points_ppr"], out["opportunities"])
    out["yards_per_touch"] = safe_div(out["scrim_yards"], out["touches"])
    out["ypc"] = safe_div(out["rushing_yards"], out["carries"])
    out["ypr"] = safe_div(out["receiving_yards"], out["receptions"])
    out["ypt"] = safe_div(out["receiving_yards"], out["targets"])
    out["catch_rate"] = safe_div(out["receptions"], out["targets"])
    out["td_rate"] = safe_div(out["rushing_tds"] + out["receiving_tds"], out["touches"])
    out["yac_per_rec"] = safe_div(out["receiving_yac"], out["receptions"])
    out["adot"] = safe_div(out["receiving_air_yards"], out["targets"])

    out["comp_pct"] = safe_div(out["completions"], out["attempts"])
    out["ypa"] = safe_div(out["passing_yards"], out["attempts"])
    out["air_yards_pa"] = safe_div(out["passing_air_yards"], out["attempts"])
    out["td_int_ratio"] = safe_div(out["passing_tds"], out["interceptions"])
    out["sack_rate"] = safe_div(out["sacks_taken"], out["attempts"] + out["sacks_taken"])
    return out


def weekly_consistency(weekly: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Floor, ceiling, volatility, boom/bust and startable-week rates.

    Season totals reward availability; these columns describe *how* a player got
    there. ``floor`` and ``ceiling`` are the 10th and 90th percentile weeks.
    """
    df = weekly.copy()
    df["week_pos_rank"] = (
        df.groupby(["week", "position"])["fantasy_points_ppr"]
          .rank(ascending=False, method="min")
    )

    boom = df["position"].map(lambda p: settings.boom_bust.get(p, (np.inf, -np.inf))[0])
    bust = df["position"].map(lambda p: settings.boom_bust.get(p, (np.inf, -np.inf))[1])
    slots = df["position"].map(lambda p: settings.starter_slots.get(p, 0))

    df["is_boom"] = df["fantasy_points_ppr"] >= boom
    df["is_bust"] = df["fantasy_points_ppr"] <= bust
    df["is_starter"] = df["week_pos_rank"] <= slots
    df["is_top5"] = df["week_pos_rank"] <= 5

    out = (
        df.groupby("player_id")
          .agg(
              best_week=("fantasy_points_ppr", "max"),
              worst_week=("fantasy_points_ppr", "min"),
              median_week=("fantasy_points_ppr", "median"),
              stdev=("fantasy_points_ppr", "std"),
              floor=("fantasy_points_ppr", lambda x: x.quantile(0.10)),
              ceiling=("fantasy_points_ppr", lambda x: x.quantile(0.90)),
              boom_weeks=("is_boom", "sum"),
              bust_weeks=("is_bust", "sum"),
              starter_weeks=("is_starter", "sum"),
              top5_weeks=("is_top5", "sum"),
              best_pos_finish=("week_pos_rank", "min"),
          )
          .reset_index()
    )
    for col in ("boom_weeks", "bust_weeks", "starter_weeks", "top5_weeks"):
        out[col] = out[col].astype(int)
    return out


def add_consistency_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the weekly counts into per-game rates. Run after merging."""
    out = df.copy()
    out["cv"] = safe_div(out["stdev"], out["ppg_ppr"])
    out["boom_rate"] = safe_div(out["boom_weeks"], out["games"])
    out["bust_rate"] = safe_div(out["bust_weeks"], out["games"])
    out["starter_week_rate"] = safe_div(out["starter_weeks"], out["games"])
    return out


def weekly_game_log(weekly: pd.DataFrame, player_ids) -> pd.DataFrame:
    """Tidy week-by-week rows for a set of players - pivot-table fuel."""
    cols = [
        "player_id", "player_name", "position", "team", "opponent", "week",
        "fantasy_points_ppr", "fantasy_points", "carries", "rushing_yards",
        "rushing_tds", "targets", "receptions", "receiving_yards",
        "receiving_tds", "attempts", "passing_yards", "passing_tds",
        "interceptions",
    ]
    df = weekly[weekly["player_id"].isin(list(player_ids))].copy()
    if "week_pos_rank" not in df.columns:
        df["week_pos_rank"] = (
            df.groupby(["week", "position"])["fantasy_points_ppr"]
              .rank(ascending=False, method="min")
        )
    cols.append("week_pos_rank")
    cols = [c for c in cols if c in df.columns]
    return df.loc[:, cols].sort_values(["player_name", "week"]).reset_index(drop=True)
