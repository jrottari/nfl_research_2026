"""Ranks, value over replacement, tiers, and the final board layouts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings

# Left-to-right: identity -> scoring -> value -> consistency -> volume ->
# efficiency -> raw box score. Freeze the first dozen in Sheets.
BOARD_COLS: tuple[str, ...] = (
    "rank_ppr",
    "player_name",
    "position",
    "pos_label",
    "team",
    "teams",
    "games",
    "games_active",
    "points_ppr",
    "ppg_ppr",
    "points_half",
    "ppg_half",
    "points_std",
    "ppg_std",
    "rank_ppg",
    "pos_rank",
    "pos_rank_ppg",
    "rank_delta",
    "rank_half",
    "rank_std",
    "vor_ppg",
    "vorp_total",
    "tier",
    "floor",
    "ceiling",
    "median_week",
    "stdev",
    "cv",
    "best_week",
    "worst_week",
    "boom_weeks",
    "boom_rate",
    "bust_weeks",
    "bust_rate",
    "starter_weeks",
    "starter_week_rate",
    "top5_weeks",
    "best_pos_finish",
    "touches",
    "touch_pg",
    "opportunities",
    "opp_pg",
    "target_share",
    "scrim_yards",
    "scrim_yds_pg",
    "total_yards",
    "total_tds",
    "tds_pg",
    "first_downs",
    "turnovers",
    "fumbles_lost",
    "two_pt",
    "pts_per_touch",
    "pts_per_opp",
    "yards_per_touch",
    "ypc",
    "ypr",
    "ypt",
    "catch_rate",
    "td_rate",
    "adot",
    "yac_per_rec",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "comp_pct",
    "ypa",
    "air_yards_pa",
    "td_int_ratio",
    "sack_rate",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)


def add_ranks(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Overall and positional ranks in all three scoring formats."""
    out = df.copy()
    qualified = out["games"] >= settings.min_games

    out["rank_ppr"] = out["points_ppr"].rank(ascending=False, method="min").astype(int)
    out["rank_half"] = out["points_half"].rank(ascending=False, method="min").astype(int)
    out["rank_std"] = out["points_std"].rank(ascending=False, method="min").astype(int)

    out["pos_rank"] = (
        out.groupby("position")["points_ppr"].rank(ascending=False, method="min").astype(int)
    )
    out["pos_label"] = out["position"] + out["pos_rank"].astype(str)

    out["rank_ppg"] = out["ppg_ppr"].where(qualified).rank(ascending=False, method="min")
    out["pos_rank_ppg"] = (
        out.where(qualified).groupby("position")["ppg_ppr"].rank(ascending=False, method="min")
    )
    # Positive = per-game play outran the season finish (i.e. missed time).
    out["rank_delta"] = out["rank_ppr"] - out["rank_ppg"]
    return out


def replacement_levels(df: pd.DataFrame, settings: Settings) -> dict[str, float]:
    """PPG of the replacement-level player at each position."""
    qualified = df["games"] >= settings.min_games
    levels: dict[str, float] = {}
    for pos, rank in settings.replacement_rank.items():
        pool = (
            df.loc[qualified & (df["position"] == pos), "ppg_ppr"]
            .dropna()
            .sort_values(ascending=False)
        )
        if len(pool) >= rank:
            levels[pos] = float(pool.iloc[rank - 1])
        elif len(pool):
            levels[pos] = float(pool.min())
        else:
            levels[pos] = float("nan")
    return levels


def add_vor(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Value over replacement, per game and season-long."""
    out = df.copy()
    levels = replacement_levels(out, settings)
    out["replacement_ppg"] = out["position"].map(levels)
    out["vor_ppg"] = out["ppg_ppr"] - out["replacement_ppg"]
    out["vorp_total"] = out["vor_ppg"] * out["games"]
    return out


def assign_tiers(values: pd.Series, gap: float) -> pd.Series:
    """Gap-based tiers: a new tier starts wherever the PPG drop exceeds ``gap``."""
    ordered = values.dropna().sort_values(ascending=False)
    tiers: list[int] = []
    tier = 1
    prev: float | None = None
    for value in ordered:
        if prev is not None and (prev - value) >= gap:
            tier += 1
        tiers.append(tier)
        prev = value
    return pd.Series(tiers, index=ordered.index, dtype="float")


def add_tiers(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    out = df.copy()
    qualified = out["games"] >= settings.min_games
    out["tier"] = np.nan
    for pos, gap in settings.tier_gap.items():
        mask = qualified & (out["position"] == pos)
        if mask.any():
            out.loc[mask, "tier"] = assign_tiers(out.loc[mask, "ppg_ppr"], gap)
    return out


def build_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by PPR total and lay the columns out in reading order."""
    ordered = df.sort_values("points_ppr", ascending=False).reset_index(drop=True)
    cols = [c for c in BOARD_COLS if c in ordered.columns]
    board = ordered.loc[:, cols].copy()
    board.attrs["player_ids"] = ordered["player_id"].tolist()
    return board


def position_board(overall: pd.DataFrame, position: str, size: int | None = None) -> pd.DataFrame:
    out = overall[overall["position"] == position].reset_index(drop=True)
    return out if size is None else out.head(size)


def flex_board(overall: pd.DataFrame, size: int = 200) -> pd.DataFrame:
    out = overall[overall["position"].isin(["RB", "WR", "TE"])].head(size).reset_index(drop=True)
    out.insert(0, "flex_rank", np.arange(1, len(out) + 1))
    return out


def value_gainers(df: pd.DataFrame, settings: Settings, n: int = 20) -> pd.DataFrame:
    """Players whose per-game production ran well ahead of their season finish."""
    qualified = (df["games"] >= settings.min_games) & (df["games"] < 15) & (df["ppg_ppr"] >= 10)
    cols = [
        "player_name",
        "pos_label",
        "team",
        "games",
        "points_ppr",
        "rank_ppr",
        "ppg_ppr",
        "rank_ppg",
        "rank_delta",
    ]
    cols = [c for c in cols if c in df.columns]
    return (
        df[qualified]
        .sort_values("rank_delta", ascending=False)
        .head(n)
        .loc[:, cols]
        .reset_index(drop=True)
    )
