"""Load and prepare multi-season weekly game data for within-season forecasting."""

from __future__ import annotations

import pandas as pd

from .. import schema

FORECAST_POSITIONS = ("QB", "RB", "WR", "TE")


def _import_nflreadpy():
    try:
        import nflreadpy as nfl

        return nfl
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("nflreadpy is required. pip install nflreadpy") from err


def _to_pandas(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    raise TypeError(f"Cannot convert {type(df)}")


def load_multi_season_weekly(
    seasons: list[int],
    positions: tuple[str, ...] = FORECAST_POSITIONS,
    season_type: str = "REG",
) -> pd.DataFrame:
    """Weekly game-level stats for multiple seasons, standardized to canonical columns.

    Returns one row per (player_id, season, week) for regular-season games only
    (unless season_type is overridden).
    """
    nfl = _import_nflreadpy()
    raw = _to_pandas(nfl.load_player_stats(seasons, summary_level="week"))

    df = schema.standardize(raw, strict=False)
    df = schema.coerce_numeric(df)

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    if season_type and "season_type" in df.columns:
        df = df[df["season_type"] == season_type].copy()

    if positions:
        df = df[df["position"].isin(positions)].copy()

    df = df[df["season"].isin(seasons)].copy()

    return df.reset_index(drop=True)


def defense_vs_position(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute, for each (team, season, week, position), the cumulative average PPR
    points allowed to that position through the *prior* week.

    This is the matchup quality signal: a "soft" defense that has given up 25+ PPR
    PPG to WRs this season will inflate WR projections this week.
    """
    if "opponent" not in weekly.columns or "team" not in weekly.columns:
        return pd.DataFrame()

    needed = ["opponent", "season", "week", "position", "fantasy_points_ppr"]
    if not all(c in weekly.columns for c in needed):
        return pd.DataFrame()

    # Points scored BY players AGAINST each defense
    df = weekly[needed].copy()
    df = df.rename(columns={"opponent": "def_team"})

    # Sum all players of this position who faced this defense this week
    weekly_allowed = (
        df.groupby(["def_team", "season", "week", "position"])["fantasy_points_ppr"]
        .sum()
        .reset_index()
        .rename(columns={"fantasy_points_ppr": "ppr_allowed_this_week"})
    )

    # Cumulative average through prior week within each (def_team, season, position)
    weekly_allowed = weekly_allowed.sort_values(["def_team", "season", "position", "week"])
    g = weekly_allowed.groupby(["def_team", "season", "position"], sort=False)

    weekly_allowed["cum_count"] = g.cumcount()
    weekly_allowed["cum_sum"] = g["ppr_allowed_this_week"].cumsum()

    # Shift so this week sees only *prior* weeks' data
    weekly_allowed["cum_count_prior"] = g["cum_count"].shift(1).fillna(0)
    weekly_allowed["cum_sum_prior"] = g["cum_sum"].shift(1).fillna(0)
    weekly_allowed["opp_ppr_allowed_avg"] = weekly_allowed["cum_sum_prior"] / weekly_allowed[
        "cum_count_prior"
    ].replace(0, float("nan"))

    return weekly_allowed[["def_team", "season", "week", "position", "opp_ppr_allowed_avg"]]
