"""Load and prepare multi-season weekly game data for within-season forecasting."""

from __future__ import annotations

import pandas as pd

from .. import schema
from ..forecasting.source_cache import CACHE_DIR

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


def _cached_weekly_seasons() -> pd.DataFrame:
    """Concat every ``raw_player_weekly_*.parquet`` found in the shared source cache.

    These are raw (pre-``schema.standardize``) frames written by
    ``forecasting.source_cache.cached_loader`` — same shape ``load_player_stats``
    would return, so they can be concatenated with a fresh network pull before
    standardizing.
    """
    frames = []
    for path in sorted(CACHE_DIR.glob("raw_player_weekly_*.parquet")):
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_multi_season_weekly(
    seasons: list[int],
    positions: tuple[str, ...] = FORECAST_POSITIONS,
    season_type: str = "REG",
) -> pd.DataFrame:
    """Weekly game-level stats for multiple seasons, standardized to canonical columns.

    Seasons already present in the local source cache (``data/cache/raw_player_weekly_*.parquet``)
    are read from disk; any remaining seasons (typically the in-progress current season) are
    fetched live via nflreadpy. This makes CV/backtests reproducible offline while still pulling
    fresh data for the season actually being played.

    Returns one row per (player_id, season, week) for regular-season games only
    (unless season_type is overridden).
    """
    cached = _cached_weekly_seasons()
    have_seasons: set[int] = set()
    frames = []

    if not cached.empty and "season" in cached.columns:
        cached_seasons = pd.to_numeric(cached["season"], errors="coerce")
        have_seasons = set(cached_seasons.dropna().astype(int).unique())
        hit = cached[cached_seasons.isin(seasons)]
        if not hit.empty:
            frames.append(hit)

    missing = [s for s in seasons if s not in have_seasons]
    if missing:
        nfl = _import_nflreadpy()
        for yr in missing:
            try:
                frames.append(_to_pandas(nfl.load_player_stats([yr], summary_level="week")))
            except Exception:
                # nflverse doesn't publish a season's weekly file until that
                # season has actually started (e.g. preseason, no games yet).
                continue

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True, sort=False)

    df = schema.standardize(raw, strict=False)
    df = schema.coerce_numeric(df)

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    if season_type and "season_type" in df.columns:
        df = df[df["season_type"] == season_type].copy()

    if positions:
        df = df[df["position"].isin(positions)].copy()

    df = df[df["season"].isin(seasons)].copy()

    dedup_keys = [c for c in ("player_id", "season", "week") if c in df.columns]
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="last")

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
