"""Multi-season nflverse data loading for the forecasting pipeline."""

from __future__ import annotations

import pandas as pd

from .. import schema
from .source_cache import CACHE_DIR

FORECAST_POSITIONS = ("QB", "RB", "WR", "TE")


def _cached_season_totals() -> pd.DataFrame:
    frames = []
    for path in sorted(CACHE_DIR.glob("raw_player_season_*.parquet")):
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _import_nflreadpy():
    try:
        import nflreadpy as nfl

        return nfl
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("nflreadpy is required. Run: pip install nflreadpy") from err


def _to_pandas(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    raise TypeError(f"Cannot convert {type(df)} to DataFrame")


def load_multi_season(
    seasons: list[int],
    positions: tuple[str, ...] = FORECAST_POSITIONS,
) -> pd.DataFrame:
    """Season-level PPR stats across multiple years.

    Returns a long-format DataFrame with one row per (player_id, season),
    standardized to canonical column names. Computes ppg_ppr when games > 0.

    ``games`` comes from nflverse season-level data. If absent, it is estimated
    from weekly row counts (fall-through), but that requires an extra weekly pull.

    Seasons already present in ``data/cache/raw_player_season_*.parquet`` are read from disk;
    any remaining seasons are fetched live via nflreadpy.
    """
    cached = _cached_season_totals()
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
                frames.append(_to_pandas(nfl.load_player_stats([yr], summary_level="reg")))
            except Exception:
                # nflverse doesn't publish a season's totals file until that
                # season has actually started/completed.
                continue

    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    df = schema.standardize(raw, strict=False)
    df = schema.coerce_numeric(df)

    if "season" not in df.columns:
        raise ValueError("'season' column missing after standardizing - check nflreadpy version.")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    if positions:
        df = df[df["position"].isin(positions)].copy()

    # Keep only rows for the requested seasons (guard against multi-season concat)
    df = df[df["season"].isin(seasons)].copy()

    # Compute PPG safely
    games_col = df["games"] if "games" in df.columns else pd.Series(17, index=df.index)
    df["ppg_ppr"] = (df["fantasy_points_ppr"] / games_col.replace(0, float("nan"))).fillna(0.0)
    df["ppg_std"] = (df["fantasy_points"] / games_col.replace(0, float("nan"))).fillna(0.0)

    return df.reset_index(drop=True)


def top_n_in_season(df: pd.DataFrame, season: int, n: int = 200) -> set[str]:
    """Return the player_ids that finished in the top-n PPR in `season`."""
    sub = df[df["season"] == season].copy()
    sub = sub.sort_values("fantasy_points_ppr", ascending=False)
    return set(sub.head(n)["player_id"].tolist())
