"""Thin wrappers around nflreadpy.

nflreadpy returns **Polars** DataFrames. This module is the one and only place
that converts to pandas, so if you ever want to go Polars-native you rewrite
these three functions and the metrics modules, and nothing else.

nflreadpy already caches downloads on the filesystem, so there is deliberately
no second raw-data cache here. Derived boards get cached in ``data/processed``
by :mod:`nfl_research.pipeline`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import schema


def _to_pandas(df: Any) -> pd.DataFrame:
    """Convert a Polars frame to pandas; pass a pandas frame straight through."""
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        # Polars -> pandas needs pyarrow installed.
        return df.to_pandas()
    raise TypeError(f"Don't know how to convert {type(df)!r} to a pandas DataFrame")


def _import_nflreadpy():
    try:
        import nflreadpy as nfl
    except ModuleNotFoundError as err:  # pragma: no cover - env problem, not logic
        raise ModuleNotFoundError(
            "nflreadpy is not installed. Run `uv add nflreadpy` or "
            "`pip install 'nflreadpy[pandas]'`."
        ) from err
    return nfl


def load_weekly(
    seasons: int | list[int],
    *,
    season_type: str | None = "REG",
    positions: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Week-level player stats, renamed to canonical columns.

    Args:
        seasons: One season or a list of them.
        season_type: Keep only this season type ("REG", "POST"), or None for all.
        positions: Optional position filter, e.g. ``("QB", "RB", "WR", "TE")``.
    """
    nfl = _import_nflreadpy()
    raw = _to_pandas(nfl.load_player_stats(seasons, summary_level="week"))

    df = schema.standardize(raw)
    df = schema.coerce_numeric(df)

    if season_type is not None and "season_type" in df.columns:
        df = df[df["season_type"] == season_type]
    if positions is not None:
        df = df[df["position"].isin(positions)]

    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    return df.reset_index(drop=True)


def load_season_totals(
    seasons: int | list[int],
    *,
    summary_level: str = "reg",
    positions: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Season-level player stats.

    nflverse aggregates these server-side, which is why this repo no longer
    hand-rolls a groupby-sum. ``summary_level`` is one of "reg", "post" or
    "reg+post".
    """
    if summary_level not in {"reg", "post", "reg+post"}:
        raise ValueError('summary_level must be "reg", "post" or "reg+post"')

    nfl = _import_nflreadpy()
    raw = _to_pandas(nfl.load_player_stats(seasons, summary_level=summary_level))

    df = schema.standardize(raw)
    df = schema.coerce_numeric(df)

    if positions is not None:
        df = df[df["position"].isin(positions)]

    return df.reset_index(drop=True)


def load_schedule(seasons: int | list[int]) -> pd.DataFrame:
    """Game schedule - useful for bye weeks and strength-of-schedule work."""
    nfl = _import_nflreadpy()
    return _to_pandas(nfl.load_schedules(seasons))
