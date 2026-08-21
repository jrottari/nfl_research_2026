"""Raw nflreadpy cache helpers used by tier feature builders.

The cache is deliberately written while the object is still a Polars frame.
That preserves nflreadpy's native types and makes an already-fetched build
reproducible without a network connection.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


def cached_loader(
    name: str, loader: Callable[[], Any], cache_dir: Path = CACHE_DIR
) -> pd.DataFrame:
    """Load a raw frame from parquet, or fetch, cache, and convert to pandas."""
    path = cache_dir / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    raw = loader()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(raw, "write_parquet"):
        raw.write_parquet(path)
        return raw.to_pandas()
    if isinstance(raw, pd.DataFrame):
        raw.to_parquet(path, index=False)
        return raw.copy()
    if hasattr(raw, "to_pandas"):
        frame = raw.to_pandas()
        frame.to_parquet(path, index=False)
        return frame
    raise TypeError(f"Unsupported loader result: {type(raw)!r}")


def as_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    raise TypeError(f"Unsupported frame: {type(frame)!r}")


def load_tier_sources(
    seasons: list[int],
    *,
    max_tier: int = 4,
    include_pbp: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch and raw-cache every nflreadpy source required through a tier."""
    import nflreadpy as nfl

    years = sorted(set(int(year) for year in seasons))
    key = f"{years[0]}_{years[-1]}"
    sources: dict[str, pd.DataFrame] = {
        "players": cached_loader("raw_players", nfl.load_players),
        "draft_picks": cached_loader("raw_draft_picks", nfl.load_draft_picks),
    }
    if max_tier >= 1:
        sources.update(
            {
                "ff_opportunity": cached_loader(
                    f"raw_ff_opportunity_{key}",
                    lambda: nfl.load_ff_opportunity(
                        seasons=[year for year in years if year >= 2006], stat_type="weekly"
                    ),
                ),
                "ff_rankings": cached_loader(
                    "raw_ff_rankings_all", lambda: nfl.load_ff_rankings(type="all")
                ),
                "player_weekly": cached_loader(
                    f"raw_player_weekly_{key}",
                    lambda: nfl.load_player_stats(seasons=years, summary_level="week"),
                ),
                "player_season": cached_loader(
                    f"raw_player_season_{key}",
                    lambda: nfl.load_player_stats(seasons=years, summary_level="reg"),
                ),
            }
        )
    if max_tier >= 2:
        tier2_years = [year for year in years if year >= 2012]
        sources.update(
            {
                "snap_counts": cached_loader(
                    f"raw_snap_counts_{key}", lambda: nfl.load_snap_counts(tier2_years)
                ),
                "depth_charts": cached_loader(
                    f"raw_depth_charts_{key}", lambda: nfl.load_depth_charts(tier2_years)
                ),
                "schedules": cached_loader(
                    f"raw_schedules_{key}", lambda: nfl.load_schedules(tier2_years)
                ),
                "rosters": cached_loader(
                    f"raw_rosters_weekly_{key}", lambda: nfl.load_rosters_weekly(tier2_years)
                ),
                "trades": cached_loader("raw_trades", nfl.load_trades),
            }
        )
        if include_pbp:
            sources["pbp"] = cached_loader(f"raw_pbp_{key}", lambda: nfl.load_pbp(tier2_years))
    if max_tier >= 3:
        tier3_years = [year for year in years if year >= 2016]
        for stat_type in ("receiving", "rushing", "passing"):
            sources[f"ngs_{stat_type}"] = cached_loader(
                f"raw_ngs_{stat_type}_{key}",
                partial(nfl.load_nextgen_stats, tier3_years, stat_type=stat_type),
            )
        try:
            sources["participation"] = cached_loader(
                f"raw_participation_{key}", lambda: nfl.load_participation(tier3_years)
            )
        except Exception:
            # nflverse publishes this source only after the season is complete.
            sources["participation"] = pd.DataFrame()
    if max_tier >= 4:
        tier4_years = [year for year in years if year >= 2018]
        pfr_parts = []
        for stat_type in ("rec", "rush", "pass"):
            pfr_parts.append(
                cached_loader(
                    f"raw_pfr_{stat_type}_{key}",
                    partial(
                        nfl.load_pfr_advstats,
                        tier4_years,
                        stat_type=stat_type,
                        summary_level="season",
                    ),
                )
            )
        sources["pfr"] = pd.concat(pfr_parts, ignore_index=True, sort=False)
        sources["ftn"] = cached_loader(
            f"raw_ftn_{key}",
            lambda: nfl.load_ftn_charting([year for year in years if year >= 2022]),
        )
        sources["injuries"] = cached_loader(
            f"raw_injuries_{key}",
            lambda: nfl.load_injuries([year for year in years if year >= 2009]),
        )
        sources["contracts"] = cached_loader("raw_contracts", nfl.load_contracts)
    return sources
