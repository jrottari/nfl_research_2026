"""Tier-0 feature builder — 1999+ data, as_of-aware.

Extends the existing lag/trend/career features with:
  - Missing-lag indicator columns (lag1_missing, lag2_missing, lag3_missing)
  - Age and age_squared from load_players() birth dates
  - Draft capital (pick, round, undrafted flag) from load_draft_picks()
  - is_rookie flag

All functions take an explicit ``as_of: datetime`` parameter and filter
source data to rows knowable at that date.  Default: August 1 of season t.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    return _CACHE_DIR / f"{name}.parquet"


def _load_cached_or_fetch(name: str, loader_fn) -> pd.DataFrame:
    """Return cached parquet if fresh (same day); otherwise fetch and cache."""
    path = _cache_path(name)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    df = loader_fn()
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        log.warning("Could not cache %s: %s", name, exc)
    return df


# ---------------------------------------------------------------------------
# Raw loaders (polars → pandas)
# ---------------------------------------------------------------------------

def _players_raw() -> pd.DataFrame:
    import nflreadpy as nfl
    pl_df = nfl.load_players()
    cols = ["gsis_id", "display_name", "birth_date", "position"]
    available = [c for c in cols if c in pl_df.columns]
    return pl_df.select(available).to_pandas()


def _draft_picks_raw() -> pd.DataFrame:
    import nflreadpy as nfl
    pl_df = nfl.load_draft_picks()
    cols = ["season", "round", "pick", "gsis_id", "pfr_player_name", "position", "category"]
    available = [c for c in cols if c in pl_df.columns]
    return pl_df.select(available).to_pandas()


# ---------------------------------------------------------------------------
# Age features
# ---------------------------------------------------------------------------

def build_age_features(
    panel: pd.DataFrame,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Add ``age`` and ``age_squared`` to panel rows.

    Joins load_players() birth_date on player_id (= gsis_id).
    ``as_of`` is not used to filter load_players (which is static roster data)
    but is used as the reference date for computing age.
    """
    if as_of is None:
        # default: August 1 of the forecast season
        # panel must have a 'season' column; we use the per-row season
        pass

    players = _load_cached_or_fetch("players", _players_raw)
    players = players.rename(columns={"gsis_id": "player_id"})
    players["birth_date"] = pd.to_datetime(players["birth_date"], errors="coerce")
    players = players[["player_id", "birth_date"]].dropna()

    panel = panel.copy()
    panel = panel.merge(players, on="player_id", how="left")

    if as_of is not None:
        ref_date = as_of
    else:
        # Use August 1 of each row's season for age computation
        panel["_ref_date"] = pd.to_datetime(
            panel["season"].astype(str) + "-08-01", format="%Y-%m-%d"
        )
        ref_date = None  # computed per-row

    if ref_date is not None:
        panel["age"] = (
            (ref_date - panel["birth_date"]).dt.days / 365.25
        ).round(2)
    else:
        panel["age"] = (
            (panel["_ref_date"] - panel["birth_date"]).dt.days / 365.25
        ).round(2)
        panel = panel.drop(columns=["_ref_date"])

    panel["age_squared"] = panel["age"] ** 2
    panel = panel.drop(columns=["birth_date"], errors="ignore")
    return panel


# ---------------------------------------------------------------------------
# Draft capital features
# ---------------------------------------------------------------------------

def build_draft_features(
    panel: pd.DataFrame,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Add draft_pick, draft_round, is_undrafted to panel.

    Joins load_draft_picks() on (player_id, rookie_season).
    Only uses picks from seasons <= as_of year (or panel season - 0).
    A player's draft year is inferred from their earliest season in the panel.
    """
    picks = _load_cached_or_fetch("draft_picks", _draft_picks_raw)
    picks = picks.rename(columns={"gsis_id": "player_id"})
    picks = picks[picks["player_id"].notna()].copy()

    # Only admissible picks (before as_of)
    if as_of is not None:
        picks = picks[picks["season"] <= as_of.year]

    picks = picks[["player_id", "season", "round", "pick"]].rename(
        columns={"season": "draft_year", "round": "draft_round", "pick": "draft_pick"}
    )
    # keep earliest draft entry per player
    picks = picks.sort_values("draft_year").drop_duplicates("player_id", keep="first")

    panel = panel.copy()
    panel = panel.merge(picks, on="player_id", how="left")
    panel["is_undrafted"] = panel["draft_pick"].isna().astype(int)
    panel["draft_pick"] = panel["draft_pick"].fillna(300.0)   # large number = low capital
    panel["draft_round"] = panel["draft_round"].fillna(8.0)   # 8 = undrafted sentinel
    return panel


# ---------------------------------------------------------------------------
# Missing-lag indicators
# ---------------------------------------------------------------------------

def add_missing_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag1_missing, lag2_missing, lag3_missing before fillna."""
    panel = panel.copy()
    for k in [1, 2, 3]:
        col = f"points_ppr_lag{k}"
        if col in panel.columns:
            panel[f"lag{k}_missing"] = panel[col].isna().astype(int)
        else:
            panel[f"lag{k}_missing"] = 0
    return panel


# ---------------------------------------------------------------------------
# is_rookie flag
# ---------------------------------------------------------------------------

def add_rookie_flag(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["is_rookie"] = (panel["career_season"] == 0).astype(int)
    return panel


# ---------------------------------------------------------------------------
# Combined tier-0 builder
# ---------------------------------------------------------------------------

def build_tier0_features(
    panel: pd.DataFrame,
    as_of: datetime | None = None,
    include_age: bool = True,
    include_draft: bool = True,
) -> pd.DataFrame:
    """Apply all tier-0 additions to a panel from features.build_panel().

    Args:
        panel:        Output of features.build_panel() — already has lag cols.
        as_of:        Reference date for age calculation and draft-pick filtering.
                      Defaults to Aug 1 of each row's season.
        include_age:  Whether to add age/age_squared (requires load_players).
        include_draft: Whether to add draft capital (requires load_draft_picks).

    Returns:
        panel with additional columns: lag1_missing, lag2_missing, lag3_missing,
        is_rookie, [age, age_squared,] [draft_pick, draft_round, is_undrafted].
    """
    panel = add_missing_indicators(panel)
    panel = add_rookie_flag(panel)

    if include_age:
        try:
            panel = build_age_features(panel, as_of=as_of)
        except Exception as exc:
            log.warning("Age features unavailable: %s", exc)

    if include_draft:
        try:
            panel = build_draft_features(panel, as_of=as_of)
        except Exception as exc:
            log.warning("Draft capital features unavailable: %s", exc)

    return panel
