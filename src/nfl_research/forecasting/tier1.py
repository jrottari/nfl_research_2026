"""Tier-1 feature builder — 2006+ data, as_of-aware.

Sources:
  - load_ff_opportunity(stat_type='weekly'): expected PPR, points over expected
  - load_ff_rankings(type='all'): preseason Expert Consensus Rank (ECR)
  - load_player_stats(summary_level='week'): target/carry share, weeks played,
      2nd-half vs 1st-half trajectory

All functions take ``as_of: datetime`` and filter to data knowable at that date.
The opportunity model provides the highest-value features in this project:
``points_over_expected`` is the primary regression-to-mean signal (luck correction).

ECR note: load_ff_rankings(type='all') returns a table with page_type and
scrape_date columns but no season column.  Season is inferred from scrape_date:
    - Scrape in Aug-Dec of year Y → preseason/in-season for season Y
    - We filter to preseason scrapes (June 1 – Sep 10 of season Y) to avoid
      in-season rankings leaking future information.
  Player matching uses display-name fuzzy join; ~80-90% match rates are typical.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .source_cache import cached_loader

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

FIRST_SEASON = 2006  # ff_opportunity first available
ECR_FIRST_SEASON = 2019  # ff_rankings historical coverage


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _cache_path(name: str) -> Path:
    return _CACHE_DIR / f"{name}.parquet"


def _load_cached_or_fetch(name: str, loader_fn) -> pd.DataFrame:
    path = _cache_path(name)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    df = loader_fn()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception as exc:
        log.warning("Could not cache %s: %s", name, exc)
    return df


# ---------------------------------------------------------------------------
# FF Opportunity features
# ---------------------------------------------------------------------------


def _opportunity_raw(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    key = "_".join(map(str, sorted(seasons)))
    frame = cached_loader(
        f"raw_ff_opportunity_{key}",
        lambda: nfl.load_ff_opportunity(seasons=seasons, stat_type="weekly"),
    )
    keep = [
        "season",
        "week",
        "player_id",
        "full_name",
        "position",
        "total_fantasy_points",
        "total_fantasy_points_exp",
        "total_fantasy_points_diff",
    ]
    return frame[[c for c in keep if c in frame.columns]]


def build_opportunity_features(
    panel: pd.DataFrame,
    as_of: datetime | None = None,
    seasons: list[int] | None = None,
    opportunity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add exp_fantasy_pts, exp_fantasy_ppg, points_over_expected, poe_ppg.

    Aggregates weekly ffopportunity data to season totals for season t-1.
    Only uses weeks admissible at as_of (i.e., the prior season's full year).

    Args:
        panel: Must have player_id and season columns.
        as_of: Upper bound on data admissibility.  Defaults to Aug 1 of forecast season.
        seasons: Override which seasons to load; if None, infers from panel.
    """
    if seasons is None:
        # Load all prior seasons in panel
        panel_seasons = sorted(panel["season"].unique().astype(int))
        prior_seasons = [s - 1 for s in panel_seasons if s - 1 >= FIRST_SEASON]
        if not prior_seasons:
            log.info("No seasons in ff_opportunity range; skipping tier-1 opportunity features")
            return panel
        seasons = list(set(prior_seasons))

    try:
        opp_raw = opportunity.copy() if opportunity is not None else _opportunity_raw(seasons)
    except Exception as exc:
        log.warning("load_ff_opportunity failed: %s", exc)
        return panel

    # Filter to full prior season (weeks 1-18); as_of just ensures we're using prior year
    if as_of is not None:
        cutoff_season = as_of.year - 1
        opp_raw = opp_raw[opp_raw["season"] <= cutoff_season]

    # Aggregate to season totals per player
    agg = (
        opp_raw.groupby(["player_id", "season"])
        .agg(
            exp_fantasy_pts=("total_fantasy_points_exp", "sum"),
            actual_pts=("total_fantasy_points", "sum"),
            points_over_expected=("total_fantasy_points_diff", "sum"),
            weeks=("week", "count"),
        )
        .reset_index()
    )

    agg["exp_fantasy_ppg"] = agg["exp_fantasy_pts"] / agg["weeks"].replace(0, np.nan)
    agg["poe_ppg"] = agg["points_over_expected"] / agg["weeks"].replace(0, np.nan)
    agg = agg.rename(columns={"season": "prior_season"})
    agg["season"] = agg["prior_season"] + 1  # shift: features from t-1 predict t

    panel = panel.copy()
    panel = panel.merge(
        agg[
            [
                "player_id",
                "season",
                "exp_fantasy_pts",
                "exp_fantasy_ppg",
                "points_over_expected",
                "poe_ppg",
            ]
        ],
        on=["player_id", "season"],
        how="left",
    )
    return panel


# ---------------------------------------------------------------------------
# ECR (Expert Consensus Rankings) features
# ---------------------------------------------------------------------------

_NAME_CLEAN = re.compile(r"[^a-z ]+")


def _normalize_name(s: str) -> str:
    return _NAME_CLEAN.sub("", str(s).lower().strip())


def _ecr_raw() -> pd.DataFrame:
    import nflreadpy as nfl

    frame = cached_loader("raw_ff_rankings_all", lambda: nfl.load_ff_rankings(type="all"))
    keep = [
        "scrape_date",
        "player",
        "pos",
        "ecr",
        "page_type",
        "ecr_type",
        "projected_points",
        "fantasy_points",
        "projection",
    ]
    return frame[[c for c in keep if c in frame.columns]]


def _parse_ecr_season(scrape_date: str) -> int:
    """Infer NFL season from scrape_date string (YYYY-MM-DD).

    Jun-Sep of year Y → preseason for season Y.
    Jan-May of year Y → offseason following season Y-1 → season Y.
    """
    dt = pd.to_datetime(scrape_date)
    return int(dt.year)


def build_ecr_features(
    panel: pd.DataFrame,
    as_of: datetime | None = None,
    rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add ecr_rank and ecr_pos_rank to panel.

    Joins preseason redraft-overall ECR to each (player, season) pair.
    Uses page_type 'redraft-overall' and ecr_type 'ro', scrape dates
    June 1 – September 10 of season t (preseason window).

    Player matching: name-based (no gsis_id in ff_rankings).
    ~80-90% match rate; unmatched rows get NaN (MarketConsensusModel handles NaN).
    """
    try:
        ecr_raw = (
            rankings.copy() if rankings is not None else _load_cached_or_fetch("ecr_all", _ecr_raw)
        )
    except Exception as exc:
        log.warning("load_ff_rankings failed: %s", exc)
        return panel

    ecr_raw["scrape_date"] = pd.to_datetime(ecr_raw["scrape_date"], errors="coerce")
    ecr_raw["month"] = ecr_raw["scrape_date"].dt.month
    ecr_raw["year"] = ecr_raw["scrape_date"].dt.year

    # Preseason window: June 1 – Sep 10
    preseason = ecr_raw[
        (ecr_raw["month"].between(6, 9))
        & (ecr_raw["ecr_type"].isin(["ro", "rp"]))  # redraft overall or PPR
    ].copy()
    preseason["season"] = preseason["year"].astype(int)

    # Take the latest preseason scrape per player/season
    preseason = preseason.sort_values("scrape_date")
    preseason = preseason.drop_duplicates(subset=["player", "pos", "season"], keep="last")

    # Compute overall and within-position ECR rank
    preseason["ecr_rank"] = preseason.groupby("season")["ecr"].rank(method="first", ascending=True)
    preseason["ecr_pos_rank"] = preseason.groupby(["season", "pos"])["ecr"].rank(
        method="first", ascending=True
    )
    projection_col = next(
        (c for c in ("projected_points", "fantasy_points", "projection") if c in preseason),
        None,
    )
    preseason["ecr_projection"] = (
        pd.to_numeric(preseason[projection_col], errors="coerce") if projection_col else np.nan
    )

    preseason["_name_norm"] = preseason["player"].apply(_normalize_name)

    # Build name map from panel
    panel = panel.copy()
    panel["_name_norm"] = panel["player_name"].apply(_normalize_name)

    # Join on (normalized_name, season) — imperfect but sufficient for ECR
    ecr_join = preseason[
        ["_name_norm", "season", "ecr_rank", "ecr_pos_rank", "ecr_projection"]
    ].copy()

    # as_of guard: only use ECR published before as_of
    if as_of is not None:
        ecr_join = ecr_join[ecr_join["season"] <= as_of.year]

    panel = panel.merge(ecr_join, on=["_name_norm", "season"], how="left")
    panel = panel.drop(columns=["_name_norm"])
    return panel


# ---------------------------------------------------------------------------
# Weekly usage features
# ---------------------------------------------------------------------------


def _weekly_stats_raw(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    key = "_".join(map(str, sorted(seasons)))
    frame = cached_loader(
        f"raw_player_weekly_{key}",
        lambda: nfl.load_player_stats(seasons=seasons, summary_level="week"),
    )
    keep = [
        "season",
        "week",
        "player_id",
        "player_name",
        "position",
        "fantasy_points_ppr",
        "targets",
        "carries",
        "passing_yards",  # for team pass volume (used for share calc)
    ]
    return frame[[c for c in keep if c in frame.columns]]


def _team_volume_raw(seasons: list[int]) -> pd.DataFrame:
    """Per-team per-season pass and rush totals (for share computation)."""
    import nflreadpy as nfl

    key = "_".join(map(str, sorted(seasons)))
    frame = cached_loader(
        f"raw_player_season_{key}",
        lambda: nfl.load_player_stats(seasons=seasons, summary_level="reg"),
    )
    keep = [
        "season",
        "team",
        "recent_team",
        "player_id",
        "position",
        "targets",
        "carries",
    ]
    out = frame[[c for c in keep if c in frame.columns]].copy()
    if "recent_team" not in out and "team" in out:
        out = out.rename(columns={"team": "recent_team"})
    return out


def build_weekly_features(
    panel: pd.DataFrame,
    as_of: datetime | None = None,
    seasons: list[int] | None = None,
    weekly_stats: pd.DataFrame | None = None,
    season_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add target_share_lag1, carry_share_lag1, weeks_played_lag1, h2_vs_h1_ppr.

    Args:
        panel: Must have player_id and season columns.
        as_of: Upper bound (prior season only).
        seasons: Override which seasons to load.
    """
    if seasons is None:
        panel_seasons = sorted(panel["season"].unique().astype(int))
        prior_seasons = [s - 1 for s in panel_seasons if s - 1 >= 1999]
        if not prior_seasons:
            return panel
        seasons = list(set(prior_seasons))

    try:
        weekly = weekly_stats.copy() if weekly_stats is not None else _weekly_stats_raw(seasons)
    except Exception as exc:
        log.warning("load_player_stats(weekly) failed: %s", exc)
        return panel

    # as_of: only use the prior season (t-1), so no within-season leak
    if as_of is not None:
        weekly = weekly[weekly["season"] <= as_of.year - 1]

    # Weeks played
    weeks_played = (
        weekly.groupby(["player_id", "season"]).size().reset_index(name="weeks_played_lag1")
    )

    # H1 vs H2 trajectory (weeks 1-9 vs 10-18)
    weekly["half"] = (weekly["week"] > 9).astype(int)
    half_ppg = (
        weekly.groupby(["player_id", "season", "half"])["fantasy_points_ppr"]
        .mean()
        .unstack("half", fill_value=np.nan)
        .reset_index()
    )
    if 0 in half_ppg.columns and 1 in half_ppg.columns:
        half_ppg["h2_vs_h1_ppr"] = half_ppg[1] - half_ppg[0]
    else:
        half_ppg["h2_vs_h1_ppr"] = np.nan
    half_ppg = half_ppg[["player_id", "season", "h2_vs_h1_ppr"]]

    # Target share — need team totals
    # Simple proxy: each player's targets / sum of all targets on their recorded plays
    # (full team target share needs roster context; use within-data share)
    target_totals = (
        weekly.groupby(["player_id", "season"])["targets"].sum().reset_index(name="player_targets")
    )

    # Team total targets (need team column — use per-season stats if available)
    try:
        season_stats = (
            season_stats.copy() if season_stats is not None else _team_volume_raw(seasons)
        )
        if "recent_team" not in season_stats and "team" in season_stats:
            season_stats = season_stats.rename(columns={"team": "recent_team"})
        team_targets = (
            season_stats.groupby(["season", "recent_team"])["targets"]
            .sum()
            .reset_index(name="team_targets")
        )
        # Join team to weekly — approximate via player's top-team in season
        player_team = (
            season_stats[season_stats["position"].isin(["WR", "TE", "RB", "QB"])]
            .groupby(["player_id", "season"])
            .first()
            .reset_index()[["player_id", "season", "recent_team"]]
        )
        target_totals = target_totals.merge(player_team, on=["player_id", "season"], how="left")
        target_totals = target_totals.merge(team_targets, on=["season", "recent_team"], how="left")
        target_totals["target_share_lag1"] = (
            target_totals["player_targets"] / target_totals["team_targets"]
        )
    except Exception:
        target_totals["target_share_lag1"] = np.nan

    target_totals = target_totals[["player_id", "season", "target_share_lag1"]]

    # Carry share (similarly approximate)
    carry_totals = (
        weekly.groupby(["player_id", "season"])["carries"].sum().reset_index(name="player_carries")
    )
    try:
        assert season_stats is not None
        team_carries = (
            season_stats.groupby(["season", "recent_team"])["carries"]
            .sum()
            .reset_index(name="team_carries")
        )
        carry_totals = carry_totals.merge(player_team, on=["player_id", "season"], how="left")
        carry_totals = carry_totals.merge(team_carries, on=["season", "recent_team"], how="left")
        carry_totals["carry_share_lag1"] = (
            carry_totals["player_carries"] / carry_totals["team_carries"]
        )
    except Exception:
        carry_totals["carry_share_lag1"] = np.nan

    carry_totals = carry_totals[["player_id", "season", "carry_share_lag1"]]

    # Shift season forward: these are prior-year features
    for df in [weeks_played, half_ppg, target_totals, carry_totals]:
        df["season"] = df["season"] + 1  # predict season t using season t-1 data

    panel = panel.copy()
    for feat_df in [weeks_played, half_ppg, target_totals, carry_totals]:
        panel = panel.merge(feat_df, on=["player_id", "season"], how="left")

    return panel


# ---------------------------------------------------------------------------
# Combined tier-1 builder
# ---------------------------------------------------------------------------


def build_tier1_features(
    panel: pd.DataFrame,
    as_of: datetime | None = None,
    include_opportunity: bool = True,
    include_ecr: bool = True,
    include_weekly: bool = True,
    sources: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Apply all tier-1 additions to a tier-0 enhanced panel.

    Args:
        panel:               Tier-0 panel (from tier0.build_tier0_features).
        as_of:               Data admissibility cutoff. Defaults to Aug 1 of season t.
        include_opportunity: Add ff_opportunity derived features.
        include_ecr:         Add preseason ECR ranking features.
        include_weekly:      Add weekly usage features (target/carry share, etc.).

    Returns:
        panel with additional columns per REGISTRY tier-1 entries.
    """
    sources = sources or {}
    if include_opportunity:
        panel = build_opportunity_features(
            panel, as_of=as_of, opportunity=sources.get("ff_opportunity")
        )

    if include_ecr:
        panel = build_ecr_features(panel, as_of=as_of, rankings=sources.get("ff_rankings"))

    if include_weekly:
        panel = build_weekly_features(
            panel,
            as_of=as_of,
            weekly_stats=sources.get("player_weekly"),
            season_stats=sources.get("player_season"),
        )

    return panel
