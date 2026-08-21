"""Feature registry — the authoritative column-to-tier manifest.

Each entry maps a column name to a ColumnSpec describing:
  tier          : int  — which data tier provides this column (0-4)
  first_season  : int  — first season where data is available
  source        : str  — nflreadpy loader that provides the raw data
  description   : str  — one-line description for documentation

The evaluation harness and tier-aware models read tier membership from
this registry rather than from hardcoded lists.

Columns absent from the registry are treated as ID / target columns and
are excluded from all model feature vectors.
"""

from __future__ import annotations

from typing import NamedTuple


class ColumnSpec(NamedTuple):
    tier: int
    first_season: int
    source: str
    description: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ColumnSpec] = {
    # ------------------------------------------------------------------
    # Tier 0 — 1999+ (lag history + basic derived features)
    # ------------------------------------------------------------------
    "points_ppr_lag1": ColumnSpec(
        0, 1999, "load_player_stats", "PPR fantasy points, prior season (lag 1)"
    ),
    "points_ppr_lag2": ColumnSpec(
        0, 1999, "load_player_stats", "PPR fantasy points, 2 seasons prior (lag 2)"
    ),
    "points_ppr_lag3": ColumnSpec(
        0, 1999, "load_player_stats", "PPR fantasy points, 3 seasons prior (lag 3)"
    ),
    "ppg_lag1": ColumnSpec(0, 1999, "load_player_stats", "PPR points per game, prior season"),
    "ppg_lag2": ColumnSpec(0, 1999, "load_player_stats", "PPR points per game, 2 seasons prior"),
    "games_lag1": ColumnSpec(0, 1999, "load_player_stats", "Games played, prior season"),
    "games_lag2": ColumnSpec(0, 1999, "load_player_stats", "Games played, 2 seasons prior"),
    "trend_1": ColumnSpec(0, 1999, "derived", "1-year PPR change: lag1 - lag2"),
    "trend_2": ColumnSpec(0, 1999, "derived", "2-year PPR change: lag2 - lag3"),
    "career_season": ColumnSpec(0, 1999, "derived", "Number of prior seasons in data (0 = rookie)"),
    "exp_smooth": ColumnSpec(
        0, 1999, "derived", "Exponentially-weighted blend of lag1/2/3 (weights 0.5/0.3/0.2)"
    ),
    "pos_code": ColumnSpec(0, 1999, "derived", "Integer position code: QB=0, RB=1, WR=2, TE=3"),
    # Missing-lag indicators — crucial for rookies / returns from injury
    "lag1_missing": ColumnSpec(0, 1999, "derived", "1 if points_ppr_lag1 was NaN before fillna"),
    "lag2_missing": ColumnSpec(0, 1999, "derived", "1 if points_ppr_lag2 was NaN before fillna"),
    "lag3_missing": ColumnSpec(0, 1999, "derived", "1 if points_ppr_lag3 was NaN before fillna"),
    # Age features (birth date from load_players)
    "age": ColumnSpec(0, 1999, "load_players", "Player age at season start (Aug 1)"),
    "age_squared": ColumnSpec(0, 1999, "load_players", "age^2 for non-linear career-arc effects"),
    "is_rookie": ColumnSpec(
        0, 1999, "derived", "1 if career_season == 0 (first NFL season in data)"
    ),
    # Draft capital (1980+ but we key off player's draft year)
    "draft_pick": ColumnSpec(
        0, 1980, "load_draft_picks", "Overall draft pick number (NaN = undrafted)"
    ),
    "draft_round": ColumnSpec(0, 1980, "load_draft_picks", "Draft round (NaN = undrafted)"),
    "is_undrafted": ColumnSpec(0, 1980, "load_draft_picks", "1 if player went undrafted"),
    # ------------------------------------------------------------------
    # Tier 1 — 2006+  (opportunity model + market rankings + usage)
    # ------------------------------------------------------------------
    "exp_fantasy_pts": ColumnSpec(
        1,
        2006,
        "load_ff_opportunity",
        "Expected PPR fantasy points from ffverse opportunity model, prior season",
    ),
    "exp_fantasy_ppg": ColumnSpec(
        1, 2006, "load_ff_opportunity", "Expected PPR fantasy points per game, prior season"
    ),
    "points_over_expected": ColumnSpec(
        1,
        2006,
        "load_ff_opportunity",
        "Actual PPR - expected PPR (primary regression-to-mean signal)",
    ),
    "poe_ppg": ColumnSpec(
        1, 2006, "load_ff_opportunity", "Points over expected per game, prior season"
    ),
    # FantasyPros ECR preseason consensus
    "ecr_rank": ColumnSpec(
        1, 2019, "load_ff_rankings", "Preseason FantasyPros Expert Consensus Rank for this season"
    ),
    "ecr_pos_rank": ColumnSpec(
        1, 2019, "load_ff_rankings", "Preseason ECR within position (1=best at that position)"
    ),
    "ecr_projection": ColumnSpec(
        1, 2019, "load_ff_rankings", "Preseason FantasyPros projected PPR points for this season"
    ),
    # Weekly usage from player stats
    "target_share_lag1": ColumnSpec(
        1, 1999, "load_player_stats", "Targets as fraction of team pass attempts, prior season"
    ),
    "carry_share_lag1": ColumnSpec(
        1, 1999, "load_player_stats", "Carries as fraction of team rush attempts, prior season"
    ),
    "weeks_played_lag1": ColumnSpec(
        1, 1999, "load_player_stats", "Number of weeks with a stat line, prior season"
    ),
    "h2_vs_h1_ppr": ColumnSpec(
        1, 1999, "load_player_stats", "2nd-half PPG minus 1st-half PPG (trajectory within season)"
    ),
    # ------------------------------------------------------------------
    # Tier 2 — 2012+  (snap counts + depth + schedule + PBP)
    # ------------------------------------------------------------------
    "snap_pct_lag1": ColumnSpec(
        2, 2012, "load_snap_counts", "Mean offensive snap share, prior season"
    ),
    "snap_pct_trend": ColumnSpec(2, 2012, "load_snap_counts", "Snap share change: lag1 - lag2"),
    "wks_above_50pct_snaps": ColumnSpec(
        2, 2012, "load_snap_counts", "Weeks with >50% snap share, prior season"
    ),
    "wks_above_75pct_snaps": ColumnSpec(
        2, 2012, "load_snap_counts", "Weeks with >75% snap share, prior season"
    ),
    "depth_chart_pos": ColumnSpec(
        2,
        2012,
        "load_depth_charts",
        "Depth chart position code as of Aug 1 (0=starter, 1=backup, ...)",
    ),
    "vegas_team_total": ColumnSpec(
        2, 2012, "load_schedules", "Season-aggregated Vegas team total (proxy for pass volume)"
    ),
    "team_pass_roe": ColumnSpec(
        2, 2012, "load_schedules", "Team pass rate over expected from prior season"
    ),
    "air_yards_share": ColumnSpec(
        2, 2012, "load_pbp", "Player air yards / team total air yards, prior season"
    ),
    "wopr": ColumnSpec(
        2, 2012, "load_pbp", "Weighted opportunity rating (targets + air yards share), prior season"
    ),
    "rz_target_share": ColumnSpec(2, 2012, "load_pbp", "Red-zone target share, prior season"),
    "rz_carry_share": ColumnSpec(2, 2012, "load_pbp", "Red-zone carry share, prior season"),
    "changed_team": ColumnSpec(
        2, 2012, "load_rosters", "1 if player changed teams between T-1 and as_of date"
    ),
    "new_starting_qb": ColumnSpec(
        2, 2012, "load_rosters", "1 if player's team has a new starting QB this season"
    ),
    "competition_draft_capital": ColumnSpec(
        2, 2012, "load_draft_picks", "Sum of (33-round) for same-position draftees on same team"
    ),
    "competition_count": ColumnSpec(
        2,
        2012,
        "load_draft_picks/load_rosters/load_trades",
        "New same-position teammates added before the forecast cutoff",
    ),
    "new_offensive_coordinator": ColumnSpec(
        2, 2012, "load_rosters", "Team changed offensive coordinator before the season"
    ),
    # ------------------------------------------------------------------
    # Tier 3 — 2016+  (NextGen Stats)
    # ------------------------------------------------------------------
    "separation_avg": ColumnSpec(
        3,
        2016,
        "load_nextgen_stats",
        "Average separation at time of target (receivers) / catch point",
    ),
    "cushion_avg": ColumnSpec(
        3, 2016, "load_nextgen_stats", "Average cushion from nearest defender at snap (receivers)"
    ),
    "intended_air_yards": ColumnSpec(
        3, 2016, "load_nextgen_stats", "Average intended air yards per target (receivers)"
    ),
    "rush_yards_oe": ColumnSpec(
        3, 2016, "load_nextgen_stats", "Rush yards over expected per attempt"
    ),
    "efficiency": ColumnSpec(3, 2016, "load_nextgen_stats", "NextGen rushing efficiency score"),
    "time_to_los": ColumnSpec(
        3, 2016, "load_nextgen_stats", "Average time for a rusher to cross the line of scrimmage"
    ),
    "completion_pct_oe": ColumnSpec(
        3, 2016, "load_nextgen_stats", "Completion percentage over expectation for passers"
    ),
    "participation_rate": ColumnSpec(
        3, 2016, "load_participation", "Prior-season share of team offensive plays participated in"
    ),
    # ------------------------------------------------------------------
    # Tier 4 — 2018+/2022+  (PFR advanced + FTN + injuries + contracts)
    # ------------------------------------------------------------------
    "broken_tackles": ColumnSpec(
        4, 2018, "load_pfr_advstats", "Broken tackles per attempt (rushing/receiving)"
    ),
    "drop_rate": ColumnSpec(
        4, 2018, "load_pfr_advstats", "Drop rate (targets that resulted in a drop / total targets)"
    ),
    "yac_per_rec": ColumnSpec(4, 2018, "load_pfr_advstats", "Yards after catch per reception"),
    "pressure_rate": ColumnSpec(4, 2018, "load_pfr_advstats", "QB pressure rate (passers only)"),
    "contract_guaranteed": ColumnSpec(
        4, 2018, "load_contracts", "Guaranteed money in current contract (before as_of date)"
    ),
    "contract_years_remaining": ColumnSpec(
        4, 2018, "load_contracts", "Years remaining on contract as of as_of date"
    ),
    "games_missed_lag1": ColumnSpec(
        4, 2009, "load_injuries", "Games missed due to injury in prior season"
    ),
    "injury_weeks_lag1": ColumnSpec(
        4, 2009, "load_injuries", "Weeks on injury report in prior season"
    ),
    "ftn_xyac": ColumnSpec(
        4, 2022, "load_ftn_charting", "Expected YAC from FTN charting (CC-BY-SA licensed)"
    ),
    "ftn_no_huddle_rate": ColumnSpec(
        4, 2022, "load_ftn_charting", "No-huddle snap rate from FTN charting"
    ),
}


def cols_for_tier(max_tier: int) -> list[str]:
    """Return all columns available up through max_tier, sorted."""
    return sorted(col for col, spec in REGISTRY.items() if spec.tier <= max_tier)


def tier_first_season(tier: int) -> int:
    """Return the first season where all columns in this exact tier are available."""
    tier_cols = [spec for col, spec in REGISTRY.items() if spec.tier == tier]
    if not tier_cols:
        return 1999
    return max(spec.first_season for spec in tier_cols)


def cumulative_first_season(max_tier: int) -> int:
    """First season where all columns through max_tier are available."""
    relevant = [spec for col, spec in REGISTRY.items() if spec.tier <= max_tier]
    if not relevant:
        return 1999
    return max(spec.first_season for spec in relevant)
