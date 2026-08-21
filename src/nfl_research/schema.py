"""Canonical column names for the nflverse player-stats schema.

nflverse rebuilt player stats on top of ``nflfastR::calculate_stats()``. Several
columns were renamed relative to the old ``nfl_data_py`` weekly file, and the
file keeps growing (the docs show 115 columns in one build and 145 in a later
one). Known renames so far:

    recent_team          -> team
    interceptions        -> passing_interceptions
    sacks                -> sacks_suffered
    sack_yards           -> sack_yards_lost

Rather than hard-code names in every module, everything downstream asks for a
*canonical* name and this module maps it onto whatever the loaded file actually
uses. When nflverse renames something again, add an alias here and nothing else
in the repo has to change.
"""

from __future__ import annotations

import pandas as pd

# canonical name -> candidate source names, in priority order
ALIASES: dict[str, tuple[str, ...]] = {
    # --- identity -----------------------------------------------------------
    "player_id": ("player_id", "gsis_id"),
    "player_name": ("player_display_name", "player_name", "full_name"),
    "position": ("position",),
    "position_group": ("position_group",),
    "team": ("team", "recent_team"),
    "opponent": ("opponent_team", "opponent"),
    "season": ("season",),
    "week": ("week",),
    "season_type": ("season_type",),
    "game_id": ("game_id",),
    "games": ("games", "g"),
    # --- passing ------------------------------------------------------------
    "completions": ("completions", "passing_completions"),
    "attempts": ("attempts", "passing_attempts"),
    "passing_yards": ("passing_yards",),
    "passing_tds": ("passing_tds",),
    "interceptions": ("passing_interceptions", "interceptions"),
    "sacks_taken": ("sacks_suffered", "sacks"),
    "sack_yards": ("sack_yards_lost", "sack_yards"),
    "sack_fumbles": ("sack_fumbles",),
    "sack_fumbles_lost": ("sack_fumbles_lost",),
    "passing_air_yards": ("passing_air_yards",),
    "passing_yac": ("passing_yards_after_catch",),
    "passing_first_downs": ("passing_first_downs",),
    "passing_2pt": ("passing_2pt_conversions",),
    "passing_epa": ("passing_epa",),
    # --- rushing ------------------------------------------------------------
    "carries": ("carries", "rushing_attempts"),
    "rushing_yards": ("rushing_yards",),
    "rushing_tds": ("rushing_tds",),
    "rushing_fumbles": ("rushing_fumbles",),
    "rushing_fumbles_lost": ("rushing_fumbles_lost",),
    "rushing_first_downs": ("rushing_first_downs",),
    "rushing_2pt": ("rushing_2pt_conversions",),
    "rushing_epa": ("rushing_epa",),
    # --- receiving ----------------------------------------------------------
    "receptions": ("receptions",),
    "targets": ("targets",),
    "receiving_yards": ("receiving_yards",),
    "receiving_tds": ("receiving_tds",),
    "receiving_fumbles": ("receiving_fumbles",),
    "receiving_fumbles_lost": ("receiving_fumbles_lost",),
    "receiving_air_yards": ("receiving_air_yards",),
    "receiving_yac": ("receiving_yards_after_catch",),
    "receiving_first_downs": ("receiving_first_downs",),
    "receiving_2pt": ("receiving_2pt_conversions",),
    "receiving_epa": ("receiving_epa",),
    "target_share": ("target_share", "tgt_share"),
    "air_yards_share": ("air_yards_share",),
    "wopr": ("wopr", "wopr_x", "wopr_y"),
    # --- misc / scoring -----------------------------------------------------
    "special_teams_tds": ("special_teams_tds",),
    "fantasy_points": ("fantasy_points",),
    "fantasy_points_ppr": ("fantasy_points_ppr",),
}

# Without these the analysis cannot run at all.
REQUIRED: tuple[str, ...] = (
    "player_id",
    "player_name",
    "position",
    "season",
    "fantasy_points",
    "fantasy_points_ppr",
)

# Everything numeric that we coerce and zero-fill. Missing ones are created as 0
# so downstream arithmetic never has to guard for absence.
NUMERIC: tuple[str, ...] = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks_taken",
    "sack_fumbles",
    "sack_fumbles_lost",
    "passing_air_yards",
    "passing_yac",
    "passing_first_downs",
    "passing_2pt",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_2pt",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles",
    "receiving_fumbles_lost",
    "receiving_air_yards",
    "receiving_yac",
    "receiving_first_downs",
    "receiving_2pt",
    "special_teams_tds",
    "fantasy_points",
    "fantasy_points_ppr",
)

_ALIAS_POOL = {alias for aliases in ALIASES.values() for alias in aliases}


def resolve(columns) -> dict[str, str]:
    """Map canonical name -> the actual column present in ``columns``."""
    available = list(columns)
    found: dict[str, str] = {}
    for canonical, candidates in ALIASES.items():
        for candidate in candidates:
            if candidate in available:
                found[canonical] = candidate
                break
    return found


def missing(columns, names: tuple[str, ...] = REQUIRED) -> list[str]:
    """Canonical names in ``names`` that could not be resolved."""
    found = resolve(columns)
    return [n for n in names if n not in found]


def standardize(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Rename a raw nflverse frame to canonical column names.

    Unresolved alias columns are dropped so a rename can never produce two
    columns with the same name (e.g. both ``player_name`` and
    ``player_display_name`` mapping to ``player_name``). Columns that aren't
    aliases of anything are passed through untouched.
    """
    gaps = missing(df.columns)
    if gaps and strict:
        raise KeyError(
            f"nflverse frame is missing required columns: {gaps}. "
            f"The schema may have changed again - add an alias in "
            f"{__name__}.ALIASES. Columns present: {sorted(df.columns)[:25]}..."
        )

    chosen = resolve(df.columns)
    keep_as_is = set(chosen.values())
    losers = [c for c in df.columns if c in _ALIAS_POOL and c not in keep_as_is]

    out = df.drop(columns=losers)
    out = out.rename(columns={actual: canon for canon, actual in chosen.items()})
    return out


def coerce_numeric(df: pd.DataFrame, names: tuple[str, ...] = NUMERIC) -> pd.DataFrame:
    """Force stat columns to float, filling absent columns and NaNs with 0."""
    out = df.copy()
    for name in names:
        if name in out.columns:
            out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        else:
            out[name] = 0.0
    return out


def report(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable view of what resolved to what - handy in the notebook."""
    chosen = resolve(df.columns)
    rows = [
        {
            "canonical": canon,
            "source_column": chosen.get(canon, ""),
            "status": "ok" if canon in chosen else "MISSING",
        }
        for canon in ALIASES
    ]
    return pd.DataFrame(rows)
